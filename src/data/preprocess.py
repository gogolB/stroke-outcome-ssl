#!/usr/bin/env python3
"""
ISLES 2022 Preprocessing Pipeline

Handles the heterogeneous multi-center data by:
1. Resampling to common spacing and dimensions
2. Co-registering modalities
3. Intensity normalization
4. Cropping to remove empty space
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import nibabel as nib
from tqdm import tqdm
import torch
import hydra
from omegaconf import DictConfig
import logging

# MONAI imports
import monai
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd,
    Orientationd, ScaleIntensityRangePercentilesd, ScaleIntensityd,
    CropForegroundd, Resized, SaveImaged, CastToTyped,
    RandAffined, RandFlipd, RandScaleIntensityd
)
from monai.data import Dataset, DataLoader, MetaTensor
from monai.utils import set_determinism

# Set up logging
log = logging.getLogger(__name__)


class ISLES2022Preprocessor:
    """Preprocessing pipeline for ISLES 2022 dataset"""
    
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.root_dir = Path(cfg.data.root_dir)
        self.output_dir = Path(cfg.data.output_dir)
        self.splits_dir = Path(cfg.data.splits_dir)
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "train").mkdir(exist_ok=True)
        (self.output_dir / "val").mkdir(exist_ok=True)
        
        if cfg.processing.save_qc_images:
            self.qc_dir = self.output_dir / "qc"
            self.qc_dir.mkdir(exist_ok=True)
        
        # Set device
        self.device = torch.device(
            f"cuda:{cfg.processing.gpu_id}" if cfg.processing.use_gpu and torch.cuda.is_available() 
            else "cpu"
        )
        
        log.info(f"Using device: {self.device}")
        
    def load_splits(self) -> Tuple[List[str], List[str]]:
        """Load train/validation splits"""
        train_cases = []
        val_cases = []
        
        with open(self.splits_dir / "train.txt", 'r') as f:
            train_cases = [line.strip() for line in f.readlines()]
            
        with open(self.splits_dir / "val.txt", 'r') as f:
            val_cases = [line.strip() for line in f.readlines()]
            
        log.info(f"Loaded {len(train_cases)} training and {len(val_cases)} validation cases")
        return train_cases, val_cases
    
    def get_case_paths(self, case_id: str) -> Dict[str, str]:
        """Get file paths for all modalities of a case"""
        case_dir = self.root_dir / case_id / "ses-0001"
        
        paths = {}
        
        # FLAIR
        flair_files = list((case_dir / "anat").glob("*FLAIR*.nii.gz"))
        if flair_files:
            paths["FLAIR"] = str(flair_files[0])
            
        # DWI
        dwi_files = list((case_dir / "dwi").glob("*dwi*.nii.gz"))
        dwi_files = [f for f in dwi_files if 'adc' not in f.name.lower()]
        if dwi_files:
            paths["DWI"] = str(dwi_files[0])
            
        # ADC
        adc_files = list((case_dir / "dwi").glob("*adc*.nii.gz"))
        if adc_files:
            paths["ADC"] = str(adc_files[0])
            
        return paths
    
    def create_preprocessing_transforms(self) -> Dict[str, Compose]:
        """Create MONAI transformation pipelines for each modality"""
        transforms = {}
        
        for modality in ["FLAIR", "DWI", "ADC"]:
            transform_list = []
            
            # Load image
            transform_list.append(LoadImaged(keys=["image"]))
            transform_list.append(EnsureChannelFirstd(keys=["image"]))
            
            # Reorient to RAS
            transform_list.append(
                Orientationd(keys=["image"], axcodes="RAS")
            )
            
            # Resample to target spacing
            if self.cfg.preprocessing.resampling.enabled:
                transform_list.append(
                    Spacingd(
                        keys=["image"],
                        pixdim=self.cfg.data.target_spacing[modality],
                        mode="bilinear",
                        anti_aliasing=self.cfg.preprocessing.resampling.anti_aliasing
                    )
                )
            
            # Intensity normalization
            if self.cfg.preprocessing.normalization.enabled:
                if self.cfg.preprocessing.normalization.method == "zscore_nonzero":
                    # Z-score normalization for non-zero values
                    transform_list.append(
                        ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0)
                    )
                elif self.cfg.preprocessing.normalization.method == "percentile":
                    # Percentile-based normalization
                    transform_list.append(
                        ScaleIntensityRangePercentilesd(
                            keys=["image"],
                            lower=self.cfg.preprocessing.normalization.percentile_lower,
                            upper=self.cfg.preprocessing.normalization.percentile_upper,
                            b_min=0.0,
                            b_max=1.0,
                            clip=True
                        )
                    )
            
            # Crop foreground
            if self.cfg.preprocessing.cropping.enabled:
                transform_list.append(
                    CropForegroundd(
                        keys=["image"],
                        source_key="image",
                        margin=self.cfg.preprocessing.cropping.margin
                    )
                )
            
            # Resize to target shape
            transform_list.append(
                Resized(
                    keys=["image"],
                    spatial_size=self.cfg.data.target_shape[modality],
                    mode="trilinear"
                )
            )
            
            # Cast to target dtype
            transform_list.append(
                CastToTyped(keys=["image"], dtype=np.float32)
            )
            
            transforms[modality] = Compose(transform_list)
            
        return transforms
    
    def process_case(self, case_id: str, split: str) -> bool:
        """Process a single case"""
        try:
            # Get file paths
            paths = self.get_case_paths(case_id)
            
            if len(paths) != 3:
                log.warning(f"Case {case_id} missing modalities: found {list(paths.keys())}")
                return False
            
            # Create preprocessing transforms
            transforms = self.create_preprocessing_transforms()
            
            # Process each modality
            processed_data = {}
            
            for modality, path in paths.items():
                # Apply transforms
                data_dict = {"image": path}
                transformed = transforms[modality](data_dict)
                processed_data[modality] = transformed["image"]
            
            # Co-registration (if enabled)
            if self.cfg.preprocessing.registration.enabled:
                processed_data = self._coregister_modalities(processed_data)
            
            # Save processed data
            output_case_dir = self.output_dir / split / case_id
            output_case_dir.mkdir(parents=True, exist_ok=True)
            
            for modality, data in processed_data.items():
                output_path = output_case_dir / f"{modality}.nii.gz"
                
                # Convert MetaTensor to numpy array if needed
                if isinstance(data, MetaTensor):
                    data_np = data.cpu().numpy()
                else:
                    data_np = data
                
                # Remove channel dimension if present
                if data_np.ndim == 4 and data_np.shape[0] == 1:
                    data_np = data_np[0]
                
                # Save
                nib.save(nib.Nifti1Image(data_np, np.eye(4)), output_path)
            
            # Generate QC images
            if self.cfg.processing.save_qc_images:
                self._generate_qc_images(case_id, processed_data)
            
            return True
            
        except Exception as e:
            log.error(f"Error processing case {case_id}: {str(e)}")
            return False
    
    def _coregister_modalities(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Co-register DWI and ADC to FLAIR (placeholder for now)"""
        # TODO: Implement actual registration using MONAI or SimpleITK
        # For now, return data as-is since they're already roughly aligned
        return data
    
    def _generate_qc_images(self, case_id: str, data: Dict[str, torch.Tensor]):
        """Generate quality control images"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        
        for i, (modality, volume) in enumerate(data.items()):
            # Convert to numpy
            if isinstance(volume, torch.Tensor):
                volume = volume.cpu().numpy()
            
            if volume.ndim == 4:
                volume = volume[0]
            
            # Get slices at specified positions
            for j, slice_pos in enumerate(self.cfg.processing.qc_slices):
                slice_idx = int(volume.shape[2] * slice_pos)
                axes[i, j].imshow(volume[:, :, slice_idx].T, cmap='gray', origin='lower')
                axes[i, j].set_title(f"{modality} - Slice {slice_idx}")
                axes[i, j].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.qc_dir / f"{case_id}_qc.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def run(self):
        """Run the preprocessing pipeline"""
        log.info("Starting ISLES 2022 preprocessing pipeline")
        
        # Load splits
        train_cases, val_cases = self.load_splits()
        
        # Limit cases if dry run
        if self.cfg.processing.dry_run:
            log.info("Dry run mode: processing first 5 cases only")
            train_cases = train_cases[:3]
            val_cases = val_cases[:2]
        
        # Process training cases
        log.info("Processing training cases...")
        success_count = 0
        for case_id in tqdm(train_cases, desc="Training cases"):
            if self.process_case(case_id, "train"):
                success_count += 1
        
        log.info(f"Successfully processed {success_count}/{len(train_cases)} training cases")
        
        # Process validation cases
        log.info("Processing validation cases...")
        success_count = 0
        for case_id in tqdm(val_cases, desc="Validation cases"):
            if self.process_case(case_id, "val"):
                success_count += 1
        
        log.info(f"Successfully processed {success_count}/{len(val_cases)} validation cases")
        
        # Save preprocessing configuration
        config_path = self.output_dir / "preprocessing_config.yaml"
        with open(config_path, 'w') as f:
            f.write(hydra.core.hydra_config.HydraConfig.get().job.override_dirname)
        
        log.info(f"Preprocessing complete! Output saved to: {self.output_dir}")


@hydra.main(config_path="../../configs", config_name="preprocessing", version_base=None)
def main(cfg: DictConfig):
    """Main preprocessing entry point"""
    # Set random seed for reproducibility
    set_determinism(seed=42)
    
    # Create and run preprocessor
    preprocessor = ISLES2022Preprocessor(cfg)
    preprocessor.run()


if __name__ == "__main__":
    main()