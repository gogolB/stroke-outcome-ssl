"""
Data module for BYOL training on ISLES 2022 dataset

Handles loading preprocessed data and applying augmentations
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import nibabel as nib

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, RandSpatialCropd,
    RandFlipd, RandRotate90d, RandScaleIntensityd, RandShiftIntensityd,
    RandAdjustContrastd, RandGaussianNoised, RandGaussianSmoothd,
    RandAffined, ToTensord, EnsureTyped, ConcatItemsd, Resized,
    CenterSpatialCropd
)
from monai.data import CacheDataset, DataLoader as MonaiDataLoader


class ISLESBYOLDataset(Dataset):
    """ISLES dataset for BYOL training"""
    
    def __init__(
        self,
        root_dir: str,
        split: str,
        modalities: List[str],
        transform: Optional[Callable] = None,
        cache_rate: float = 0.0
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.modalities = modalities
        self.transform = transform
        
        # Load case IDs
        self.cases = sorted([
            d.name for d in (self.root_dir / split).iterdir() 
            if d.is_dir() and d.name.startswith('sub-')
        ])
        
        # Create data dictionaries
        self.data_dicts = []
        for case_id in self.cases:
            case_dict = {"case_id": case_id}
            
            # Add paths for each modality
            for modality in modalities:
                modality_path = self.root_dir / split / case_id / f"{modality}.nii.gz"
                if modality_path.exists():
                    case_dict[modality] = str(modality_path)
                else:
                    print(f"Warning: {modality} not found for {case_id}")
                    
            # Only add if all modalities exist
            if len(case_dict) == len(modalities) + 1:  # +1 for case_id
                self.data_dicts.append(case_dict)
                
        print(f"Loaded {len(self.data_dicts)} cases for {split} split")
        
    def __len__(self):
        return len(self.data_dicts)
    
    def __getitem__(self, idx):
        data_dict = self.data_dicts[idx].copy()
        
        if self.transform:
            # Apply transform twice for two views
            view_1 = self.transform(data_dict.copy())
            view_2 = self.transform(data_dict.copy())
            
            return {
                'view_1': view_1['image'],
                'view_2': view_2['image'],
                'case_id': data_dict['case_id']
            }
        else:
            return data_dict


class ISLESBYOLDataModule(pl.LightningDataModule):
    """PyTorch Lightning data module for BYOL training"""
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.train_transform = None
        self.val_transform = None
        
    def setup(self, stage: Optional[str] = None):
        """Set up transforms and datasets"""
        
        # Create augmentation pipeline for BYOL
        self.train_transform = self._create_train_transforms()
        self.val_transform = self._create_val_transforms()
        
        # Create datasets
        if stage == "fit" or stage is None:
            self.train_dataset = ISLESBYOLDataset(
                root_dir=self.cfg.data.root_dir,
                split="train",
                modalities=self.cfg.data.modalities,
                transform=self.train_transform,
                cache_rate=self.cfg.data.cache_rate
            )
            
            self.val_dataset = ISLESBYOLDataset(
                root_dir=self.cfg.data.root_dir,
                split="val",
                modalities=self.cfg.data.modalities,
                transform=self.val_transform,
                cache_rate=self.cfg.data.cache_rate
            )
    
    def _create_train_transforms(self):
        """Create training augmentation pipeline"""
        transforms = []
        
        # Load images
        transforms.append(LoadImaged(keys=self.cfg.data.modalities))
        transforms.append(EnsureChannelFirstd(keys=self.cfg.data.modalities))
        
        # Concatenate modalities (early fusion)
        if self.cfg.data.modality_fusion == "early":
            # Resize all modalities to the same size before concatenation
            # Use the FLAIR size as target
            target_size = [192, 224, 192]
            transforms.append(
                Resized(
                    keys=self.cfg.data.modalities,
                    spatial_size=target_size,
                    mode="trilinear"
                )
            )
            transforms.append(ConcatItemsd(keys=self.cfg.data.modalities, name="image", dim=0))
        
        # Spatial augmentations
        if self.cfg.augmentation.spatial.random_crop.enabled:
            # Use random spatial crop
            crop_size = self.cfg.augmentation.spatial.random_crop.size
            transforms.append(
                RandSpatialCropd(
                    keys=["image"],
                    roi_size=crop_size,
                    random_center=True,
                    random_size=False
                )
            )
        
        if self.cfg.augmentation.spatial.random_flip.enabled:
            transforms.append(
                RandFlipd(
                    keys=["image"],
                    prob=self.cfg.augmentation.spatial.random_flip.prob,
                    spatial_axis=self.cfg.augmentation.spatial.random_flip.axes
                )
            )
        
        if self.cfg.augmentation.spatial.random_rotate.enabled:
            angle_range = self.cfg.augmentation.spatial.random_rotate.angle_range
            angle_rad = [np.deg2rad(a) for a in angle_range]
            transforms.append(
                RandAffined(
                    keys=["image"],
                    prob=self.cfg.augmentation.spatial.random_rotate.prob,
                    rotate_range=[angle_rad, angle_rad, angle_rad],
                    translate_range=None,
                    scale_range=None,
                    mode="bilinear",
                    padding_mode="zeros"
                )
            )
        
        # Intensity augmentations
        if self.cfg.augmentation.intensity.random_brightness.enabled:
            transforms.append(
                RandShiftIntensityd(
                    keys=["image"],
                    prob=self.cfg.augmentation.intensity.random_brightness.prob,
                    offsets=self.cfg.augmentation.intensity.random_brightness.brightness_range
                )
            )
        
        if self.cfg.augmentation.intensity.random_contrast.enabled:
            transforms.append(
                RandAdjustContrastd(
                    keys=["image"],
                    prob=self.cfg.augmentation.intensity.random_contrast.prob,
                    gamma=self.cfg.augmentation.intensity.random_contrast.contrast_range
                )
            )
        
        if self.cfg.augmentation.intensity.gaussian_noise.enabled:
            transforms.append(
                RandGaussianNoised(
                    keys=["image"],
                    prob=self.cfg.augmentation.intensity.gaussian_noise.prob,
                    mean=0.0,
                    std=self.cfg.augmentation.intensity.gaussian_noise.std
                )
            )
        
        # Medical-specific augmentations
        if self.cfg.augmentation.medical.elastic_deformation.enabled:
            transforms.append(
                RandAffined(
                    keys=["image"],
                    prob=self.cfg.augmentation.medical.elastic_deformation.prob,
                    mode="bilinear",
                    padding_mode="zeros",
                    spatial_size=None,
                    translate_range=[10, 10, 10],  # Small translations
                    rotate_range=None,
                    scale_range=[0.1, 0.1, 0.1],  # Small scaling
                    shear_range=None
                )
            )
        
        # Ensure tensor output
        transforms.append(EnsureTyped(keys=["image"], data_type="tensor"))
        
        return Compose(transforms)
    
    def _create_val_transforms(self):
        """Create validation transforms (minimal augmentation)"""
        transforms = []
        
        # Load images
        transforms.append(LoadImaged(keys=self.cfg.data.modalities))
        transforms.append(EnsureChannelFirstd(keys=self.cfg.data.modalities))
        
        # Concatenate modalities
        if self.cfg.data.modality_fusion == "early":
            # Resize all modalities to the same size before concatenation
            target_size = [192, 224, 192]
            transforms.append(
                Resized(
                    keys=self.cfg.data.modalities,
                    spatial_size=target_size,
                    mode="trilinear"
                )
            )
            transforms.append(ConcatItemsd(keys=self.cfg.data.modalities, name="image", dim=0))
        
        # Center crop to match training size
        if self.cfg.augmentation.spatial.random_crop.enabled:
            crop_size = self.cfg.augmentation.spatial.random_crop.size
            transforms.append(
                CenterSpatialCropd(
                    keys=["image"],
                    roi_size=crop_size
                )
            )
        
        # Light augmentations for validation
        transforms.append(
            RandFlipd(
                keys=["image"],
                prob=0.5,
                spatial_axis=[0]  # Only flip along one axis
            )
        )
        
        transforms.append(EnsureTyped(keys=["image"], data_type="tensor"))
        
        return Compose(transforms)
    
    def train_dataloader(self):
        """Create training dataloader"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.data.batch_size,
            shuffle=True,
            num_workers=self.cfg.data.num_workers,
            pin_memory=self.cfg.data.pin_memory,
            persistent_workers=self.cfg.data.persistent_workers,
            drop_last=True  # Important for batch norm
        )
    
    def val_dataloader(self):
        """Create validation dataloader"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.data.batch_size,
            shuffle=False,
            num_workers=self.cfg.data.num_workers,
            pin_memory=self.cfg.data.pin_memory,
            persistent_workers=self.cfg.data.persistent_workers
        )


# Test the data module
if __name__ == "__main__":
    from omegaconf import OmegaConf
    
    # Load config
    cfg = OmegaConf.load("configs/byol.yaml")
    
    # Create data module
    dm = ISLESBYOLDataModule(cfg)
    dm.setup()
    
    # Test loading a batch
    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))
    
    print(f"Batch keys: {batch.keys()}")
    print(f"View 1 shape: {batch['view_1'].shape}")
    print(f"View 2 shape: {batch['view_2'].shape}")
    print(f"Case IDs: {batch['case_id']}")