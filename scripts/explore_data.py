#!/usr/bin/env python3
"""
ISLES 2022 Data Exploration Script

This script explores the downloaded ISLES 2022 dataset to understand:
- Data structure and organization
- Image dimensions and voxel spacing
- Intensity distributions
- Center-specific characteristics
"""

import os
import json
import nibabel as nib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from collections import defaultdict

class ISLES2022Explorer:
    def __init__(self, data_root: str = "./data"):
        self.data_root = Path(data_root)
        self.bids_dir = self.data_root / "ISLES2022" / "processed" / "BIDS"
        self.metadata_dir = self.data_root / "ISLES2022" / "processed" / "metadata"
        self.splits_dir = self.data_root / "ISLES2022" / "processed" / "splits"
        
        # Results storage
        self.results = {
            "cases": [],
            "dimensions": defaultdict(list),
            "voxel_sizes": defaultdict(list),
            "intensity_stats": defaultdict(dict),
            "file_sizes": defaultdict(list),
            "center_info": {}
        }
        
    def load_center_ids(self):
        """Load center IDs from Excel file"""
        center_ids_path = self.metadata_dir / "center_ids.xlsx"
        if center_ids_path.exists():
            try:
                df = pd.read_excel(center_ids_path)
                print(f"✓ Loaded center IDs: {len(df)} entries")
                print(f"  Columns: {list(df.columns)}")
                return df
            except Exception as e:
                print(f"⚠ Could not load center IDs: {e}")
                return None
        else:
            print("⚠ center_ids.xlsx not found")
            return None
    
    def load_splits(self):
        """Load train/val split information"""
        train_path = self.splits_dir / "train.txt"
        val_path = self.splits_dir / "val.txt"
        
        train_cases = []
        val_cases = []
        
        if train_path.exists():
            with open(train_path, 'r') as f:
                train_cases = [line.strip() for line in f.readlines()]
                
        if val_path.exists():
            with open(val_path, 'r') as f:
                val_cases = [line.strip() for line in f.readlines()]
                
        print(f"✓ Loaded splits: {len(train_cases)} train, {len(val_cases)} val")
        return train_cases, val_cases
    
    def analyze_case(self, case_dir: Path) -> Dict:
        """Analyze a single case"""
        case_id = case_dir.name
        case_info = {"id": case_id, "modalities": {}}
        
        # Expected modalities and their paths
        modality_paths = {
            "FLAIR": case_dir / "ses-0001" / "anat",
            "DWI": case_dir / "ses-0001" / "dwi",
            "ADC": case_dir / "ses-0001" / "dwi"
        }
        
        for modality, path in modality_paths.items():
            if not path.exists():
                continue
                
            # Find the appropriate file
            if modality == "FLAIR":
                files = list(path.glob("*FLAIR*.nii.gz"))
            elif modality == "DWI":
                files = list(path.glob("*dwi*.nii.gz"))
                files = [f for f in files if 'adc' not in f.name.lower()]
            elif modality == "ADC":
                files = list(path.glob("*adc*.nii.gz"))
            
            if files:
                nii_path = files[0]
                
                # Load image
                img = nib.load(nii_path)
                data = img.get_fdata()
                
                # Extract information
                info = {
                    "path": str(nii_path),
                    "shape": data.shape,
                    "voxel_size": img.header.get_zooms()[:3],
                    "dtype": str(data.dtype),
                    "file_size_mb": nii_path.stat().st_size / 1e6,
                    "intensity_stats": {
                        "min": float(np.min(data)),
                        "max": float(np.max(data)),
                        "mean": float(np.mean(data)),
                        "std": float(np.std(data)),
                        "percentiles": {
                            "1": float(np.percentile(data, 1)),
                            "5": float(np.percentile(data, 5)),
                            "95": float(np.percentile(data, 95)),
                            "99": float(np.percentile(data, 99))
                        }
                    }
                }
                
                case_info["modalities"][modality] = info
                
                # Store aggregate statistics
                self.results["dimensions"][modality].append(info["shape"])
                self.results["voxel_sizes"][modality].append(info["voxel_size"])
                self.results["file_sizes"][modality].append(info["file_size_mb"])
                
        return case_info
    
    def explore_dataset(self):
        """Explore the entire dataset"""
        print("\nExploring ISLES 2022 Dataset...")
        print("=" * 50)
        
        # Load center IDs
        center_df = self.load_center_ids()
        
        # Load splits
        train_cases, val_cases = self.load_splits()
        
        # Get all cases
        case_dirs = sorted([d for d in self.bids_dir.iterdir() 
                           if d.is_dir() and 'sub-strokecase' in d.name])
        
        print(f"\nFound {len(case_dirs)} cases")
        
        # Analyze each case
        all_case_info = []
        for case_dir in tqdm(case_dirs, desc="Analyzing cases"):
            case_info = self.analyze_case(case_dir)
            case_info["split"] = "train" if case_dir.name in train_cases else "val"
            all_case_info.append(case_info)
        
        self.results["cases"] = all_case_info
        
        # Compute aggregate statistics
        self._compute_aggregate_stats()
        
        # Add center information if available
        if center_df is not None:
            self._analyze_center_distribution(center_df, train_cases, val_cases)
        
        return self.results
    
    def _compute_aggregate_stats(self):
        """Compute aggregate statistics across all cases"""
        print("\n" + "=" * 50)
        print("Dataset Statistics")
        print("=" * 50)
        
        # Modality availability
        modality_counts = defaultdict(int)
        complete_cases = 0
        
        for case in self.results["cases"]:
            if len(case["modalities"]) == 3:
                complete_cases += 1
            for mod in case["modalities"]:
                modality_counts[mod] += 1
        
        print(f"\n✓ Complete cases (all 3 modalities): {complete_cases}/{len(self.results['cases'])}")
        print("\n✓ Modality counts:")
        for mod, count in modality_counts.items():
            print(f"  - {mod}: {count}")
        
        # Dimension statistics
        print("\n✓ Image dimensions:")
        for modality in ["FLAIR", "DWI", "ADC"]:
            if modality in self.results["dimensions"]:
                dims = self.results["dimensions"][modality]
                unique_dims = list(set([str(d) for d in dims]))
                print(f"  - {modality}: {len(unique_dims)} unique dimensions")
                for dim_str in unique_dims[:3]:  # Show first 3
                    count = sum(1 for d in dims if str(d) == dim_str)
                    print(f"      {dim_str}: {count} cases")
                if len(unique_dims) > 3:
                    print(f"      ... and {len(unique_dims) - 3} more")
        
        # Voxel size statistics
        print("\n✓ Voxel sizes (mm):")
        for modality in ["FLAIR", "DWI", "ADC"]:
            if modality in self.results["voxel_sizes"]:
                voxels = self.results["voxel_sizes"][modality]
                unique_voxels = list(set([str(v) for v in voxels]))
                print(f"  - {modality}: {len(unique_voxels)} unique voxel sizes")
                for vox_str in unique_voxels[:3]:
                    count = sum(1 for v in voxels if str(v) == vox_str)
                    print(f"      {vox_str}: {count} cases")
        
        # File size statistics
        print("\n✓ File sizes (MB):")
        for modality in ["FLAIR", "DWI", "ADC"]:
            if modality in self.results["file_sizes"]:
                sizes = self.results["file_sizes"][modality]
                print(f"  - {modality}: {np.mean(sizes):.1f} ± {np.std(sizes):.1f} MB")
    
    def _analyze_center_distribution(self, center_df, train_cases, val_cases):
        """Analyze distribution of cases across centers"""
        print("\n✓ Center Distribution:")
        
        # Map case IDs to centers
        if 'Case' in center_df.columns and 'Center' in center_df.columns:
            center_map = dict(zip(center_df['Case'], center_df['Center']))
            
            train_centers = []
            val_centers = []
            
            for case_id in train_cases:
                # Extract case number from ID
                case_num = int(case_id.replace('sub-strokecase', ''))
                if case_num in center_map:
                    train_centers.append(center_map[case_num])
                    
            for case_id in val_cases:
                case_num = int(case_id.replace('sub-strokecase', ''))
                if case_num in center_map:
                    val_centers.append(center_map[case_num])
            
            # Print distribution
            train_dist = pd.Series(train_centers).value_counts()
            val_dist = pd.Series(val_centers).value_counts()
            
            print("\n  Training set:")
            for center, count in train_dist.items():
                print(f"    Center {center}: {count} cases ({count/len(train_centers)*100:.1f}%)")
                
            print("\n  Validation set:")
            for center, count in val_dist.items():
                print(f"    Center {center}: {count} cases ({count/len(val_centers)*100:.1f}%)")
    
    def save_results(self):
        """Save exploration results"""
        output_dir = self.data_root / "ISLES2022" / "processed" / "exploration"
        output_dir.mkdir(exist_ok=True)
        
        # Save detailed results as JSON
        output_path = output_dir / "dataset_exploration.json"
        with open(output_path, 'w') as f:
            # Convert numpy types for JSON serialization
            json_results = {
                "num_cases": len(self.results["cases"]),
                "modality_counts": {k: len(v) for k, v in self.results["dimensions"].items()},
                "summary": "See printed output for detailed statistics"
            }
            json.dump(json_results, f, indent=2)
        
        print(f"\n✓ Results saved to: {output_path}")
        
        # Create visualization plots
        self._create_visualizations(output_dir)
    
    def _create_visualizations(self, output_dir):
        """Create visualization plots"""
        print("\n✓ Creating visualizations...")
        
        # File size distribution
        plt.figure(figsize=(10, 6))
        for i, modality in enumerate(["FLAIR", "DWI", "ADC"]):
            if modality in self.results["file_sizes"]:
                plt.subplot(1, 3, i+1)
                plt.hist(self.results["file_sizes"][modality], bins=20, alpha=0.7)
                plt.xlabel("File Size (MB)")
                plt.ylabel("Count")
                plt.title(f"{modality} File Sizes")
        
        plt.tight_layout()
        plt.savefig(output_dir / "file_size_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  - Saved file size distribution plot")


def main():
    """Main exploration workflow"""
    print("ISLES 2022 Data Exploration")
    print("=" * 50)
    
    # Initialize explorer
    explorer = ISLES2022Explorer(data_root="./data")
    
    # Check if data exists
    if not explorer.bids_dir.exists():
        print(f"❌ Data directory not found: {explorer.bids_dir}")
        print("Please run the download script first.")
        return
    
    # Explore dataset
    results = explorer.explore_dataset()
    
    # Save results
    explorer.save_results()
    
    print("\n✓ Exploration complete!")
    print("\nKey findings to note for preprocessing:")
    print("- Check for varying image dimensions across cases")
    print("- Note different voxel sizes between modalities/centers")
    print("- Consider intensity normalization strategies")
    print("- Account for center-specific variations")


if __name__ == "__main__":
    main()
