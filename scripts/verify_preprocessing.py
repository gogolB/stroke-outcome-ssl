#!/usr/bin/env python3
"""
Verify preprocessed data - check dimensions, spacing, and quality
"""

import os
import nibabel as nib
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def verify_preprocessing(output_dir="./data/ISLES2022/processed/preprocessed"):
    """Verify the preprocessed data"""
    output_path = Path(output_dir)
    
    print("Verifying Preprocessing Output")
    print("=" * 50)
    
    # Check directory structure
    if not output_path.exists():
        print(f"❌ Output directory not found: {output_path}")
        return
    
    # Check for train/val splits
    train_dir = output_path / "train"
    val_dir = output_path / "val"
    
    print(f"\n✓ Output directory: {output_path}")
    print(f"  Train cases: {len(list(train_dir.glob('sub-*')))} found")
    print(f"  Val cases: {len(list(val_dir.glob('sub-*')))} found")
    
    # Check a sample case
    sample_cases = list(train_dir.glob('sub-*'))[:1] + list(val_dir.glob('sub-*'))[:1]
    
    for case_dir in sample_cases:
        print(f"\n📁 Checking case: {case_dir.name}")
        
        # Check each modality
        for modality in ["FLAIR", "DWI", "ADC"]:
            modality_file = case_dir / f"{modality}.nii.gz"
            
            if modality_file.exists():
                # Load the image
                img = nib.load(modality_file)
                data = img.get_fdata()
                
                print(f"\n  {modality}:")
                print(f"    Shape: {data.shape}")
                print(f"    Spacing: {img.header.get_zooms()[:3]}")
                print(f"    Data type: {data.dtype}")
                print(f"    Value range: [{data.min():.3f}, {data.max():.3f}]")
                print(f"    File size: {modality_file.stat().st_size / 1e6:.1f} MB")
                
                # Check expected dimensions
                expected_shapes = {
                    "FLAIR": (192, 224, 192),
                    "DWI": (96, 112, 96),
                    "ADC": (96, 112, 96)
                }
                
                if data.shape == expected_shapes[modality]:
                    print(f"    ✓ Shape matches expected: {expected_shapes[modality]}")
                else:
                    print(f"    ⚠️  Shape differs from expected: {expected_shapes[modality]}")
            else:
                print(f"\n  ❌ {modality} not found!")
    
    # Check QC images
    qc_dir = output_path / "qc"
    if qc_dir.exists():
        qc_files = list(qc_dir.glob("*.png"))
        print(f"\n✓ QC images found: {len(qc_files)} files")
        if qc_files:
            print(f"  Example: {qc_files[0].name}")
    else:
        print("\n⚠️  No QC images directory found")
    
    # Create a simple visualization of one case
    if sample_cases:
        print("\n📊 Creating visualization of first case...")
        case_dir = sample_cases[0]
        
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        for i, modality in enumerate(["FLAIR", "DWI", "ADC"]):
            modality_file = case_dir / f"{modality}.nii.gz"
            if modality_file.exists():
                img = nib.load(modality_file)
                data = img.get_fdata()
                
                # Show middle slice
                mid_slice = data.shape[2] // 2
                axes[i].imshow(data[:, :, mid_slice].T, cmap='gray', origin='lower')
                axes[i].set_title(f"{modality}\nShape: {data.shape}")
                axes[i].axis('off')
        
        plt.tight_layout()
        output_fig = output_path / "sample_preprocessing.png"
        plt.savefig(output_fig, dpi=150, bbox_inches='tight')
        print(f"  Saved to: {output_fig}")
        plt.close()
    
    print("\n" + "=" * 50)
    print("Preprocessing verification complete!")
    print("\nIf everything looks good, run full preprocessing with:")
    print("  python src/data/preprocess.py processing.dry_run=false processing.use_gpu=false")


if __name__ == "__main__":
    verify_preprocessing()