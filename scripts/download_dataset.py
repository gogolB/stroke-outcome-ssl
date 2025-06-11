#!/usr/bin/env python3
"""
ISLES 2022 Dataset Download and Organization Script

This script automatically downloads and organizes the ISLES 2022 dataset from Zenodo.
It supports resume functionality if the download is interrupted.
"""

import os
import zipfile
import hashlib
import shutil
from pathlib import Path
from typing import Optional
import requests
from tqdm import tqdm
import json
import sys

class ISLES2022Downloader:
    def __init__(self, data_root: str = "./data"):
        self.data_root = Path(data_root)
        self.raw_dir = self.data_root / "ISLES2022" / "raw"
        self.processed_dir = self.data_root / "ISLES2022" / "processed"
        
        # Create directory structure
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
    def download_from_url(self, url: str, filename: str, expected_hash: Optional[str] = None):
        """Download file with progress bar, resume capability, and optional hash verification"""
        filepath = self.raw_dir / filename
        
        # Check if file exists and is complete
        headers = requests.head(url)
        total_size = int(headers.headers.get('content-length', 0))
        
        if filepath.exists():
            existing_size = filepath.stat().st_size
            if existing_size == total_size:
                print(f"File {filename} already completely downloaded.")
                if expected_hash:
                    self._verify_hash(filepath, expected_hash)
                return filepath
            elif existing_size < total_size:
                print(f"Resuming download of {filename} from {existing_size / 1e6:.1f} MB...")
                resume_header = {'Range': f'bytes={existing_size}-'}
            else:
                print(f"File {filename} seems corrupted. Re-downloading...")
                filepath.unlink()
                resume_header = {}
                existing_size = 0
        else:
            print(f"Downloading {filename} ({total_size / 1e6:.1f} MB)...")
            resume_header = {}
            existing_size = 0
        
        # Download with resume support
        response = requests.get(url, headers=resume_header, stream=True)
        
        # Open file in append mode if resuming, write mode if new
        mode = 'ab' if existing_size > 0 else 'wb'
        
        with open(filepath, mode) as f:
            with tqdm(total=total_size, initial=existing_size, unit='iB', unit_scale=True) as pbar:
                for chunk in response.iter_content(chunk_size=1024*1024):  # 1MB chunks
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        print("✓ Download complete!")
        
        if expected_hash:
            self._verify_hash(filepath, expected_hash)
        
        return filepath
    
    def _verify_hash(self, filepath: Path, expected_hash: str):
        """Verify file integrity using MD5 hash"""
        print(f"Verifying integrity of {filepath.name}...")
        file_hash = hashlib.md5()
        
        # Read in chunks to handle large files
        with open(filepath, 'rb') as f:
            with tqdm(total=filepath.stat().st_size, unit='iB', unit_scale=True, desc="Verifying") as pbar:
                for chunk in iter(lambda: f.read(1024*1024), b""):
                    file_hash.update(chunk)
                    pbar.update(len(chunk))
        
        if file_hash.hexdigest() != expected_hash:
            raise ValueError(f"Hash mismatch for {filepath.name}")
        print("✓ Integrity verified")
    
    def extract_dataset(self, archive_path: Path):
        """Extract the dataset archive"""
        print(f"Extracting {archive_path.name}...")
        
        extract_dir = self.raw_dir / "extracted"
        extract_dir.mkdir(exist_ok=True)
        
        if archive_path.suffix == '.zip':
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                # Extract with progress bar
                members = zip_ref.namelist()
                with tqdm(total=len(members), desc="Extracting files") as pbar:
                    for member in members:
                        zip_ref.extract(member, extract_dir)
                        pbar.update(1)
        else:
            # Handle .tar.gz or other formats
            import tarfile
            with tarfile.open(archive_path, 'r:gz') as tar:
                tar.extractall(extract_dir)
        
        print("✓ Extraction complete")
        return extract_dir
    
    def organize_dataset(self, extracted_dir: Path):
        """Organize dataset into consistent structure"""
        print("Organizing dataset structure...")
        
        # Create organized structure
        organized_dir = self.processed_dir / "BIDS"
        organized_dir.mkdir(exist_ok=True)
        
        # Dataset info file
        dataset_info = {
            "name": "ISLES 2022",
            "modalities": ["FLAIR", "DWI", "ADC"],
            "num_total_cases": 250,  # Only training data is available for download
            "num_train_split": 200,  # We'll create our own train split
            "num_val_split": 50,     # We'll create our own validation split
            "format": "BIDS",
            "note": "Test set (150 cases) is held private by challenge organizers"
        }
        
        with open(organized_dir / "dataset_info.json", 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        # Count and organize files
        case_count = 0
        modality_count = {"FLAIR": 0, "DWI": 0, "ADC": 0}
        
        # Walk through extracted directory and organize
        for root, dirs, files in os.walk(extracted_dir):
            for file in files:
                if file.endswith('.nii.gz'):
                    src_path = Path(root) / file
                    
                    # Parse filename to determine case and modality
                    if 'FLAIR' in file:
                        modality = 'FLAIR'
                    elif 'dwi' in file.lower() and 'adc' not in file.lower():
                        modality = 'DWI'
                    elif 'adc' in file.lower():
                        modality = 'ADC'
                    else:
                        continue
                    
                    # Extract case ID
                    if 'sub-strokecase' in file:
                        case_id = file.split('sub-strokecase')[1].split('_')[0]
                        
                        # Create case directory
                        case_dir = organized_dir / f"sub-strokecase{case_id}" / "ses-0001"
                        
                        if modality == 'FLAIR':
                            dest_dir = case_dir / "anat"
                        else:
                            dest_dir = case_dir / "dwi"
                        
                        dest_dir.mkdir(parents=True, exist_ok=True)
                        
                        # Copy file
                        dest_path = dest_dir / file
                        if not dest_path.exists():
                            shutil.copy2(src_path, dest_path)
                            modality_count[modality] += 1
        
        # Summary statistics
        print("\n✓ Dataset organization complete!")
        print(f"Total modality files found:")
        for mod, count in modality_count.items():
            print(f"  - {mod}: {count} files")
        
        return organized_dir
    
    def verify_dataset(self, dataset_dir: Path):
        """Verify dataset completeness"""
        print("\nVerifying dataset completeness...")
        
        issues = []
        case_dirs = sorted([d for d in dataset_dir.iterdir() if d.is_dir() and 'sub-strokecase' in d.name])
        
        print(f"Found {len(case_dirs)} cases")
        
        for case_dir in case_dirs:
            case_id = case_dir.name
            
            # Check for all modalities
            flair_path = case_dir / "ses-0001" / "anat"
            dwi_path = case_dir / "ses-0001" / "dwi"
            
            if not flair_path.exists() or not any(flair_path.glob("*FLAIR*.nii.gz")):
                issues.append(f"{case_id}: Missing FLAIR")
            
            if not dwi_path.exists():
                issues.append(f"{case_id}: Missing DWI directory")
            else:
                if not any(dwi_path.glob("*dwi*.nii.gz")):
                    issues.append(f"{case_id}: Missing DWI")
                if not any(dwi_path.glob("*adc*.nii.gz")):
                    issues.append(f"{case_id}: Missing ADC")
        
        if issues:
            print(f"\n⚠ Found {len(issues)} issues:")
            for issue in issues[:10]:  # Show first 10 issues
                print(f"  - {issue}")
            if len(issues) > 10:
                print(f"  ... and {len(issues) - 10} more")
        else:
            print("✓ All cases have complete modalities!")
        
        return len(issues) == 0
    
    def create_train_val_split(self, dataset_dir: Path, val_ratio: float = 0.2, random_seed: int = 42):
        """Create train/validation split from available training data"""
        import random
        
        print(f"\nCreating train/validation split (val_ratio={val_ratio})...")
        
        # Get all case directories
        case_dirs = sorted([d for d in dataset_dir.iterdir() if d.is_dir() and 'sub-strokecase' in d.name])
        
        # Randomly split cases
        random.seed(random_seed)
        random.shuffle(case_dirs)
        
        num_val = int(len(case_dirs) * val_ratio)
        val_cases = case_dirs[:num_val]
        train_cases = case_dirs[num_val:]
        
        # Create split files
        splits_dir = self.processed_dir / "splits"
        splits_dir.mkdir(exist_ok=True)
        
        # Write train split
        with open(splits_dir / "train.txt", 'w') as f:
            for case in train_cases:
                f.write(f"{case.name}\n")
        
        # Write validation split
        with open(splits_dir / "val.txt", 'w') as f:
            for case in val_cases:
                f.write(f"{case.name}\n")
        
        # Create split info
        split_info = {
            "random_seed": random_seed,
            "val_ratio": val_ratio,
            "num_train": len(train_cases),
            "num_val": len(val_cases),
            "total_cases": len(case_dirs)
        }
        
        with open(splits_dir / "split_info.json", 'w') as f:
            json.dump(split_info, f, indent=2)
        
        print(f"✓ Created splits:")
        print(f"  - Training: {len(train_cases)} cases")
        print(f"  - Validation: {len(val_cases)} cases")
        print(f"  - Split files saved to: {splits_dir}")
        
        return train_cases, val_cases


def main():
    """Main download workflow"""
    print("ISLES 2022 Dataset Setup")
    print("=" * 50)
    
    # Initialize downloader
    downloader = ISLES2022Downloader(data_root="./data")
    
    # Dataset information
    dataset_files = [
        {
            "url": "https://zenodo.org/records/7960856/files/ISLES-2022.zip?download=1",
            "filename": "ISLES-2022.zip",
            "md5": "302ee280373cdd5c190ab763d72a7a50",
            "description": "Main dataset (250 training cases)"
        },
        {
            "url": "https://zenodo.org/records/7960856/files/center_ids.xlsx?download=1",
            "filename": "center_ids.xlsx",
            "md5": "173438275ea63e1df62793940feaa638",
            "description": "Center IDs for each case"
        }
    ]
    
    print("\nDataset Information:")
    print("- Source: Zenodo (DOI: 10.5281/zenodo.7960856)")
    print("- Available for download: 250 training cases + center metadata")
    print("- Test set: 150 cases (held private for challenge evaluation)")
    print("- We'll create our own 80/20 train/val split from the 250 cases")
    print()
    
    try:
        # Step 1: Download all dataset files
        downloaded_files = []
        for file_info in dataset_files:
            print(f"\n--- Downloading: {file_info['description']} ---")
            file_path = downloader.download_from_url(
                file_info["url"], 
                file_info["filename"], 
                file_info["md5"]
            )
            downloaded_files.append(file_path)
        
        # Step 2: Extract main dataset
        archive_path = downloaded_files[0]  # ISLES-2022.zip
        extracted_dir = downloader.extract_dataset(archive_path)
        
        # Step 3: Copy center IDs file to processed directory
        center_ids_src = downloaded_files[1]  # center_ids.xlsx
        center_ids_dst = downloader.processed_dir / "metadata" / "center_ids.xlsx"
        center_ids_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(center_ids_src, center_ids_dst)
        print(f"\n✓ Copied center IDs to: {center_ids_dst}")
        
        # Step 4: Organize into BIDS structure
        organized_dir = downloader.organize_dataset(extracted_dir)
        
        # Step 5: Verify completeness
        is_complete = downloader.verify_dataset(organized_dir)
        
        # Step 6: Create train/validation split
        if is_complete:
            train_cases, val_cases = downloader.create_train_val_split(
                organized_dir, 
                val_ratio=0.2,  # 80/20 split: 200 train, 50 val
                random_seed=42
            )
            
            # Optional: Load and display center distribution
            try:
                import pandas as pd
                center_df = pd.read_excel(center_ids_dst)
                print(f"\n✓ Center IDs loaded: {len(center_df)} entries")
                if 'center' in center_df.columns or 'Center' in center_df.columns:
                    center_col = 'center' if 'center' in center_df.columns else 'Center'
                    print(f"  Center distribution:")
                    print(center_df[center_col].value_counts())
            except ImportError:
                print("\n  Note: Install pandas and openpyxl to analyze center distribution")
                print("  pip install pandas openpyxl")
            except Exception as e:
                print(f"\n  Could not analyze center IDs: {e}")
            
            print("\n" + "=" * 50)
            print("✓ Dataset setup complete!")
            print(f"✓ Data location: {downloader.processed_dir}")
            print(f"✓ Center IDs: {center_ids_dst}")
            print("✓ Ready for preprocessing pipeline")
        else:
            print("\n⚠ Dataset verification failed. Please check the extracted files.")
            
    except KeyboardInterrupt:
        print("\n\n⚠ Download interrupted. Run the script again to resume.")
        print("The download will automatically resume from where it left off.")
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        print("Please check your internet connection and try again.")


if __name__ == "__main__":
    main()