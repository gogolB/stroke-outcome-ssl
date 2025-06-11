#!/usr/bin/env python3
"""
Test BYOL setup before training

This script verifies:
1. Data loading works correctly
2. Model can be created
3. Forward pass works
4. GPU memory usage is reasonable
"""

import sys
from pathlib import Path
import torch
import numpy as np
from omegaconf import OmegaConf
import gc

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.byol import BYOL
from src.data.byol_datamodule import ISLESBYOLDataModule


def test_data_loading(cfg):
    """Test data loading and augmentations"""
    print("\n1. Testing Data Loading")
    print("=" * 50)
    
    # Create data module
    dm = ISLESBYOLDataModule(cfg)
    dm.setup()
    
    # Check dataset sizes
    print(f"Training samples: {len(dm.train_dataset)}")
    print(f"Validation samples: {len(dm.val_dataset)}")
    
    # Load a batch
    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))
    
    print(f"\nBatch contents:")
    print(f"  - View 1 shape: {batch['view_1'].shape}")
    print(f"  - View 2 shape: {batch['view_2'].shape}")
    print(f"  - Data type: {batch['view_1'].dtype}")
    print(f"  - Value range: [{batch['view_1'].min():.3f}, {batch['view_1'].max():.3f}]")
    
    return True


def test_model_creation(cfg):
    """Test model creation and architecture"""
    print("\n2. Testing Model Creation")
    print("=" * 50)
    
    # Create model
    model = BYOL(cfg)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model created successfully!")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")
    print(f"  - Encoder: {cfg.model.encoder.name}")
    print(f"  - Features dimension: {model.features_dim}")
    
    return model


def test_forward_pass(model, cfg):
    """Test forward pass and loss computation"""
    print("\n3. Testing Forward Pass")
    print("=" * 50)
    
    # Create dummy batch
    batch_size = 2  # Small batch for testing
    channels = cfg.model.encoder.in_channels
    crop_size = cfg.augmentation.spatial.random_crop.size
    
    dummy_batch = {
        'view_1': torch.randn(batch_size, channels, *crop_size),
        'view_2': torch.randn(batch_size, channels, *crop_size)
    }
    
    # Move to GPU if available
    device = torch.device(f"cuda:{cfg.hardware.gpu_id}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    dummy_batch['view_1'] = dummy_batch['view_1'].to(device)
    dummy_batch['view_2'] = dummy_batch['view_2'].to(device)
    
    print(f"Using device: {device}")
    
    # Forward pass
    try:
        with torch.cuda.amp.autocast(enabled=(cfg.hardware.precision == 16)):
            loss = model.training_step(dummy_batch, 0)
        
        print(f"✓ Forward pass successful!")
        print(f"  - Loss: {loss.item():.4f}")
        
        # Test feature extraction
        features = model.extract_features(dummy_batch['view_1'])
        print(f"  - Extracted features shape: {features.shape}")
        
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        return False
    
    return True


def test_memory_usage(cfg):
    """Test GPU memory usage with actual batch size"""
    print("\n4. Testing Memory Usage")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("GPU not available, skipping memory test")
        return True
    
    # Clear cache
    torch.cuda.empty_cache()
    gc.collect()
    
    # Get initial memory
    initial_memory = torch.cuda.memory_allocated() / 1024**3  # GB
    
    # Create model and data
    model = BYOL(cfg).cuda()
    
    # Simulate training batch
    batch_size = cfg.data.batch_size
    channels = cfg.model.encoder.in_channels
    crop_size = cfg.augmentation.spatial.random_crop.size
    
    batch = {
        'view_1': torch.randn(batch_size, channels, *crop_size).cuda(),
        'view_2': torch.randn(batch_size, channels, *crop_size).cuda()
    }
    
    # Forward and backward pass
    try:
        with torch.cuda.amp.autocast(enabled=(cfg.hardware.precision == 16)):
            loss = model.training_step(batch, 0)
            loss.backward()
        
        # Get peak memory
        peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
        current_memory = torch.cuda.memory_allocated() / 1024**3  # GB
        
        print(f"✓ Memory test successful!")
        print(f"  - Initial memory: {initial_memory:.2f} GB")
        print(f"  - Current memory: {current_memory:.2f} GB")
        print(f"  - Peak memory: {peak_memory:.2f} GB")
        print(f"  - GPU memory available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        
        if peak_memory > 14:  # 16GB GPU with some buffer
            print(f"\n⚠️  Warning: High memory usage! Consider reducing batch size.")
            print(f"   Current batch size: {batch_size}")
            suggested_batch = max(1, int(batch_size * 12 / peak_memory))
            print(f"   Suggested batch size: {suggested_batch}")
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"✗ Out of memory with batch size {batch_size}!")
            print(f"   Try reducing batch size in the config.")
        else:
            print(f"✗ Error during memory test: {e}")
        return False
    
    # Cleanup
    del model, batch, loss
    torch.cuda.empty_cache()
    gc.collect()
    
    return True


def main():
    """Run all tests"""
    print("BYOL Setup Test")
    print("=" * 70)
    
    # Load configuration
    cfg_path = Path("configs/byol.yaml")
    if not cfg_path.exists():
        print(f"❌ Config file not found: {cfg_path}")
        print("   Please create the BYOL configuration file first.")
        return
    
    cfg = OmegaConf.load(cfg_path)
    print(f"Loaded configuration from: {cfg_path}")
    
    # Run tests
    tests_passed = 0
    total_tests = 4
    
    # Test 1: Data loading
    try:
        if test_data_loading(cfg):
            tests_passed += 1
    except Exception as e:
        print(f"❌ Data loading test failed: {e}")
    
    # Test 2: Model creation
    try:
        model = test_model_creation(cfg)
        if model:
            tests_passed += 1
    except Exception as e:
        print(f"❌ Model creation test failed: {e}")
        return
    
    # Test 3: Forward pass
    try:
        if test_forward_pass(model, cfg):
            tests_passed += 1
    except Exception as e:
        print(f"❌ Forward pass test failed: {e}")
    
    # Test 4: Memory usage
    try:
        if test_memory_usage(cfg):
            tests_passed += 1
    except Exception as e:
        print(f"❌ Memory usage test failed: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("\n✅ All tests passed! Ready to start BYOL training.")
        print("\nTo start training, run:")
        print("  python scripts/train_byol.py")
    else:
        print("\n❌ Some tests failed. Please fix the issues before training.")


if __name__ == "__main__":
    main()