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
import argparse

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
    
    # Move to appropriate device
    if cfg.hardware.device == "cuda":
        device = torch.device(f"cuda:{cfg.hardware.gpu_id}" if torch.cuda.is_available() else "cpu")
    elif cfg.hardware.device == "mps":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    
    model = model.to(device)
    dummy_batch['view_1'] = dummy_batch['view_1'].to(device)
    dummy_batch['view_2'] = dummy_batch['view_2'].to(device)
    
    print(f"Using device: {device}")
    
    # Forward pass
    try:
        # MPS doesn't support mixed precision yet
        use_amp = cfg.hardware.precision == 16 and cfg.hardware.device == "cuda"
        device_type = cfg.hardware.device if cfg.hardware.device in ["cuda", "mps"] else "cpu"
        
        with torch.amp.autocast(device_type=device_type, enabled=use_amp):
            # Test online network forward pass
            pred_1 = model.forward_online(dummy_batch['view_1'])
            pred_2 = model.forward_online(dummy_batch['view_2'])
            
            # Test target network forward pass
            proj_1 = model.forward_target(dummy_batch['view_1'])
            proj_2 = model.forward_target(dummy_batch['view_2'])
            
            # Compute loss manually
            loss = model.byol_loss(pred_1, proj_2, pred_2, proj_1)
        
        print(f"✓ Forward pass successful!")
        print(f"  - Loss: {loss.item():.4f}")
        print(f"  - Prediction shape: {pred_1.shape}")
        
        # Test feature extraction
        features = model.extract_features(dummy_batch['view_1'])
        print(f"  - Extracted features shape: {features.shape}")
        
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        return False
    
    return True


def test_memory_usage(cfg):
    """Test GPU/MPS memory usage with actual batch size"""
    print("\n4. Testing Memory Usage")
    print("=" * 50)
    
    device_type = cfg.hardware.device
    
    if device_type == "cpu":
        print("Using CPU, skipping memory test")
        return True
    
    if device_type == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available, skipping memory test")
        return True
        
    if device_type == "mps" and not torch.backends.mps.is_available():
        print("MPS requested but not available, skipping memory test")
        return True
    
    # Clear cache
    if device_type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    
    # Get device
    if device_type == "cuda":
        device = torch.device(f"cuda:{cfg.hardware.gpu_id}")
        initial_memory = torch.cuda.memory_allocated() / 1024**3
    elif device_type == "mps":
        device = torch.device("mps")
        initial_memory = torch.mps.current_allocated_memory() / 1024**3 if hasattr(torch.mps, 'current_allocated_memory') else 0
    
    # Create model and data
    model = BYOL(cfg).to(device)
    
    # Simulate training batch
    batch_size = cfg.data.batch_size
    channels = cfg.model.encoder.in_channels
    crop_size = cfg.augmentation.spatial.random_crop.size
    
    batch = {
        'view_1': torch.randn(batch_size, channels, *crop_size).to(device),
        'view_2': torch.randn(batch_size, channels, *crop_size).to(device)
    }
    
    # Forward and backward pass
    try:
        # Note: MPS doesn't support mixed precision yet
        use_amp = cfg.hardware.precision == 16 and device_type == "cuda"
        
        with torch.amp.autocast(device_type=device_type, enabled=use_amp):
            # Forward pass through both networks
            pred_1 = model.forward_online(batch['view_1'])
            pred_2 = model.forward_online(batch['view_2'])
            proj_1 = model.forward_target(batch['view_1'])
            proj_2 = model.forward_target(batch['view_2'])
            
            # Compute loss
            loss = model.byol_loss(pred_1, proj_2, pred_2, proj_1)
            loss.backward()
        
        # Get memory usage
        if device_type == "cuda":
            peak_memory = torch.cuda.max_memory_allocated() / 1024**3
            current_memory = torch.cuda.memory_allocated() / 1024**3
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        elif device_type == "mps":
            # MPS memory reporting is limited
            current_memory = torch.mps.current_allocated_memory() / 1024**3 if hasattr(torch.mps, 'current_allocated_memory') else 0
            peak_memory = current_memory  # MPS doesn't track peak
            total_memory = 0  # Can't query total MPS memory easily
        
        print(f"✓ Memory test successful!")
        print(f"  - Device: {device}")
        print(f"  - Initial memory: {initial_memory:.2f} GB")
        print(f"  - Current memory: {current_memory:.2f} GB")
        
        if device_type == "cuda":
            print(f"  - Peak memory: {peak_memory:.2f} GB")
            print(f"  - Total GPU memory: {total_memory:.2f} GB")
            
            if peak_memory > 14:  # 16GB GPU with buffer
                print(f"\n⚠️  Warning: High memory usage! Consider reducing batch size.")
                print(f"   Current batch size: {batch_size}")
                suggested_batch = max(1, int(batch_size * 12 / peak_memory))
                print(f"   Suggested batch size: {suggested_batch}")
        elif device_type == "mps":
            print(f"  - Note: MPS has shared memory with system RAM")
            print(f"  - Monitor Activity Monitor for total memory usage")
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"✗ Out of memory with batch size {batch_size}!")
            print(f"   Try reducing batch size in the config.")
        else:
            print(f"✗ Error during memory test: {e}")
        return False
    
    # Cleanup
    del model, batch, loss
    if device_type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    
    return True


def main():
    """Run all tests"""
    print("BYOL Setup Test")
    print("=" * 70)
    
    # Load configuration
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-name', default='byol', help='Config name')
    args = parser.parse_args()
    
    cfg_path = Path(f"configs/{args.config_name}.yaml")
    if not cfg_path.exists():
        print(f"❌ Config file not found: {cfg_path}")
        print("   Please create the configuration file first.")
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