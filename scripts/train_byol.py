#!/usr/bin/env python3
"""
Train BYOL on ISLES 2022 dataset

Usage:
    python train_byol.py
    python train_byol.py training.epochs=100
    python train_byol.py data.batch_size=8
"""

import os
import sys
from pathlib import Path
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, RichProgressBar
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
import hydra
from omegaconf import DictConfig, OmegaConf
import warnings

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.byol import BYOL
from src.data.byol_datamodule import ISLESBYOLDataModule

# Suppress some warnings
warnings.filterwarnings("ignore", category=UserWarning)


def setup_callbacks(cfg: DictConfig):
    """Set up training callbacks"""
    callbacks = []
    
    # Model checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=Path("experiments") / cfg.experiment.name / "checkpoints",
        filename="{epoch:03d}-{train_loss:.4f}",
        save_top_k=cfg.training.checkpoint.save_top_k,
        monitor=cfg.training.checkpoint.monitor,
        mode=cfg.training.checkpoint.mode,
        save_last=cfg.training.checkpoint.save_last,
        verbose=True
    )
    callbacks.append(checkpoint_callback)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)
    
    # Progress bar
    progress_bar = RichProgressBar(leave=True)
    callbacks.append(progress_bar)
    
    return callbacks


def setup_loggers(cfg: DictConfig):
    """Set up experiment loggers"""
    loggers = []
    
    # Weights & Biases
    if cfg.experiment.wandb.enabled:
        wandb_logger = WandbLogger(
            project=cfg.experiment.wandb.project,
            name=cfg.experiment.name,
            entity=cfg.experiment.wandb.entity,
            tags=cfg.experiment.wandb.tags,
            config=OmegaConf.to_container(cfg, resolve=True),
            save_dir=Path("experiments") / cfg.experiment.name
        )
        loggers.append(wandb_logger)
    
    # TensorBoard
    if cfg.experiment.tensorboard.enabled:
        tb_logger = TensorBoardLogger(
            save_dir=cfg.experiment.tensorboard.log_dir,
            name=cfg.experiment.name
        )
        loggers.append(tb_logger)
    
    return loggers


@hydra.main(config_path="../configs", config_name="byol", version_base=None)
def main(cfg: DictConfig):
    """Main training function"""
    
    # Print configuration
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))
    
    # Set random seed for reproducibility
    if cfg.experiment.deterministic:
        pl.seed_everything(cfg.experiment.seed, workers=True)
    
    # Create data module
    print("\nSetting up data module...")
    data_module = ISLESBYOLDataModule(cfg)
    
    # Create model
    print("Creating BYOL model...")
    model = BYOL(cfg)
    
    # Set up callbacks and loggers
    callbacks = setup_callbacks(cfg)
    loggers = setup_loggers(cfg)
    
    # Create trainer
    print("Creating trainer...")
    
    # Handle device configuration
    if cfg.hardware.device == "cuda":
        accelerator = "gpu"
        devices = [cfg.hardware.gpu_id]
    elif cfg.hardware.device == "mps":
        accelerator = "mps"
        devices = 1  # MPS only supports single device
    else:
        accelerator = "cpu"
        devices = 1
    
    # MPS doesn't support 16-bit precision yet
    if cfg.hardware.device == "mps" and cfg.hardware.precision == 16:
        print("Warning: MPS doesn't support 16-bit precision, using 32-bit")
        precision = 32
    else:
        precision = cfg.hardware.precision
    
    trainer = pl.Trainer(
        max_epochs=cfg.training.epochs,
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        callbacks=callbacks,
        logger=loggers,
        gradient_clip_val=cfg.training.gradient_clip_val,
        deterministic=cfg.experiment.deterministic,
        enable_checkpointing=True,
        log_every_n_steps=10,
        val_check_interval=1.0,  # Validate every epoch
        accumulate_grad_batches=1,  # Can increase if batch size is too small
        num_sanity_val_steps=2,
        detect_anomaly=False,  # Set to True for debugging
    )
    
    # Start training
    print("\nStarting BYOL training...")
    print(f"Training for {cfg.training.epochs} epochs")
    print(f"Batch size: {cfg.data.batch_size}")
    print(f"Learning rate: {cfg.training.optimizer.lr}")
    
    trainer.fit(model, data_module)
    
    # Save final model
    final_model_path = Path("experiments") / cfg.experiment.name / "final_model.ckpt"
    trainer.save_checkpoint(final_model_path)
    print(f"\nTraining complete! Final model saved to: {final_model_path}")
    
    # Extract and save encoder weights for downstream tasks
    encoder_path = Path("experiments") / cfg.experiment.name / "encoder.pth"
    torch.save(model.online_encoder.state_dict(), encoder_path)
    print(f"Encoder weights saved to: {encoder_path}")


if __name__ == "__main__":
    main()