"""
BYOL (Bootstrap Your Own Latent) implementation for 3D medical images

Based on: https://arxiv.org/abs/2006.07733
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, Tuple, Optional
import numpy as np

from src.models.resnet3d import resnet18_3d, resnet34_3d, resnet50_3d


class MLP(nn.Module):
    """Multi-layer perceptron for projector and predictor"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        use_bn: bool = True
    ):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BYOL(pl.LightningModule):
    """BYOL self-supervised learning module"""
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()
        
        # Create encoder backbone
        encoder_name = cfg.model.encoder.name
        if encoder_name == "resnet18_3d":
            encoder = resnet18_3d(in_channels=cfg.model.encoder.in_channels)
            features_dim = 512
        elif encoder_name == "resnet34_3d":
            encoder = resnet34_3d(in_channels=cfg.model.encoder.in_channels)
            features_dim = 512
        elif encoder_name == "resnet50_3d":
            encoder = resnet50_3d(in_channels=cfg.model.encoder.in_channels)
            features_dim = 2048
        else:
            raise ValueError(f"Unknown encoder: {encoder_name}")
        
        # Remove the final FC layer
        self.features_dim = features_dim
        encoder.fc = nn.Identity()
        
        # Online network
        self.online_encoder = encoder
        self.online_projector = MLP(
            input_dim=features_dim,
            hidden_dim=cfg.model.projector.hidden_dim,
            output_dim=cfg.model.projector.output_dim,
            use_bn=cfg.model.projector.use_bn
        )
        self.online_predictor = MLP(
            input_dim=cfg.model.projector.output_dim,
            hidden_dim=cfg.model.predictor.hidden_dim,
            output_dim=cfg.model.predictor.output_dim,
            use_bn=cfg.model.predictor.use_bn
        )
        
        # Target network (no predictor)
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.target_projector = copy.deepcopy(self.online_projector)
        
        # Freeze target network
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        for param in self.target_projector.parameters():
            param.requires_grad = False
            
        # Initialize momentum
        self.current_momentum = cfg.model.momentum.base
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through encoder only (for inference)"""
        return self.online_encoder(x)
    
    def forward_online(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through online network"""
        features = self.online_encoder.forward_features(x)
        projection = self.online_projector(features)
        prediction = self.online_predictor(projection)
        return prediction
    
    def forward_target(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through target network"""
        with torch.no_grad():
            features = self.target_encoder.forward_features(x)
            projection = self.target_projector(features)
        return projection.detach()
    
    def byol_loss(
        self,
        pred_1: torch.Tensor,
        proj_2: torch.Tensor,
        pred_2: torch.Tensor,
        proj_1: torch.Tensor
    ) -> torch.Tensor:
        """BYOL loss: symmetrized L2 loss between predictions and projections"""
        pred_1 = F.normalize(pred_1, dim=-1, p=2)
        pred_2 = F.normalize(pred_2, dim=-1, p=2)
        proj_1 = F.normalize(proj_1, dim=-1, p=2)
        proj_2 = F.normalize(proj_2, dim=-1, p=2)
        
        loss_1 = 2 - 2 * (pred_1 * proj_2.detach()).sum(dim=-1).mean()
        loss_2 = 2 - 2 * (pred_2 * proj_1.detach()).sum(dim=-1).mean()
        
        return (loss_1 + loss_2) / 2
    
    def update_momentum(self):
        """Update momentum coefficient during training"""
        # Cosine schedule for momentum
        max_steps = self.cfg.training.epochs * self.trainer.num_training_batches
        current_step = self.global_step
        
        base_momentum = self.cfg.model.momentum.base
        final_momentum = self.cfg.model.momentum.final
        
        self.current_momentum = final_momentum - (final_momentum - base_momentum) * \
                               (np.cos(np.pi * current_step / max_steps) + 1) / 2
        
    def update_target_network(self):
        """Update target network with exponential moving average"""
        for param_online, param_target in zip(
            self.online_encoder.parameters(),
            self.target_encoder.parameters()
        ):
            param_target.data = param_target.data * self.current_momentum + \
                               param_online.data * (1 - self.current_momentum)
            
        for param_online, param_target in zip(
            self.online_projector.parameters(),
            self.target_projector.parameters()
        ):
            param_target.data = param_target.data * self.current_momentum + \
                               param_online.data * (1 - self.current_momentum)
    
    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        """Training step for BYOL"""
        # Batch contains two augmented views
        view_1, view_2 = batch['view_1'], batch['view_2']
        
        # Forward pass through both networks
        pred_1 = self.forward_online(view_1)
        pred_2 = self.forward_online(view_2)
        
        proj_1 = self.forward_target(view_1)
        proj_2 = self.forward_target(view_2)
        
        # Compute loss
        loss = self.byol_loss(pred_1, proj_2, pred_2, proj_1)
        
        # Update momentum
        self.update_momentum()
        
        # Update target network
        self.update_target_network()
        
        # Logging
        self.log('train_loss', loss, prog_bar=True)
        self.log('momentum', self.current_momentum, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        """Validation step - compute loss without updates"""
        view_1, view_2 = batch['view_1'], batch['view_2']
        
        pred_1 = self.forward_online(view_1)
        pred_2 = self.forward_online(view_2)
        
        proj_1 = self.forward_target(view_1)
        proj_2 = self.forward_target(view_2)
        
        loss = self.byol_loss(pred_1, proj_2, pred_2, proj_1)
        
        self.log('val_loss', loss, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler"""
        # Optimizer
        if self.cfg.training.optimizer.name == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.cfg.training.optimizer.lr,
                weight_decay=self.cfg.training.optimizer.weight_decay,
                betas=self.cfg.training.optimizer.betas
            )
        elif self.cfg.training.optimizer.name == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.cfg.training.optimizer.lr,
                weight_decay=self.cfg.training.optimizer.weight_decay,
                betas=self.cfg.training.optimizer.betas
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.cfg.training.optimizer.name}")
        
        # Learning rate scheduler
        if self.cfg.training.scheduler.name == "cosine":
            # Warmup scheduler
            def warmup_lambda(epoch):
                if epoch < self.cfg.training.scheduler.warmup_epochs:
                    return (self.cfg.training.scheduler.warmup_start_lr + 
                           (self.cfg.training.optimizer.lr - self.cfg.training.scheduler.warmup_start_lr) * 
                           epoch / self.cfg.training.scheduler.warmup_epochs) / self.cfg.training.optimizer.lr
                return 1.0
            
            warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=warmup_lambda
            )
            
            # Cosine annealing after warmup
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.cfg.training.epochs - self.cfg.training.scheduler.warmup_epochs,
                eta_min=self.cfg.training.scheduler.eta_min
            )
            
            scheduler = {
                'scheduler': torch.optim.lr_scheduler.SequentialLR(
                    optimizer,
                    schedulers=[warmup_scheduler, cosine_scheduler],
                    milestones=[self.cfg.training.scheduler.warmup_epochs]
                ),
                'interval': 'epoch'
            }
            
            return [optimizer], [scheduler]
        
        return optimizer
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features for downstream tasks"""
        return self.online_encoder.forward_features(x)


# Test the module
if __name__ == "__main__":
    from omegaconf import OmegaConf
    
    # Create dummy config
    cfg = OmegaConf.create({
        'model': {
            'encoder': {
                'name': 'resnet50_3d',
                'in_channels': 3
            },
            'projector': {
                'hidden_dim': 4096,
                'output_dim': 256,
                'use_bn': True
            },
            'predictor': {
                'hidden_dim': 4096,
                'output_dim': 256,
                'use_bn': True
            },
            'momentum': {
                'base': 0.996,
                'final': 1.0,
                'epochs': 200
            }
        },
        'training': {
            'epochs': 200,
            'optimizer': {
                'name': 'adamw',
                'lr': 1e-4,
                'weight_decay': 1e-6,
                'betas': [0.9, 0.999]
            },
            'scheduler': {
                'name': 'cosine',
                'warmup_epochs': 10,
                'warmup_start_lr': 1e-6,
                'eta_min': 1e-7
            }
        }
    })
    
    # Create model
    model = BYOL(cfg)
    
    # Test with dummy data
    batch_size = 2
    channels = 3
    depth, height, width = 128, 128, 128
    
    batch = {
        'view_1': torch.randn(batch_size, channels, depth, height, width),
        'view_2': torch.randn(batch_size, channels, depth, height, width)
    }
    
    # Forward pass
    loss = model.training_step(batch, 0)
    
    print(f"Loss: {loss.item():.4f}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")