"""
3D ResNet backbone for BYOL encoder

Adapted from torchvision ResNet for 3D medical images
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Type, Any, Callable, Union, List, Optional


class Conv3dAuto(nn.Conv3d):
    """3D Convolution with automatic padding to maintain spatial dimensions"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding = (self.kernel_size[0] // 2, 
                       self.kernel_size[1] // 2,
                       self.kernel_size[2] // 2)


class BasicBlock3d(nn.Module):
    """Basic ResNet block for 3D"""
    expansion = 1
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
            
        self.conv1 = Conv3dAuto(in_channels, out_channels, kernel_size=3, stride=stride)
        self.bn1 = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = Conv3dAuto(out_channels, out_channels, kernel_size=3)
        self.bn2 = norm_layer(out_channels)
        self.downsample = downsample
        self.stride = stride
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        
        return out


class Bottleneck3d(nn.Module):
    """Bottleneck ResNet block for 3D"""
    expansion = 4
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
            
        width = int(out_channels * (base_width / 64.0)) * groups
        
        self.conv1 = nn.Conv3d(in_channels, width, kernel_size=1, bias=False)
        self.bn1 = norm_layer(width)
        self.conv2 = Conv3dAuto(width, width, kernel_size=3, stride=stride, groups=groups)
        self.bn2 = norm_layer(width)
        self.conv3 = nn.Conv3d(width, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = norm_layer(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        
        return out


class ResNet3d(nn.Module):
    """3D ResNet for medical imaging"""
    
    def __init__(
        self,
        block: Type[Union[BasicBlock3d, Bottleneck3d]],
        layers: List[int],
        in_channels: int = 3,
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super().__init__()
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        self._norm_layer = norm_layer
        
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
            
        self.groups = groups
        self.base_width = width_per_group
        
        # Initial convolution - smaller kernel for 3D
        self.conv1 = nn.Conv3d(in_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        # ResNet layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Final FC layer (will be removed for BYOL)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def _make_layer(
        self,
        block: Type[Union[BasicBlock3d, Bottleneck3d]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        
        if dilate:
            self.dilation *= stride
            stride = 1
            
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion),
            )
            
        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups,
                self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer
                )
            )
            
        return nn.Sequential(*layers)
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before the final FC layer"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.fc(x)
        return x


def resnet18_3d(**kwargs) -> ResNet3d:
    """ResNet-18 3D"""
    return ResNet3d(BasicBlock3d, [2, 2, 2, 2], **kwargs)


def resnet34_3d(**kwargs) -> ResNet3d:
    """ResNet-34 3D"""
    return ResNet3d(BasicBlock3d, [3, 4, 6, 3], **kwargs)


def resnet50_3d(**kwargs) -> ResNet3d:
    """ResNet-50 3D"""
    return ResNet3d(Bottleneck3d, [3, 4, 6, 3], **kwargs)


def resnet101_3d(**kwargs) -> ResNet3d:
    """ResNet-101 3D"""
    return ResNet3d(Bottleneck3d, [3, 4, 23, 3], **kwargs)


def resnet152_3d(**kwargs) -> ResNet3d:
    """ResNet-152 3D"""
    return ResNet3d(Bottleneck3d, [3, 8, 36, 3], **kwargs)


# Test the model
if __name__ == "__main__":
    # Create model
    model = resnet50_3d(in_channels=3, num_classes=1000)
    
    # Test with random input
    batch_size = 2
    channels = 3  # FLAIR + DWI + ADC
    depth, height, width = 128, 128, 128
    
    x = torch.randn(batch_size, channels, depth, height, width)
    
    # Forward pass
    features = model.forward_features(x)
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Features shape: {features.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")