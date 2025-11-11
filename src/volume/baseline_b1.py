#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Baseline B1: 直接体积回归器
ResNet18 + FC层 → 标量体积输出
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class VolumeRegressorB1(nn.Module):
    """
    B1基线模型：直接体积回归
    
    架构:
        Input: RGB [3, H, W]
          ↓
        ResNet18 Backbone (ImageNet预训练)
          ↓
        AdaptiveAvgPool → [512]
          ↓
        FC: 512 → 256 → 128 → 1
          ↓
        Output: volume (scalar)
    """
    
    def __init__(
        self,
        backbone: str = 'resnet18',
        pretrained: bool = True,
        dropout_p: float = 0.3,
        freeze_backbone: bool = False
    ):
        """
        初始化B1模型
        
        Args:
            backbone: 骨干网络 ('resnet18', 'resnet34', 'mobilenetv3_small')
            pretrained: 是否使用ImageNet预训练权重
            dropout_p: Dropout比例
            freeze_backbone: 是否冻结骨干网络
        """
        super().__init__()
        
        self.backbone_name = backbone
        self.pretrained = pretrained
        
        # 加载骨干网络
        if backbone == 'resnet18':
            self.backbone = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
            feature_dim = 512
        elif backbone == 'resnet34':
            self.backbone = models.resnet34(weights='IMAGENET1K_V1' if pretrained else None)
            feature_dim = 512
        elif backbone == 'mobilenetv3_small':
            self.backbone = models.mobilenet_v3_small(weights='IMAGENET1K_V1' if pretrained else None)
            feature_dim = 576  # MobileNetV3-Small的最后一层特征维度
        else:
            raise ValueError(f"不支持的backbone: {backbone}")
        
        # 移除最后的分类层
        if 'resnet' in backbone:
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])  # 去掉FC层
        elif 'mobilenet' in backbone:
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])  # 去掉classifier
        
        # 冻结骨干网络（可选）
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            logger.info("骨干网络已冻结")
        
        # 全局平均池化（确保输出[B, feature_dim]）
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 回归头
        self.regressor = nn.Sequential(
            nn.Flatten(),
            
            # FC1: feature_dim → 256
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            
            # FC2: 256 → 128
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p * 0.67),  # 稍微降低dropout
            
            # FC3: 128 → 1
            nn.Linear(128, 1)
        )
        
        # 初始化回归头（骨干网络已有预训练权重）
        self._init_regressor()
        
        logger.info(f"B1模型初始化完成: {backbone}, "
                   f"pretrained={pretrained}, dropout={dropout_p}")
    
    def _init_regressor(self):
        """初始化回归头权重"""
        for m in self.regressor.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入图像 [B, 3, H, W]
        
        Returns:
            volume: 预测体积 [B, 1]
        """
        # 骨干网络特征提取
        features = self.backbone(x)  # [B, feature_dim, h, w]
        
        # 全局池化
        features = self.global_pool(features)  # [B, feature_dim, 1, 1]
        
        # 回归预测
        volume = self.regressor(features)  # [B, 1]
        
        return volume
    
    def get_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        """
        提取特征图（用于可视化）
        
        Args:
            x: 输入图像 [B, 3, H, W]
        
        Returns:
            features: 特征图 [B, feature_dim, h, w]
        """
        return self.backbone(x)


def count_parameters(model: nn.Module) -> int:
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size(model: nn.Module) -> float:
    """计算模型大小（MB）"""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    size_mb = (param_size + buffer_size) / (1024 ** 2)
    return size_mb


if __name__ == "__main__":
    """测试B1模型"""
    logging.basicConfig(level=logging.INFO)
    
    # 创建模型
    model = VolumeRegressorB1(backbone='resnet18', pretrained=True, dropout_p=0.3)
    
    # 模型信息
    num_params = count_parameters(model)
    model_size = get_model_size(model)
    
    print(f"\n模型信息:")
    print(f"  可训练参数: {num_params:,}")
    print(f"  模型大小: {model_size:.2f} MB")
    
    # 测试前向传播
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # 创建随机输入
    batch_size = 4
    test_input = torch.randn(batch_size, 3, 224, 224).to(device)
    
    with torch.no_grad():
        output = model(test_input)
    
    print(f"\n前向传播测试:")
    print(f"  输入形状: {test_input.shape}")
    print(f"  输出形状: {output.shape}")
    print(f"  输出值: {output.squeeze().tolist()}")
    
    # 测试特征提取
    with torch.no_grad():
        features = model.get_feature_maps(test_input)
    print(f"  特征图形状: {features.shape}")
