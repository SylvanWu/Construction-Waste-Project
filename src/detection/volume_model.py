"""
体积回归模型封装
基于DepthFunction的B1 Baseline模型
"""

import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from loguru import logger
from pathlib import Path
from typing import Optional
import cv2
from PIL import Image
import torchvision.transforms as transforms


class VolumeRegressorB1(nn.Module):
    """
    B1基线模型：直接体积回归
    ResNet18 → AdaptiveAvgPool → FC → volume
    """
    
    def __init__(
        self,
        backbone: str = 'resnet18',
        pretrained: bool = True,
        dropout_p: float = 0.3,
        freeze_backbone: bool = False
    ):
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
        else:
            raise ValueError(f"不支持的backbone: {backbone}")
        
        # 移除最后的分类层
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # 冻结骨干网络（可选）
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 回归头
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p * 0.67),
            nn.Linear(128, 1)
        )
        
        self._init_regressor()
    
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
        features = self.backbone(x)
        features = self.global_pool(features)
        volume = self.regressor(features)
        return volume


class VolumeModel:
    """体积估测模型封装"""
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = "auto",
        input_size: int = 224,
        baseline_volume: float = 0.0
    ):
        """
        初始化体积模型
        
        Args:
            checkpoint_path: 模型checkpoint路径
            device: 设备（"auto", "cuda", "cpu"）
            input_size: 输入图像大小（224或448）
            baseline_volume: 空桶基准体积
        """
        self.checkpoint_path = checkpoint_path
        self.input_size = input_size
        self.baseline_volume = baseline_volume
        self.model = None
        
        # 设置设备
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # 图像预处理transform
        self.transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet标准
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # 验证模型文件
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"模型checkpoint不存在: {checkpoint_path}")
        
        self.load_model()
    
    def load_model(self):
        """加载模型checkpoint"""
        try:
            logger.info(f"正在加载体积模型: {self.checkpoint_path}")
            
            # 创建模型
            self.model = VolumeRegressorB1(
                backbone='resnet18',
                pretrained=False,  # 加载checkpoint，不需要预训练权重
                dropout_p=0.3
            )
            
            # 加载checkpoint
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # 移动到设备并设置为评估模式
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"体积模型加载成功，使用设备: {self.device}")
            logger.info(f"输入尺寸: {self.input_size}×{self.input_size}")
            
            # 记录checkpoint信息
            if 'val_mae' in checkpoint:
                logger.info(f"模型验证MAE: {checkpoint['val_mae']:.2f}L")
            if 'val_r2' in checkpoint:
                logger.info(f"模型验证R²: {checkpoint['val_r2']:.4f}")
                
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
    
    def predict(self, image: np.ndarray, bin_mask: Optional[np.ndarray] = None) -> float:
        """
        预测体积
        
        Args:
            image: RGB图像 (H, W, 3) 或 BGR图像
            bin_mask: 可选的bin区域mask，用于裁剪ROI
            
        Returns:
            预测的体积值（升）
        """
        try:
            # 如果是BGR，转换为RGB
            if image.shape[2] == 3:
                # 检测是否是BGR（OpenCV格式）
                # 简单假设：如果均值通道顺序看起来像BGR就转换
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
            
            # 如果提供了mask，裁剪ROI
            if bin_mask is not None:
                image_rgb = self._extract_roi(image_rgb, bin_mask)
            
            # 预处理
            image_pil = Image.fromarray(image_rgb)
            image_tensor = self.transform(image_pil).unsqueeze(0)  # [1, 3, H, W]
            image_tensor = image_tensor.to(self.device)
            
            # 推理
            with torch.no_grad():
                volume_tensor = self.model(image_tensor)
                volume = volume_tensor.item()
            
            return volume
            
        except Exception as e:
            logger.error(f"体积预测失败: {e}")
            return 0.0
    
    def _extract_roi(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        根据mask裁剪ROI区域
        
        Args:
            image: RGB图像
            mask: 二值mask
            
        Returns:
            裁剪后的ROI图像
        """
        # 确保mask尺寸匹配
        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(mask.astype(np.uint8), 
                            (image.shape[1], image.shape[0]))
        
        # 找到mask的边界框
        coords = np.argwhere(mask > 0)
        if len(coords) == 0:
            return image  # 如果mask为空，返回原图
        
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        # 裁剪ROI
        roi = image[y_min:y_max+1, x_min:x_max+1]
        
        return roi
    
    def set_baseline(self, baseline_volume: float):
        """设置空桶基准体积"""
        self.baseline_volume = baseline_volume
        logger.info(f"设置空桶基准体积: {baseline_volume:.2f}L")
    
    def calculate_fill_percentage(self, current_volume: float, max_volume: float = 100.0) -> float:
        """
        计算填充率
        
        Args:
            current_volume: 当前体积
            max_volume: 最大容量（默认100L）
            
        Returns:
            填充率（百分比）
        """
        if max_volume <= 0:
            return 0.0
        return (current_volume / max_volume) * 100.0
    
    def get_relative_volume(self, current_volume: float) -> float:
        """
        获取相对于基准的体积增量
        
        Args:
            current_volume: 当前预测体积
            
        Returns:
            相对体积增量
        """
        return max(0.0, current_volume - self.baseline_volume)
