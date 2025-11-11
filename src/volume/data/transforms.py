#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据增强模块
针对RGB体积估计任务的专项增强，特别是对抗蓝通道噪声
"""

import random
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from io import BytesIO
from typing import Tuple


class ChannelDropout:
    """
    通道Dropout - 对抗蓝通道噪声
    随机丢弃或降低某个颜色通道的权重
    """
    
    def __init__(self, p: float = 0.25, drop_channel: str = 'blue'):
        """
        Args:
            p: 执行dropout的概率
            drop_channel: 要dropout的通道 ('red', 'green', 'blue')
        """
        self.p = p
        self.channel_map = {'red': 0, 'green': 1, 'blue': 2}
        self.drop_channel = drop_channel
        
        if drop_channel not in self.channel_map:
            raise ValueError(f"drop_channel必须是 'red', 'green' 或 'blue'")
    
    def __call__(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img_tensor: [C, H, W] Tensor (已归一化到[0, 1])
        
        Returns:
            augmented_tensor: [C, H, W] Tensor
        """
        if random.random() < self.p:
            channel_idx = self.channel_map[self.drop_channel]
            
            # 策略1: 完全置零 (50%概率)
            if random.random() < 0.5:
                img_tensor[channel_idx] = 0
            # 策略2: 降权重到30-70% (50%概率)
            else:
                scale = random.uniform(0.3, 0.7)
                img_tensor[channel_idx] *= scale
        
        return img_tensor
    
    def __repr__(self):
        return f"ChannelDropout(p={self.p}, channel={self.drop_channel})"


class WhiteBalanceAugmentation:
    """
    白平衡增强 - 模拟不同相机的白平衡设置
    RGB通道独立缩放，增加颜色多样性
    """
    
    def __init__(
        self, 
        r_range: Tuple[float, float] = (0.85, 1.15),
        g_range: Tuple[float, float] = (0.90, 1.10),
        b_range: Tuple[float, float] = (0.80, 1.20)
    ):
        """
        Args:
            r_range: 红色通道缩放范围
            g_range: 绿色通道缩放范围 (通常更稳定)
            b_range: 蓝色通道缩放范围 (波动更大)
        """
        self.r_range = r_range
        self.g_range = g_range
        self.b_range = b_range
    
    def __call__(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img_tensor: [C, H, W] Tensor (已归一化到[0, 1])
        
        Returns:
            augmented_tensor: [C, H, W] Tensor
        """
        scale_r = random.uniform(*self.r_range)
        scale_g = random.uniform(*self.g_range)
        scale_b = random.uniform(*self.b_range)
        
        img_tensor[0] *= scale_r
        img_tensor[1] *= scale_g
        img_tensor[2] *= scale_b
        
        # Clamp到[0, 1]
        img_tensor = torch.clamp(img_tensor, 0, 1)
        
        return img_tensor
    
    def __repr__(self):
        return f"WhiteBalanceAugmentation(r={self.r_range}, g={self.g_range}, b={self.b_range})"


class JPEGCompression:
    """
    JPEG压缩模拟 - 模拟不同质量的JPEG编码
    增强对压缩伪影的鲁棒性
    """
    
    def __init__(self, quality_range: Tuple[int, int] = (70, 100)):
        """
        Args:
            quality_range: JPEG质量范围 (1-100)
        """
        self.quality_range = quality_range
    
    def __call__(self, img_pil: Image.Image) -> Image.Image:
        """
        Args:
            img_pil: PIL Image
        
        Returns:
            compressed_img: PIL Image
        """
        quality = random.randint(*self.quality_range)
        
        # 压缩到buffer
        buffer = BytesIO()
        img_pil.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        
        # 重新加载
        compressed_img = Image.open(buffer).copy()
        buffer.close()
        
        return compressed_img
    
    def __repr__(self):
        return f"JPEGCompression(quality={self.quality_range})"


class RandomShadow:
    """
    随机阴影 - 模拟局部光照变化
    """
    
    def __init__(self, p: float = 0.3, shadow_strength: Tuple[float, float] = (0.3, 0.7)):
        """
        Args:
            p: 应用阴影的概率
            shadow_strength: 阴影强度范围（降低亮度的比例）
        """
        self.p = p
        self.shadow_strength = shadow_strength
    
    def __call__(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img_tensor: [C, H, W] Tensor
        
        Returns:
            shadowed_tensor: [C, H, W] Tensor
        """
        if random.random() < self.p:
            _, h, w = img_tensor.shape
            
            # 随机生成阴影区域（圆形或矩形）
            if random.random() < 0.5:
                # 圆形阴影
                cx = random.randint(0, w)
                cy = random.randint(0, h)
                radius = random.randint(h // 4, h // 2)
                
                y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
                mask = ((x - cx) ** 2 + (y - cy) ** 2) <= radius ** 2
            else:
                # 矩形阴影
                x1 = random.randint(0, w // 2)
                y1 = random.randint(0, h // 2)
                x2 = random.randint(w // 2, w)
                y2 = random.randint(h // 2, h)
                
                mask = torch.zeros(h, w, dtype=torch.bool)
                mask[y1:y2, x1:x2] = True
            
            # 应用阴影
            strength = random.uniform(*self.shadow_strength)
            img_tensor[:, mask] *= (1 - strength)
        
        return img_tensor
    
    def __repr__(self):
        return f"RandomShadow(p={self.p}, strength={self.shadow_strength})"


def get_train_transform(
    input_size: int = 224,
    color_jitter_params: dict = None,
    channel_dropout_p: float = 0.25,
    jpeg_quality: Tuple[int, int] = (70, 100),
    gaussian_blur_p: float = 0.2,
    use_shadow: bool = True
) -> transforms.Compose:
    """
    获取训练集数据增强pipeline
    
    Args:
        input_size: 输入图像尺寸
        color_jitter_params: 色彩抖动参数
        channel_dropout_p: 通道dropout概率
        jpeg_quality: JPEG压缩质量范围
        gaussian_blur_p: 高斯模糊概率
        use_shadow: 是否使用阴影增强
    
    Returns:
        transform: torchvision.transforms.Compose
    """
    if color_jitter_params is None:
        color_jitter_params = {
            'brightness': 0.3,
            'contrast': 0.3,
            'saturation': 0.3,
            'hue': 0.05
        }
    
    transform_list = [
        # PIL操作
        transforms.ToPILImage(),
        JPEGCompression(quality_range=jpeg_quality),
        transforms.Resize((input_size + 32, input_size + 32)),  # 稍大一点用于crop
        transforms.RandomCrop((input_size, input_size)),
        transforms.ColorJitter(**color_jitter_params),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=5)], p=gaussian_blur_p),
        
        # 转Tensor
        transforms.ToTensor(),  # [0, 1]
        
        # Tensor操作
        WhiteBalanceAugmentation(),
        ChannelDropout(p=channel_dropout_p, drop_channel='blue'),
    ]
    
    # 可选阴影
    if use_shadow:
        transform_list.append(RandomShadow(p=0.3))
    
    # 归一化（ImageNet统计量）
    transform_list.append(
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    )
    
    return transforms.Compose(transform_list)


def get_val_transform(input_size: int = 224) -> transforms.Compose:
    """
    获取验证/测试集数据增强pipeline（仅resize和归一化）
    
    Args:
        input_size: 输入图像尺寸
    
    Returns:
        transform: torchvision.transforms.Compose
    """
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_minimal_transform(input_size: int = 224) -> transforms.Compose:
    """
    获取最小增强（消融实验用）
    
    Args:
        input_size: 输入图像尺寸
    
    Returns:
        transform: torchvision.transforms.Compose
    """
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(p=0.5),  # 仅水平翻转
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_grayscale_transform(input_size: int = 224) -> transforms.Compose:
    """
    获取灰度图增强（消融实验：验证颜色信息的作用）
    
    Args:
        input_size: 输入图像尺寸
    
    Returns:
        transform: torchvision.transforms.Compose
    """
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=3),  # 转灰度但保持3通道
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


if __name__ == "__main__":
    """测试数据增强效果"""
    import matplotlib.pyplot as plt
    import cv2
    
    # 加载测试图像
    test_img = cv2.imread("../dataset/color/frame_000100.png")
    test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
    
    # 创建增强pipeline
    train_transform = get_train_transform()
    val_transform = get_val_transform()
    
    # 测试增强效果
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    
    # 原图
    axes[0, 0].imshow(test_img)
    axes[0, 0].set_title("Original")
    axes[0, 0].axis('off')
    
    # 9个增强样本
    for i in range(9):
        augmented = train_transform(test_img)
        
        # 反归一化用于可视化
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_denorm = augmented * std + mean
        img_denorm = torch.clamp(img_denorm, 0, 1)
        img_denorm = img_denorm.permute(1, 2, 0).numpy()
        
        row = (i + 1) // 5
        col = (i + 1) % 5
        axes[row, col].imshow(img_denorm)
        axes[row, col].set_title(f"Augmented {i+1}")
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig("../results/augmentation_examples.png", dpi=150, bbox_inches='tight')
    print("增强示例已保存至: results/augmentation_examples.png")
