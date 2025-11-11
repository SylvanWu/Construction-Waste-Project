#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练配置管理
集中管理所有超参数
"""

import torch
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class Config:
    """训练配置"""
    
    # ========== 数据配置 ==========
    dataset_h5: str = "training_data/volume_dataset_methodA.h5"
    dataset_csv: str = "training_data/volume_dataset_methodA.csv"
    total_frames: int = 603
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    split_mode: str = 'temporal'  # 'temporal' or 'random'
    
    # ========== 模型配置 ==========
    model_name: str = 'resnet18'  # 'resnet18', 'resnet34', 'mobilenetv3_small'
    input_size: int = 224
    pretrained: bool = True
    dropout_p: float = 0.3
    freeze_backbone: bool = False
    
    # ========== 训练配置 ==========
    batch_size: int = 16
    epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_accumulation: int = 2  # 有效batch=16*2=32
    
    # ========== 优化器配置 ==========
    optimizer: str = "adamw"
    scheduler: str = "cosine"  # 'cosine', 'step', 'plateau'
    t_0: int = 10  # CosineAnnealingWarmRestarts参数
    t_mult: int = 2
    eta_min: float = 1e-6
    
    # ========== 早停配置 ==========
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.1
    
    # ========== 损失配置 ==========
    loss_l1_weight: float = 0.3
    loss_huber_weight: float = 0.7
    huber_delta: float = 5.0
    
    # ========== 数据增强配置 ==========
    color_jitter_brightness: float = 0.3
    color_jitter_contrast: float = 0.3
    color_jitter_saturation: float = 0.3
    color_jitter_hue: float = 0.05
    channel_dropout_p: float = 0.25
    channel_dropout_channel: str = "blue"
    jpeg_quality_range: Tuple[int, int] = (70, 100)
    gaussian_blur_p: float = 0.2
    use_shadow: bool = True
    
    # ========== 硬件配置 ==========
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    num_workers: int = 0  # Windows兼容性（多进程问题）
    pin_memory: bool = True
    use_amp: bool = True  # 混合精度训练
    
    # ========== 保存配置 ==========
    checkpoint_dir: str = "checkpoints"
    results_dir: str = "results/b1"
    log_dir: str = "logs"
    save_best_only: bool = True
    save_frequency: int = 5  # 每5个epoch保存一次
    
    # ========== 实验配置 ==========
    experiment_name: str = "b1_baseline"
    seed: int = 42
    
    def __post_init__(self):
        """验证配置"""
        assert self.train_ratio + self.val_ratio + self.test_ratio == 1.0, "数据划分比例之和必须为1"
        assert self.loss_l1_weight + self.loss_huber_weight == 1.0, "损失权重之和必须为1"
        
        # 创建目录
        from pathlib import Path
        Path(self.checkpoint_dir).mkdir(exist_ok=True)
        Path(self.results_dir).mkdir(parents=True, exist_ok=True)
        Path(self.log_dir).mkdir(exist_ok=True)
    
    def to_dict(self):
        """转换为字典"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


@dataclass
class QuickTestConfig(Config):
    """快速测试配置（5 epoch验证）"""
    epochs: int = 5
    batch_size: int = 8
    early_stopping_patience: int = 999  # 禁用早停
    save_frequency: int = 2
    experiment_name: str = "b1_quick_test"


@dataclass
class FullTrainingConfig(Config):
    """完整训练配置"""
    epochs: int = 50
    batch_size: int = 16
    experiment_name: str = "b1_full_training"


@dataclass
class StrongB1Config(Config):
    """强化B1配置：更高输入分辨率、冻结热身、ROI与正则优化"""
    input_size: int = 448  # 增加到448获得更多细节
    pretrained: bool = True
    dropout_p: float = 0.15  # 降低dropout保留更多特征
    batch_size: int = 10  # 略降以适应更大输入
    learning_rate: float = 4e-4  # 略降学习率更稳定
    weight_decay: float = 1e-4  # 降低正则化强度
    epochs: int = 80  # 增加训练轮数
    gradient_accumulation: int = 3  # 有效batch=10*3=30
    early_stopping_patience: int = 15  # 更大耐心
    experiment_name: str = "b1_strong_training"
    save_frequency: int = 5
    
    # 优化数据增强
    color_jitter_brightness: float = 0.2
    color_jitter_contrast: float = 0.2
    color_jitter_saturation: float = 0.2
    gaussian_blur_p: float = 0.15


@dataclass
class QuickStrongTestConfig(Config):
    """快速测试强化配置（3 epoch验证脚本可用性）"""
    input_size: int = 448
    pretrained: bool = True
    dropout_p: float = 0.15
    batch_size: int = 4  # 小batch快速测试
    learning_rate: float = 4e-4
    weight_decay: float = 1e-4
    epochs: int = 3  # 仅3个epoch测试
    gradient_accumulation: int = 2
    early_stopping_patience: int = 999  # 禁用早停
    save_frequency: int = 1
    experiment_name: str = "b1_quick_strong_test"


if __name__ == "__main__":
    """测试配置"""
    
    # 默认配置
    config = Config()
    print("默认配置:")
    for k, v in config.to_dict().items():
        print(f"  {k}: {v}")
    
    print("\n快速测试配置:")
    quick_config = QuickTestConfig()
    print(f"  epochs: {quick_config.epochs}")
    print(f"  batch_size: {quick_config.batch_size}")
    print(f"  experiment_name: {quick_config.experiment_name}")
