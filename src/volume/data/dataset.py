#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VolumeDataset - RGB体积估计数据集
支持HDF5数据加载、时间序列划分、多种增强策略
"""

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import logging

logger = logging.getLogger(__name__)


class VolumeDataset(Dataset):
    """RGB体积估计数据集"""
    
    def __init__(
        self, 
        h5_path: str,
        csv_path: str,
        indices: List[int],
        transform=None,
        use_mask: bool = True,
        return_frame_id: bool = True
    ):
        """
        初始化数据集
        
        Args:
            h5_path: HDF5文件路径
            csv_path: CSV元数据路径
            indices: 数据索引列表（用于train/val/test划分）
            transform: 图像增强transform
            use_mask: 是否使用bin mask裁剪
            return_frame_id: 是否返回frame_id
        """
        self.h5_path = h5_path
        self.csv_path = csv_path
        self.indices = indices
        self.transform = transform
        self.use_mask = use_mask
        self.return_frame_id = return_frame_id
        
        # 加载元数据
        self.metadata = pd.read_csv(csv_path)
        
        # 懒加载HDF5（避免多进程问题）
        self.h5_file = None
        
        logger.info(f"数据集初始化完成: {len(self.indices)} 个样本")
    
    def __len__(self):
        return len(self.indices)
    
    def _get_h5_file(self):
        """懒加载HDF5文件（多进程安全）"""
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')
        return self.h5_file
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取单个样本
        
        Returns:
            dict: {
                'image': Tensor [3, H, W],
                'volume': Tensor (scalar),
                'frame_id': str (可选)
            }
        """
        real_idx = self.indices[idx]
        h5_file = self._get_h5_file()
        
        # 获取frame_id
        frame_id = self.metadata.iloc[real_idx]['frame_id']
        
        # 加载RGB图像 (每帧单独存储在group中)
        rgb = h5_file['rgb_images'][frame_id][:]  # [H, W, 3] uint8
        
        # 加载体积标签 (存储为[volume, fill_percentage]格式)
        volume_data = h5_file['volume_labels'][frame_id][:]
        volume = float(volume_data[0])  # 取第一个值（体积）
        
        # 可选：加载bin mask
        if self.use_mask and 'bin_masks' in h5_file:
            bin_mask = h5_file['bin_masks'][real_idx]  # [H, W]
            # TODO: 可以根据mask裁剪ROI以提升性能
        
        # 应用transform
        if self.transform:
            rgb = self.transform(rgb)
        else:
            # 默认转换为tensor
            rgb = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        
        result = {
            'image': rgb,
            'volume': torch.tensor(volume, dtype=torch.float32)
        }
        
        # 可选：返回frame_id用于分析
        if self.return_frame_id:
            result['frame_id'] = self.metadata.iloc[real_idx]['frame_id']
        
        return result
    
    def __del__(self):
        """清理HDF5文件句柄"""
        if self.h5_file is not None:
            self.h5_file.close()


def split_dataset_temporal(
    total_frames: int,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> Tuple[List[int], List[int], List[int]]:
    """
    时间序列数据划分（模拟真实部署场景）
    
    Args:
        total_frames: 总帧数
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
    
    Returns:
        train_indices, val_indices, test_indices
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例之和必须为1"
    
    indices = list(range(total_frames))
    
    train_end = int(total_frames * train_ratio)
    val_end = train_end + int(total_frames * val_ratio)
    
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    
    logger.info(f"数据划分: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}")
    
    return train_indices, val_indices, test_indices


def split_dataset_random(
    total_frames: int,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[List[int], List[int], List[int]]:
    """
    随机数据划分（对比实验）
    
    Args:
        total_frames: 总帧数
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        seed: 随机种子
    
    Returns:
        train_indices, val_indices, test_indices
    """
    from sklearn.model_selection import train_test_split
    
    indices = list(range(total_frames))
    
    # 第一次分割：train+val vs test
    train_val_indices, test_indices = train_test_split(
        indices, 
        test_size=test_ratio, 
        random_state=seed
    )
    
    # 第二次分割：train vs val
    val_size_adjusted = val_ratio / (train_ratio + val_ratio)
    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=val_size_adjusted,
        random_state=seed
    )
    
    logger.info(f"随机划分: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}")
    
    return train_indices, val_indices, test_indices


def get_data_loaders(
    h5_path: str,
    csv_path: str,
    train_transform,
    val_transform,
    batch_size: int = 16,
    num_workers: int = 4,
    pin_memory: bool = True,
    split_mode: str = 'temporal',
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建train/val/test数据加载器
    
    Args:
        h5_path: HDF5文件路径
        csv_path: CSV文件路径
        train_transform: 训练集增强
        val_transform: 验证/测试集增强
        batch_size: batch大小
        num_workers: 数据加载线程数
        pin_memory: 是否pin memory（GPU加速）
        split_mode: 'temporal' 或 'random'
        train_ratio, val_ratio, test_ratio: 数据划分比例
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # 获取总帧数 (从CSV读取更可靠)
    import pandas as pd
    df = pd.read_csv(csv_path)
    total_frames = len(df)
    
    # 数据划分
    if split_mode == 'temporal':
        train_indices, val_indices, test_indices = split_dataset_temporal(
            total_frames, train_ratio, val_ratio, test_ratio
        )
    elif split_mode == 'random':
        train_indices, val_indices, test_indices = split_dataset_random(
            total_frames, train_ratio, val_ratio, test_ratio
        )
    else:
        raise ValueError(f"未知的split_mode: {split_mode}")
    
    # 创建数据集
    train_dataset = VolumeDataset(h5_path, csv_path, train_indices, train_transform)
    val_dataset = VolumeDataset(h5_path, csv_path, val_indices, val_transform)
    test_dataset = VolumeDataset(h5_path, csv_path, test_indices, val_transform)
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # 保证batch大小一致（混合精度训练需要）
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    logger.info(f"DataLoader创建完成: Train={len(train_loader)} batches, "
                f"Val={len(val_loader)} batches, Test={len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    """测试数据加载器"""
    import sys
    sys.path.append('..')
    
    logging.basicConfig(level=logging.INFO)
    
    # 测试参数
    h5_path = "../training_data/volume_dataset_methodA.h5"
    csv_path = "../training_data/volume_dataset_methodA.csv"
    
    # 时间序列划分
    train_idx, val_idx, test_idx = split_dataset_temporal(603)
    
    # 创建简单数据集
    dataset = VolumeDataset(h5_path, csv_path, train_idx[:10])
    
    # 测试数据加载
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        print(f"\n样本 {i}:")
        print(f"  图像形状: {sample['image'].shape}")
        print(f"  体积: {sample['volume'].item():.2f} L")
        print(f"  帧ID: {sample['frame_id']}")
