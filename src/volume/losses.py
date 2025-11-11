#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
损失函数模块
用于体积回归任务的混合损失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


class VolumeLoss(nn.Module):
    """
    体积回归混合损失
    
    组合L1和Huber损失:
        - L1: 直接优化MAE目标
        - Huber: 对异常值鲁棒
    """
    
    def __init__(
        self,
        l1_weight: float = 0.3,
        huber_weight: float = 0.7,
        huber_delta: float = 5.0
    ):
        """
        初始化损失函数
        
        Args:
            l1_weight: L1损失权重
            huber_weight: Huber损失权重
            huber_delta: Huber损失的阈值（升）
        """
        super().__init__()
        
        self.l1_weight = l1_weight
        self.huber_weight = huber_weight
        self.huber_delta = huber_delta
        
        # 验证权重之和
        assert abs(l1_weight + huber_weight - 1.0) < 1e-6, "权重之和必须为1"
    
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算损失
        
        Args:
            pred: 预测体积 [B] 或 [B, 1]
            target: 真实体积 [B] 或 [B, 1]
        
        Returns:
            total_loss: 总损失
            loss_dict: 各项损失的字典
        """
        # 确保形状一致
        if pred.dim() == 2:
            pred = pred.squeeze(1)
        if target.dim() == 2:
            target = target.squeeze(1)
        
        # L1损失
        l1_loss = F.l1_loss(pred, target)
        
        # Huber损失（smooth L1）
        huber_loss = F.smooth_l1_loss(pred, target, beta=self.huber_delta)
        
        # 总损失
        total_loss = self.l1_weight * l1_loss + self.huber_weight * huber_loss
        
        # 详细损失信息
        loss_dict = {
            'total': total_loss.item(),
            'l1': l1_loss.item(),
            'huber': huber_loss.item()
        }
        
        return total_loss, loss_dict


class PercentageLoss(nn.Module):
    """
    相对百分比损失（MAPE）
    适用于不同量级的体积预测
    """
    
    def __init__(self, epsilon: float = 1e-6):
        """
        Args:
            epsilon: 防止除零的小常数
        """
        super().__init__()
        self.epsilon = epsilon
    
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        计算MAPE损失
        
        Args:
            pred: 预测体积 [B]
            target: 真实体积 [B]
        
        Returns:
            mape_loss: 平均绝对百分比误差
        """
        if pred.dim() == 2:
            pred = pred.squeeze(1)
        if target.dim() == 2:
            target = target.squeeze(1)
        
        # MAPE = mean(|pred - target| / (target + epsilon))
        mape = torch.mean(torch.abs(pred - target) / (target + self.epsilon))
        
        return mape


class AdaptiveVolumeLoss(nn.Module):
    """
    自适应体积损失
    根据体积大小动态调整权重
    """
    
    def __init__(
        self,
        low_threshold: float = 50.0,
        high_threshold: float = 120.0
    ):
        """
        Args:
            low_threshold: 低体积阈值（升）
            high_threshold: 高体积阈值（升）
        """
        super().__init__()
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
    
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算自适应损失
        
        Args:
            pred: 预测体积 [B]
            target: 真实体积 [B]
        
        Returns:
            total_loss: 总损失
            loss_dict: 损失详情
        """
        if pred.dim() == 2:
            pred = pred.squeeze(1)
        if target.dim() == 2:
            target = target.squeeze(1)
        
        # 根据体积大小分组
        low_mask = target < self.low_threshold
        mid_mask = (target >= self.low_threshold) & (target < self.high_threshold)
        high_mask = target >= self.high_threshold
        
        total_loss = 0
        loss_dict = {}
        
        # 低体积：使用L1（绝对误差更重要）
        if low_mask.sum() > 0:
            low_loss = F.l1_loss(pred[low_mask], target[low_mask])
            total_loss += low_loss * (low_mask.sum().float() / len(target))
            loss_dict['low'] = low_loss.item()
        
        # 中体积：使用Huber（平衡）
        if mid_mask.sum() > 0:
            mid_loss = F.smooth_l1_loss(pred[mid_mask], target[mid_mask], beta=5.0)
            total_loss += mid_loss * (mid_mask.sum().float() / len(target))
            loss_dict['mid'] = mid_loss.item()
        
        # 高体积：使用相对损失（百分比误差更重要）
        if high_mask.sum() > 0:
            high_loss = torch.mean(
                torch.abs(pred[high_mask] - target[high_mask]) / (target[high_mask] + 1e-6)
            )
            total_loss += high_loss * (high_mask.sum().float() / len(target))
            loss_dict['high'] = high_loss.item()
        
        loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict


def volume_loss(
    pred: torch.Tensor, 
    target: torch.Tensor,
    l1_weight: float = 0.3,
    huber_weight: float = 0.7,
    huber_delta: float = 5.0
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    函数式API - 体积损失（向后兼容）
    
    Args:
        pred: 预测体积 [B] 或 [B, 1]
        target: 真实体积 [B] 或 [B, 1]
        l1_weight: L1损失权重
        huber_weight: Huber损失权重
        huber_delta: Huber阈值
    
    Returns:
        total_loss: 总损失
        loss_dict: 损失详情
    """
    loss_fn = VolumeLoss(l1_weight, huber_weight, huber_delta)
    return loss_fn(pred, target)


if __name__ == "__main__":
    """测试损失函数"""
    
    # 创建测试数据
    batch_size = 16
    pred = torch.rand(batch_size) * 200  # 0-200L
    target = torch.rand(batch_size) * 200
    
    print("="*60)
    print("损失函数测试")
    print("="*60)
    
    # 测试VolumeLoss
    print("\n1. VolumeLoss (L1 + Huber)")
    loss_fn = VolumeLoss(l1_weight=0.3, huber_weight=0.7, huber_delta=5.0)
    total_loss, loss_dict = loss_fn(pred, target)
    print(f"   Total Loss: {total_loss.item():.4f}")
    print(f"   L1 Loss: {loss_dict['l1']:.4f}")
    print(f"   Huber Loss: {loss_dict['huber']:.4f}")
    
    # 测试PercentageLoss
    print("\n2. PercentageLoss (MAPE)")
    mape_fn = PercentageLoss()
    mape_loss = mape_fn(pred, target)
    print(f"   MAPE Loss: {mape_loss.item():.4f}")
    
    # 测试AdaptiveVolumeLoss
    print("\n3. AdaptiveVolumeLoss (分层加权)")
    adaptive_fn = AdaptiveVolumeLoss(low_threshold=50.0, high_threshold=120.0)
    adaptive_loss, adaptive_dict = adaptive_fn(pred, target)
    print(f"   Total Loss: {adaptive_loss.item():.4f}")
    for k, v in adaptive_dict.items():
        if k != 'total':
            print(f"   {k.capitalize()} Volume Loss: {v:.4f}")
    
    print("\n" + "="*60)
