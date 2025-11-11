#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评估指标计算
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from typing import Dict


def compute_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """
    计算完整评估指标
    
    Args:
        predictions: 预测值 [N]
        targets: 真实值 [N]
    
    Returns:
        metrics: 指标字典
    """
    # 基础指标
    mae = mean_absolute_error(targets, predictions)
    
    # MAPE (平均绝对百分比误差)
    mape = np.mean(np.abs((targets - predictions) / (targets + 1e-6))) * 100
    
    # R² (决定系数)
    r2 = r2_score(targets, predictions)
    
    # RMSE (均方根误差)
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    
    # 误差分析
    errors = predictions - targets
    abs_errors = np.abs(errors)
    
    # 分层精度 (按体积范围)
    low_mask = targets < 50
    mid_mask = (targets >= 50) & (targets < 120)
    high_mask = targets >= 120
    
    metrics = {
        # 总体指标
        'mae': mae,
        'mape': mape,
        'r2': r2,
        'rmse': rmse,
        'max_error': np.max(abs_errors),
        'median_error': np.median(abs_errors),
        
        # 分层精度
        'low_volume_mae': mean_absolute_error(targets[low_mask], predictions[low_mask]) if low_mask.sum() > 0 else 0,
        'low_volume_count': low_mask.sum(),
        'mid_volume_mae': mean_absolute_error(targets[mid_mask], predictions[mid_mask]) if mid_mask.sum() > 0 else 0,
        'mid_volume_count': mid_mask.sum(),
        'high_volume_mae': mean_absolute_error(targets[high_mask], predictions[high_mask]) if high_mask.sum() > 0 else 0,
        'high_volume_count': high_mask.sum(),
        
        # 百分位数误差
        'p50_error': np.percentile(abs_errors, 50),
        'p90_error': np.percentile(abs_errors, 90),
        'p95_error': np.percentile(abs_errors, 95),
        'p99_error': np.percentile(abs_errors, 99)
    }
    
    return metrics


def print_metrics(metrics: Dict[str, float], prefix: str = ""):
    """Print evaluation metrics"""
    print(f"\n{prefix} Evaluation Metrics:")
    print(f"  MAE: {metrics['mae']:.2f} L")
    print(f"  MAPE: {metrics['mape']:.2f} %")
    print(f"  R2: {metrics['r2']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.2f} L")
    print(f"  Max Error: {metrics['max_error']:.2f} L")
    print(f"  Median Error: {metrics['median_error']:.2f} L")
    
    print(f"\nStratified Accuracy:")
    print(f"  Low Volume (<50L): MAE={metrics['low_volume_mae']:.2f}L, N={metrics['low_volume_count']}")
    print(f"  Mid Volume (50-120L): MAE={metrics['mid_volume_mae']:.2f}L, N={metrics['mid_volume_count']}")
    print(f"  High Volume (>120L): MAE={metrics['high_volume_mae']:.2f}L, N={metrics['high_volume_count']}")
    
    print(f"\nError Percentiles:")
    print(f"  P50: {metrics['p50_error']:.2f} L")
    print(f"  P90: {metrics['p90_error']:.2f} L")
    print(f"  P95: {metrics['p95_error']:.2f} L")
    print(f"  P99: {metrics['p99_error']:.2f} L")
