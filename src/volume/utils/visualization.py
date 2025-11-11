#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化工具：训练曲线、预测结果散点图、误差分布
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_training_curves(history: dict, save_dir: str):
    """
    绘制训练与验证曲线

    Args:
        history: 包含 'train_loss','val_loss','train_mae','val_mae','val_r2' 的字典
        save_dir: 保存目录
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    epochs = np.arange(1, len(history.get('train_loss', [])) + 1)

    # Loss
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history.get('train_loss', []), label='Train Loss')
    plt.plot(epochs, history.get('val_loss', []), label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training/Validation Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(save_dir) / 'curve_loss.png', dpi=200)
    plt.close()

    # MAE
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history.get('train_mae', []), label='Train MAE')
    plt.plot(epochs, history.get('val_mae', []), label='Val MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE (L)')
    plt.title('Training/Validation MAE')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(save_dir) / 'curve_mae.png', dpi=200)
    plt.close()

    # R2
    if 'val_r2' in history and len(history['val_r2']) > 0:
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, history.get('val_r2', []), label='Val R2')
        plt.xlabel('Epoch')
        plt.ylabel('R2')
        plt.title('Validation R2')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(Path(save_dir) / 'curve_r2.png', dpi=200)
        plt.close()


def plot_predictions_scatter(predictions: np.ndarray, targets: np.ndarray, save_dir: str):
    """
    绘制预测 vs 真实散点图与误差直方图
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Scatter
    plt.figure(figsize=(6, 6))
    plt.scatter(targets, predictions, s=12, alpha=0.5)
    lim = [0, max(float(np.max(targets)), float(np.max(predictions)))]
    plt.plot(lim, lim, 'r--', lw=2)
    plt.xlim(lim)
    plt.ylim(lim)
    plt.xlabel('True Volume (L)')
    plt.ylabel('Predicted Volume (L)')
    plt.title('Prediction vs Ground Truth')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(save_dir) / 'scatter_pred_vs_true.png', dpi=200)
    plt.close()

    # Error histogram
    errors = predictions - targets
    plt.figure(figsize=(8, 5))
    plt.hist(errors, bins=50, alpha=0.8, edgecolor='black')
    plt.axvline(0, color='r', linestyle='--')
    plt.xlabel('Error (Pred - True) L')
    plt.ylabel('Count')
    plt.title('Prediction Error Histogram')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(save_dir) / 'hist_error.png', dpi=200)
    plt.close()


