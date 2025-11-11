"""
体积估测器
管理基准设定和体积估测逻辑
"""

import numpy as np
import cv2
from loguru import logger
from typing import Optional, Dict, Any
from pathlib import Path


class VolumeEstimator:
    """体积估测器类"""
    
    def __init__(self, volume_model, baseline_image_path: Optional[str] = None):
        """
        初始化体积估测器
        
        Args:
            volume_model: VolumeModel实例
            baseline_image_path: 空桶基准图像路径
        """
        self.volume_model = volume_model
        self.baseline_volume = 0.0
        self.max_volume = 100.0  # 默认最大容量100L
        self.current_volume = 0.0
        self.baseline_set = False
        
        # 如果提供了基准图像，立即设置基准
        if baseline_image_path and Path(baseline_image_path).exists():
            self.set_baseline_from_image(baseline_image_path)
    
    def set_baseline_from_image(self, image_path: str, bin_mask: Optional[np.ndarray] = None) -> float:
        """
        从图像设置空桶基准
        
        Args:
            image_path: 空桶图像路径
            bin_mask: 可选的bin区域mask（注意：不建议使用，模型训练时用完整图像）
            
        Returns:
            基准体积值
        """
        try:
            logger.info(f"正在设置空桶基准: {image_path}")
            
            # 加载图像
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"无法读取基准图像: {image_path}")
            
            # 预测基准体积（不使用ROI裁剪）
            self.baseline_volume = self.volume_model.predict(image, bin_mask=None)
            self.current_volume = self.baseline_volume
            self.baseline_set = True
            
            # 更新模型的基准
            self.volume_model.set_baseline(self.baseline_volume)
            
            logger.info(f"空桶基准设置完成: {self.baseline_volume:.2f}L")
            return self.baseline_volume
            
        except Exception as e:
            logger.error(f"设置基准失败: {e}")
            self.baseline_set = False
            return 0.0
    
    def set_baseline_manual(self, baseline_volume: float):
        """
        手动设置基准体积
        
        Args:
            baseline_volume: 基准体积值
        """
        self.baseline_volume = baseline_volume
        self.current_volume = baseline_volume
        self.baseline_set = True
        self.volume_model.set_baseline(baseline_volume)
        logger.info(f"手动设置空桶基准: {baseline_volume:.2f}L")
    
    def estimate_volume(self, image: np.ndarray, bin_mask: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        估测当前体积
        
        Args:
            image: 输入图像
            bin_mask: bin区域mask（用于ROI裁剪）
            
        Returns:
            包含体积信息的字典
        """
        if not self.baseline_set:
            logger.warning("基准未设置，体积估测可能不准确")
        
        try:
            # 预测体积
            predicted_volume = self.volume_model.predict(image, bin_mask)
            self.current_volume = predicted_volume
            
            # 计算相对体积（相对于基准的增量）
            relative_volume = max(0.0, predicted_volume - self.baseline_volume)
            
            # 计算填充率
            fill_percentage = self.calculate_fill_percentage(relative_volume)
            
            result = {
                "current_volume": predicted_volume,
                "relative_volume": relative_volume,
                "baseline_volume": self.baseline_volume,
                "fill_percentage": fill_percentage,
                "max_volume": self.max_volume
            }
            
            logger.debug(f"体积估测: {predicted_volume:.2f}L "
                        f"(相对: {relative_volume:.2f}L, 填充率: {fill_percentage:.1f}%)")
            
            return result
            
        except Exception as e:
            logger.error(f"体积估测失败: {e}")
            return {
                "current_volume": self.current_volume,
                "relative_volume": 0.0,
                "baseline_volume": self.baseline_volume,
                "fill_percentage": 0.0,
                "max_volume": self.max_volume
            }
    
    def calculate_fill_percentage(self, relative_volume: float) -> float:
        """
        计算填充率
        
        Args:
            relative_volume: 相对体积（相对于基准）
            
        Returns:
            填充率（百分比）
        """
        if self.max_volume <= 0:
            return 0.0
        return (relative_volume / self.max_volume) * 100.0
    
    def set_max_volume(self, max_volume: float):
        """设置最大容量"""
        self.max_volume = max_volume
        logger.info(f"设置最大容量: {max_volume:.2f}L")
    
    def reset(self):
        """重置估测器"""
        self.current_volume = self.baseline_volume
        logger.info("体积估测器已重置")
    
    def get_info(self) -> Dict[str, Any]:
        """获取估测器信息"""
        return {
            "baseline_volume": self.baseline_volume,
            "current_volume": self.current_volume,
            "max_volume": self.max_volume,
            "baseline_set": self.baseline_set,
            "fill_percentage": self.calculate_fill_percentage(
                max(0.0, self.current_volume - self.baseline_volume)
            )
        }
