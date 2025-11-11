"""
YOLO检测模型封装
基于InBinFunction的ModelLoader简化版
"""

import torch
import numpy as np
from ultralytics import YOLO
from loguru import logger
from typing import List, Dict, Any, Tuple
from pathlib import Path


class DetectionModel:
    """YOLO11检测模型类"""
    
    def __init__(self, model_path: str, device: str = "auto", confidence_threshold: float = 0.5):
        """
        初始化检测模型
        
        Args:
            model_path: 模型文件路径
            device: 设备（"auto", "cuda", "cpu"）
            confidence_threshold: 置信度阈值
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = None
        
        # 设置设备
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # 验证模型文件
        if not Path(model_path).exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        self.load_model()
    
    def load_model(self):
        """加载YOLO模型"""
        try:
            logger.info(f"正在加载YOLO模型: {self.model_path}")
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            logger.info(f"模型加载成功，使用设备: {self.device}")
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
    
    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        """
        对图像进行推理
        
        Args:
            image: 输入图像 (H, W, 3) BGR格式
            
        Returns:
            检测结果字典
        """
        try:
            results = self.model(
                image,
                conf=self.confidence_threshold,
                iou=0.3,  # NMS的IoU阈值，更严格以减少重叠框
                verbose=False
            )
            return self._parse_results(results[0])
        except Exception as e:
            logger.error(f"模型推理失败: {e}")
            raise
    
    def _parse_results(self, result) -> Dict[str, Any]:
        """解析YOLO推理结果"""
        parsed = {
            "boxes": [],
            "masks": [],
            "classes": [],
            "confidences": [],
            "image_shape": None
        }
        
        if hasattr(result, 'orig_shape'):
            parsed["image_shape"] = result.orig_shape
        
        if result.boxes is None or len(result.boxes) == 0:
            return parsed
        
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        
        masks = None
        if hasattr(result, 'masks') and result.masks is not None:
            masks = result.masks.data.cpu().numpy()
        
        for i in range(len(boxes)):
            parsed["boxes"].append(boxes[i])
            parsed["classes"].append(classes[i])
            parsed["confidences"].append(confidences[i])
            
            if masks is not None:
                parsed["masks"].append(masks[i])
            else:
                parsed["masks"].append(None)
        
        logger.debug(f"检测到 {len(parsed['boxes'])} 个物体")
        return parsed


class DetectedObject:
    """检测物体类"""
    
    def __init__(self, 
                 obj_id: int,
                 class_id: int, 
                 bbox: np.ndarray,
                 mask: np.ndarray = None,
                 confidence: float = 0.0,
                 class_names: List[str] = None):
        self.id = obj_id
        self.class_id = class_id
        self.bbox = bbox
        self.mask = mask
        self.confidence = confidence
        self.class_names = class_names or []
        
        self.center = self._calculate_center()
        self.in_roi = False
        self.counted = False
        self.last_seen_frame = 0
        self.trajectory = [self.center]
        self.counted_frame = -1
        self.roi_entry_frame = -1
    
    def _calculate_center(self) -> Tuple[int, int]:
        """计算中心点"""
        if self.bbox is not None:
            x1, y1, x2, y2 = self.bbox
            return (int((x1 + x2) / 2), int((y1 + y2) / 2))
        return (0, 0)
    
    def update_position(self, bbox: np.ndarray, mask: np.ndarray = None):
        """更新位置"""
        self.bbox = bbox
        self.mask = mask
        self.center = self._calculate_center()
        self.trajectory.append(self.center)
        if len(self.trajectory) > 10:
            self.trajectory.pop(0)
    
    def get_class_name(self) -> str:
        """获取类别名称"""
        if 0 <= self.class_id < len(self.class_names):
            return self.class_names[self.class_id]
        return f"class_{self.class_id}"
    
    def calculate_iou(self, other_bbox: np.ndarray) -> float:
        """计算IoU"""
        if self.bbox is None:
            return 0.0
        
        x1 = max(self.bbox[0], other_bbox[0])
        y1 = max(self.bbox[1], other_bbox[1])
        x2 = min(self.bbox[2], other_bbox[2])
        y2 = min(self.bbox[3], other_bbox[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])
        area2 = (other_bbox[2] - other_bbox[0]) * (other_bbox[3] - other_bbox[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def distance_to(self, center: Tuple[int, int]) -> float:
        """计算距离"""
        dx = self.center[0] - center[0]
        dy = self.center[1] - center[1]
        return np.sqrt(dx * dx + dy * dy)
