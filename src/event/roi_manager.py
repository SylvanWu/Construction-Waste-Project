"""
ROI管理模块
负责管理垃圾桶的ROI区域并检测物体接触
"""

import numpy as np
import cv2
from loguru import logger
from typing import Tuple, Optional, List
from .config import config
from .model_loader import DetectedObject


class ROIManager:
    """ROI区域管理类"""
    
    def __init__(self):
        """初始化ROI管理器"""
        self.roi_mask = None
        self.roi_contour = None
        self.roi_bbox = None
        self.roi_center = None
        self.roi_area = 0
        
        # 配置参数
        self.bin_class_id = config.roi_config["bin_class_id"]
        self.overlap_threshold = config.roi_config["overlap_threshold"]
        self.use_center_point = config.roi_config["use_center_point"]
        
        logger.info(f"ROI管理器初始化完成，桶类别ID: {self.bin_class_id}")
    
    def update_roi(self, detection_results: dict, image_shape: Tuple[int, int]):
        """
        根据检测结果更新ROI区域
        
        Args:
            detection_results: 模型检测结果
            image_shape: 图像尺寸 (height, width)
        """
        try:
            # 查找桶的检测结果
            bin_mask = self._find_bin_mask(detection_results)
            
            if bin_mask is not None:
                self.roi_mask = bin_mask
                self._calculate_roi_properties(image_shape)
                logger.debug("ROI区域更新成功")
            else:
                logger.warning("未检测到桶，无法更新ROI区域")
                
        except Exception as e:
            logger.error(f"更新ROI区域失败: {e}")
    
    def _find_bin_mask(self, detection_results: dict) -> Optional[np.ndarray]:
        """
        从检测结果中找到桶的掩码
        
        Args:
            detection_results: 检测结果
            
        Returns:
            桶的分割掩码，如果未找到则返回None
        """
        if not detection_results["masks"] or not detection_results["classes"]:
            return None
        
        # 查找所有桶的检测结果
        bin_masks = []
        for i, class_id in enumerate(detection_results["classes"]):
            if class_id == self.bin_class_id and detection_results["masks"][i] is not None:
                bin_masks.append(detection_results["masks"][i])
        
        if not bin_masks:
            return None
        
        # 如果有多个桶，合并掩码
        if len(bin_masks) == 1:
            return bin_masks[0]
        else:
            # 合并多个桶的掩码
            combined_mask = np.zeros_like(bin_masks[0])
            for mask in bin_masks:
                combined_mask = np.logical_or(combined_mask, mask)
            return combined_mask.astype(np.uint8)
    
    def _calculate_roi_properties(self, image_shape: Tuple[int, int]):
        """
        计算ROI区域的属性
        
        Args:
            image_shape: 图像尺寸 (height, width)
        """
        if self.roi_mask is None:
            return
        
        # 确保掩码尺寸正确
        if self.roi_mask.shape != image_shape:
            self.roi_mask = cv2.resize(
                self.roi_mask.astype(np.uint8), 
                (image_shape[1], image_shape[0])
            )
        
        # 计算轮廓
        contours, _ = cv2.findContours(
            self.roi_mask.astype(np.uint8), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if contours:
            # 找到最大的轮廓
            self.roi_contour = max(contours, key=cv2.contourArea)
            
            # 计算边界框
            x, y, w, h = cv2.boundingRect(self.roi_contour)
            self.roi_bbox = (x, y, x + w, y + h)
            
            # 计算中心点
            M = cv2.moments(self.roi_contour)
            if M["m00"] != 0:
                self.roi_center = (
                    int(M["m10"] / M["m00"]),
                    int(M["m01"] / M["m00"])
                )
            else:
                self.roi_center = (x + w // 2, y + h // 2)
            
            # 计算面积
            self.roi_area = cv2.contourArea(self.roi_contour)
            
            logger.debug(f"ROI属性计算完成 - 中心: {self.roi_center}, 面积: {self.roi_area}")
    
    def is_object_in_roi(self, obj: DetectedObject) -> bool:
        """
        检测物体是否在ROI区域内
        
        Args:
            obj: 检测到的物体
            
        Returns:
            True如果物体在ROI内，否则False
        """
        if self.roi_mask is None:
            return False
        
        # 跳过桶本身
        if obj.class_id == self.bin_class_id:
            return False
        
        # 方法1: 重叠面积判定
        overlap_result = self._check_overlap_area(obj)
        
        # 方法2: 中心点判定
        center_result = True
        if self.use_center_point:
            center_result = self._check_center_point(obj)
        
        # 组合判定结果
        in_roi = overlap_result and center_result
        
        # 移除详细的ROI检测日志以精简输出
        
        return in_roi
    
    def _check_overlap_area(self, obj: DetectedObject) -> bool:
        """
        基于重叠面积检测物体是否在ROI内
        
        Args:
            obj: 检测到的物体
            
        Returns:
            True如果重叠超过阈值
        """
        if obj.mask is None:
            # 如果没有掩码，使用边界框近似
            return self._check_bbox_overlap(obj)
        
        try:
            # 确保掩码尺寸匹配
            obj_mask = obj.mask
            if obj_mask.shape != self.roi_mask.shape:
                obj_mask = cv2.resize(
                    obj_mask.astype(np.uint8),
                    (self.roi_mask.shape[1], self.roi_mask.shape[0])
                )
            
            # 计算重叠区域
            overlap_mask = np.logical_and(obj_mask, self.roi_mask)
            overlap_area = np.sum(overlap_mask)
            obj_area = np.sum(obj_mask)
            
            if obj_area == 0:
                return False
            
            # 计算重叠比例
            overlap_ratio = overlap_area / obj_area
            
            # 所有类别使用相同的重叠阈值
            threshold = self.overlap_threshold  # 统一使用10%阈值
            
            return overlap_ratio >= threshold
            
        except Exception as e:
            logger.warning(f"重叠面积计算失败，回退到边界框方法: {e}")
            return self._check_bbox_overlap(obj)
    
    def _check_bbox_overlap(self, obj: DetectedObject) -> bool:
        """
        基于边界框检测重叠
        
        Args:
            obj: 检测到的物体
            
        Returns:
            True如果边界框与ROI有重叠
        """
        if obj.bbox is None or self.roi_bbox is None:
            return False
        
        # 检查边界框重叠
        obj_x1, obj_y1, obj_x2, obj_y2 = obj.bbox
        roi_x1, roi_y1, roi_x2, roi_y2 = self.roi_bbox
        
        # 计算重叠区域
        overlap_x1 = max(obj_x1, roi_x1)
        overlap_y1 = max(obj_y1, roi_y1)
        overlap_x2 = min(obj_x2, roi_x2)
        overlap_y2 = min(obj_y2, roi_y2)
        
        if overlap_x2 <= overlap_x1 or overlap_y2 <= overlap_y1:
            return False
        
        # 计算重叠面积和物体面积
        overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
        obj_area = (obj_x2 - obj_x1) * (obj_y2 - obj_y1)
        
        if obj_area == 0:
            return False
        
        overlap_ratio = overlap_area / obj_area
        return overlap_ratio >= self.overlap_threshold
    
    def _check_center_point(self, obj: DetectedObject) -> bool:
        """
        检测物体中心点是否在ROI内
        
        Args:
            obj: 检测到的物体
            
        Returns:
            True如果中心点在ROI内
        """
        if self.roi_mask is None or obj.center is None:
            return False
        
        try:
            x, y = obj.center
            # 确保坐标在图像范围内
            if (0 <= x < self.roi_mask.shape[1] and 
                0 <= y < self.roi_mask.shape[0]):
                return self.roi_mask[y, x] > 0
            return False
        except Exception as e:
            logger.warning(f"中心点检测失败: {e}")
            return False
    
    def get_roi_info(self) -> dict:
        """
        获取ROI区域信息
        
        Returns:
            ROI信息字典
        """
        return {
            "has_roi": self.roi_mask is not None,
            "roi_center": self.roi_center,
            "roi_area": self.roi_area,
            "roi_bbox": self.roi_bbox,
            "bin_class_id": self.bin_class_id,
            "overlap_threshold": self.overlap_threshold,
            "use_center_point": self.use_center_point
        }
    
    def draw_roi(self, image: np.ndarray) -> np.ndarray:
        """
        在图像上绘制ROI区域
        
        Args:
            image: 输入图像
            
        Returns:
            绘制了ROI的图像
        """
        if self.roi_mask is None:
            return image
        
        try:
            result = image.copy()
            
            # 创建彩色的ROI覆盖层
            roi_color = (0, 255, 0)  # 绿色
            alpha = config.visualization_config["roi_alpha"]
            
            # 创建彩色掩码
            colored_mask = np.zeros_like(image)
            colored_mask[self.roi_mask > 0] = roi_color
            
            # 混合图像
            result = cv2.addWeighted(result, 1 - alpha, colored_mask, alpha, 0)
            
            # 绘制轮廓
            if self.roi_contour is not None:
                cv2.drawContours(
                    result, 
                    [self.roi_contour], 
                    -1, 
                    roi_color, 
                    thickness=config.visualization_config["line_thickness"]
                )
            
            # 绘制中心点
            if self.roi_center is not None:
                cv2.circle(
                    result, 
                    self.roi_center, 
                    5, 
                    roi_color, 
                    -1
                )
            
            return result
            
        except Exception as e:
            logger.error(f"绘制ROI失败: {e}")
            return image
    
    def reset_roi(self):
        """重置ROI区域"""
        self.roi_mask = None
        self.roi_contour = None
        self.roi_bbox = None
        self.roi_center = None
        self.roi_area = 0
        logger.debug("ROI区域已重置")
