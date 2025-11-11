"""
物体跟踪模块
负责跟踪物体在连续帧间的运动，避免重复计数
"""

import numpy as np
from loguru import logger
from typing import List, Dict, Tuple, Optional
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.detection_model import DetectedObject


class ObjectTracker:
    """物体跟踪器类"""
    
    def __init__(self, max_distance=50, iou_threshold=0.3, disappear_frames=5, use_iou_matching=True):
        """初始化物体跟踪器"""
        self.tracked_objects: Dict[int, DetectedObject] = {}
        self.next_id = 1
        self.current_frame = 0
        
        # 配置参数
        self.max_distance = max_distance
        self.iou_threshold = iou_threshold
        self.disappear_frames = disappear_frames
        self.use_iou_matching = use_iou_matching
        
        logger.info("物体跟踪器初始化完成")
    
    def update(self, detection_results: dict) -> List[DetectedObject]:
        """
        更新跟踪器状态
        
        Args:
            detection_results: 当前帧的检测结果
            
        Returns:
            更新后的跟踪物体列表
        """
        self.current_frame += 1
        
        # 从检测结果创建候选物体
        detected_objects = self._create_detected_objects(detection_results)
        
        # 进行物体匹配和更新
        self._match_and_update(detected_objects)
        
        # 清理失踪的物体
        self._cleanup_lost_objects()
        
        # 返回当前所有跟踪的物体
        current_objects = list(self.tracked_objects.values())
        
        logger.debug(f"帧 {self.current_frame}: 跟踪 {len(current_objects)} 个物体")
        return current_objects
    
    def _create_detected_objects(self, detection_results: dict) -> List[DetectedObject]:
        """
        从检测结果创建DetectedObject实例
        
        Args:
            detection_results: 检测结果
            
        Returns:
            检测物体列表
        """
        detected_objects = []
        
        if not detection_results["boxes"]:
            return detected_objects
        
        for i in range(len(detection_results["boxes"])):
            # 获取检测信息
            bbox = detection_results["boxes"][i]
            class_id = detection_results["classes"][i]
            confidence = detection_results["confidences"][i]
            mask = detection_results["masks"][i] if detection_results["masks"] else None
            
            # 创建临时物体对象（ID稍后分配）
            obj = DetectedObject(
                obj_id=-1,  # 临时ID
                class_id=class_id,
                bbox=bbox,
                mask=mask,
                confidence=confidence
            )
            
            detected_objects.append(obj)
        
        return detected_objects
    
    def _match_and_update(self, detected_objects: List[DetectedObject]):
        """
        匹配检测物体与已跟踪物体，并更新状态
        
        Args:
            detected_objects: 当前帧检测到的物体
        """
        if not detected_objects:
            return
        
        # 获取当前跟踪的物体
        tracked_list = list(self.tracked_objects.values())
        
        if not tracked_list:
            # 如果没有已跟踪的物体，直接添加所有检测物体
            for obj in detected_objects:
                obj.id = self._get_next_id()
                obj.last_seen_frame = self.current_frame
                self.tracked_objects[obj.id] = obj
            return
        
        # 计算匹配矩阵
        cost_matrix = self._calculate_cost_matrix(tracked_list, detected_objects)
        
        # 执行匹配
        matches, unmatched_tracks, unmatched_detections = self._hungarian_matching(
            cost_matrix, tracked_list, detected_objects
        )
        
        # 更新匹配的物体
        for track_idx, detection_idx in matches:
            tracked_obj = tracked_list[track_idx]
            detected_obj = detected_objects[detection_idx]
            
            # 更新跟踪物体的位置
            tracked_obj.update_position(detected_obj.bbox, detected_obj.mask)
            tracked_obj.confidence = detected_obj.confidence
            tracked_obj.last_seen_frame = self.current_frame
        
        # 创建新的跟踪物体（添加重复检查）
        for detection_idx in unmatched_detections:
            obj = detected_objects[detection_idx]
            
            # 检查是否与现有物体过于相似（防止重复创建）
            if self._is_duplicate_object(obj, tracked_list):
                logger.info(f"🚫 跳过重复物体: {obj.get_class_name()}")
                continue
            
            # 额外检查：与所有已跟踪物体（包括失踪的）进行比较
            all_tracked_objects = list(self.tracked_objects.values())
            if self._is_duplicate_object(obj, all_tracked_objects):
                logger.info(f"🚫 跳过与已跟踪物体重复的新物体: {obj.get_class_name()}")
                continue
            
            obj.id = self._get_next_id()
            obj.last_seen_frame = self.current_frame
            self.tracked_objects[obj.id] = obj
            logger.info(f"✅ 创建新物体: ID={obj.id}, 类别={obj.get_class_name()}")
        
        logger.debug(f"匹配结果: {len(matches)} 个匹配, "
                    f"{len(unmatched_tracks)} 个失踪, "
                    f"{len(unmatched_detections)} 个新物体")
    
    def _calculate_cost_matrix(self, 
                              tracked_objects: List[DetectedObject], 
                              detected_objects: List[DetectedObject]) -> np.ndarray:
        """
        计算跟踪物体与检测物体之间的匹配代价矩阵
        
        Args:
            tracked_objects: 已跟踪的物体
            detected_objects: 新检测的物体
            
        Returns:
            代价矩阵 (N_tracked, N_detected)
        """
        if not tracked_objects or not detected_objects:
            return np.array([])
        
        cost_matrix = np.full(
            (len(tracked_objects), len(detected_objects)), 
            float('inf')
        )
        
        for i, tracked_obj in enumerate(tracked_objects):
            for j, detected_obj in enumerate(detected_objects):
                # 只匹配相同类别的物体
                if tracked_obj.class_id != detected_obj.class_id:
                    continue
                
                # 计算距离代价
                distance_cost = self._calculate_distance_cost(tracked_obj, detected_obj)
                
                # 计算IoU代价（如果启用）
                iou_cost = 0
                if self.use_iou_matching:
                    iou_cost = self._calculate_iou_cost(tracked_obj, detected_obj)
                
                # 组合代价（距离权重更高）
                total_cost = 0.7 * distance_cost + 0.3 * iou_cost
                
                # 设置代价阈值
                if distance_cost < self.max_distance:
                    cost_matrix[i, j] = total_cost
        
        return cost_matrix
    
    def _calculate_distance_cost(self, 
                                tracked_obj: DetectedObject, 
                                detected_obj: DetectedObject) -> float:
        """
        计算基于中心点距离的代价
        
        Args:
            tracked_obj: 跟踪物体
            detected_obj: 检测物体
            
        Returns:
            距离代价
        """
        return tracked_obj.distance_to(detected_obj.center)
    
    def _calculate_iou_cost(self, 
                           tracked_obj: DetectedObject, 
                           detected_obj: DetectedObject) -> float:
        """
        计算基于IoU的代价
        
        Args:
            tracked_obj: 跟踪物体
            detected_obj: 检测物体
            
        Returns:
            IoU代价（1 - IoU）
        """
        iou = tracked_obj.calculate_iou(detected_obj.bbox)
        return 1.0 - iou  # IoU越高，代价越低
    
    def _hungarian_matching(self, 
                           cost_matrix: np.ndarray,
                           tracked_objects: List[DetectedObject],
                           detected_objects: List[DetectedObject]) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        使用简化的匈牙利算法进行匹配
        
        Args:
            cost_matrix: 代价矩阵
            tracked_objects: 跟踪物体列表
            detected_objects: 检测物体列表
            
        Returns:
            (匹配对, 未匹配的跟踪, 未匹配的检测)
        """
        if cost_matrix.size == 0:
            return [], list(range(len(tracked_objects))), list(range(len(detected_objects)))
        
        # 简化版匹配算法：贪心匹配
        matches = []
        used_tracks = set()
        used_detections = set()
        
        # 按代价排序所有可能的匹配
        valid_pairs = []
        for i in range(cost_matrix.shape[0]):
            for j in range(cost_matrix.shape[1]):
                if cost_matrix[i, j] < float('inf'):
                    valid_pairs.append((cost_matrix[i, j], i, j))
        
        valid_pairs.sort(key=lambda x: x[0])  # 按代价升序排序
        
        # 贪心选择最优匹配
        for cost, track_idx, detection_idx in valid_pairs:
            if track_idx not in used_tracks and detection_idx not in used_detections:
                # 更严格的匹配条件
                tracked_obj = tracked_objects[track_idx]
                detected_obj = detected_objects[detection_idx]
                
                # 1. 距离阈值检查
                if cost >= self.max_distance:
                    continue
                
                # 2. IoU阈值检查（如果启用）
                if self.use_iou_matching:
                    iou = tracked_obj.calculate_iou(detected_obj.bbox)
                    if iou < self.iou_threshold:
                        continue
                
                # 3. 类别一致性检查
                if tracked_obj.class_id != detected_obj.class_id:
                    continue
                
                matches.append((track_idx, detection_idx))
                used_tracks.add(track_idx)
                used_detections.add(detection_idx)
        
        # 计算未匹配的索引
        unmatched_tracks = [i for i in range(len(tracked_objects)) if i not in used_tracks]
        unmatched_detections = [i for i in range(len(detected_objects)) if i not in used_detections]
        
        return matches, unmatched_tracks, unmatched_detections
    
    def _is_duplicate_object(self, new_obj: DetectedObject, existing_objects: List[DetectedObject]) -> bool:
        """
        检查新物体是否与现有物体重复
        
        Args:
            new_obj: 新检测到的物体
            existing_objects: 现有跟踪物体列表
            
        Returns:
            True如果认为是重复物体
        """
        import os
        debug_mode = os.getenv('TRACKER_DEBUG', '0') == '1'
        
        if debug_mode and existing_objects:
            logger.debug(f"检查物体 {new_obj.get_class_name()} 是否重复，对比 {len(existing_objects)} 个已存在物体")
        
        for existing_obj in existing_objects:
            # 1. 类别必须相同
            if existing_obj.class_id != new_obj.class_id:
                continue
            
            # 2. 距离检查 - 使用2倍max_distance作为阈值（更宽松）
            distance = existing_obj.distance_to(new_obj.center)
            if distance > self.max_distance * 2.0:  # 使用2倍max_distance，允许更大移动
                if debug_mode:
                    logger.debug(f"  与ID {existing_obj.id} 距离太远: {distance:.1f} > {self.max_distance * 2.0}")
                continue
            
            # 3. IoU检查 - 放宽IoU阈值（允许更多变形和角度变化）
            iou = existing_obj.calculate_iou(new_obj.bbox)
            if iou < self.iou_threshold * 0.3:  # 放宽到30%的IoU阈值
                if debug_mode:
                    logger.debug(f"  与ID {existing_obj.id} IoU太低: {iou:.2f} < {self.iou_threshold * 0.3:.2f}")
                continue
            
            # 4. 时间检查 - 扩大时间窗口到10帧
            frames_since_seen = self.current_frame - existing_obj.last_seen_frame
            if frames_since_seen < 10:  # 最近10帧内消失的物体都认为是重复
                logger.info(f"🚫 检测到重复物体: {new_obj.get_class_name()}, "
                           f"与ID {existing_obj.id} 相似 (距离: {distance:.1f}, IoU: {iou:.2f}, "
                           f"消失帧数: {frames_since_seen})")
                return True
            else:
                if debug_mode:
                    logger.debug(f"  与ID {existing_obj.id} 时间窗口超时: {frames_since_seen} >= 10")
            
            # 5. 额外检查：即使物体还在跟踪中，如果距离和IoU都很接近，也认为是重复
            if distance < self.max_distance * 0.5 and iou > self.iou_threshold * 0.5:
                logger.info(f"🚫 检测到高度相似的物体: {new_obj.get_class_name()}, "
                           f"与ID {existing_obj.id} 相似 (距离: {distance:.1f}, IoU: {iou:.2f})")
                return True
        
        return False
    
    def _cleanup_lost_objects(self):
        """清理失踪超过阈值的物体"""
        lost_ids = []
        
        for obj_id, obj in self.tracked_objects.items():
            frames_since_seen = self.current_frame - obj.last_seen_frame
            if frames_since_seen > self.disappear_frames:
                lost_ids.append(obj_id)
        
        for obj_id in lost_ids:
            lost_obj = self.tracked_objects.pop(obj_id)
            logger.debug(f"物体 {obj_id} ({lost_obj.get_class_name()}) 失踪超过 {self.disappear_frames} 帧，已移除")
    
    def _get_next_id(self) -> int:
        """获取下一个可用的物体ID"""
        current_id = self.next_id
        self.next_id += 1
        return current_id
    
    def get_object_by_id(self, obj_id: int) -> Optional[DetectedObject]:
        """
        根据ID获取跟踪物体
        
        Args:
            obj_id: 物体ID
            
        Returns:
            物体对象，如果不存在返回None
        """
        return self.tracked_objects.get(obj_id)
    
    def get_all_objects(self) -> List[DetectedObject]:
        """
        获取所有当前跟踪的物体
        
        Returns:
            物体列表
        """
        return list(self.tracked_objects.values())
    
    def get_tracking_info(self) -> dict:
        """
        获取跟踪器信息
        
        Returns:
            跟踪器信息字典
        """
        active_objects = len(self.tracked_objects)
        class_distribution = {}
        
        for obj in self.tracked_objects.values():
            class_name = obj.get_class_name()
            class_distribution[class_name] = class_distribution.get(class_name, 0) + 1
        
        return {
            "current_frame": self.current_frame,
            "active_objects": active_objects,
            "next_id": self.next_id,
            "class_distribution": class_distribution,
            "config": {
                "max_distance": self.max_distance,
                "iou_threshold": self.iou_threshold,
                "disappear_frames": self.disappear_frames,
                "use_iou_matching": self.use_iou_matching
            }
        }
    
    def reset(self):
        """重置跟踪器状态"""
        self.tracked_objects.clear()
        self.next_id = 1
        self.current_frame = 0
        logger.info("跟踪器已重置")
    
    def export_trajectories(self) -> Dict[int, List[Tuple[int, int]]]:
        """
        导出所有物体的轨迹数据
        
        Returns:
            物体ID到轨迹点列表的映射
        """
        trajectories = {}
        for obj_id, obj in self.tracked_objects.items():
            trajectories[obj_id] = obj.trajectory.copy()
        
        return trajectories
