"""
计数统计模块
负责统计投入垃圾桶的物体数量，避免重复计数
"""

import json
from loguru import logger
from typing import Dict, List, Set, Any
from datetime import datetime
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.detection_model import DetectedObject


class ObjectCounter:
    """物体计数器类"""
    
    def __init__(self, class_names=None, output_dir=None, bin_class_id=0):
        """初始化计数器"""
        # 总计数
        self.total_count = 0
        
        # 类别名称
        self.class_names = class_names or [
            "bin", "plastic bag", "brick", "wood", "pipe", "bottle", "cardboard"
        ]
        
        # 输出目录
        self.output_dir = Path(output_dir) if output_dir else Path("results")
        
        # 配置参数
        self.bin_class_id = bin_class_id
        self.cooldown_frames = 30
        self.default_min_frames = 3
        self.class_specific_min_frames = {
            "bottle": 1,
            "pipe": 1,
            "plastic bag": 1,
            "brick": 3,
            "wood": 3,
            "cardboard": 3,
        }
        
        # 时间窗口配置（已注释）
        # self.time_window_seconds = 1.0  # 时间窗口：1秒
        # self.max_objects_per_window = 3  # 时间窗口内最大新增物体数
        # self.recent_counts = []  # 记录最近的新增计数时间
        
        # 按类别计数
        self.class_counts = {}
        for class_name in self.class_names:
            self.class_counts[class_name] = 0
        
        # 已计数的物体ID集合
        self.counted_objects: Set[int] = set()
        
        # 计数历史记录
        self.counting_history: List[Dict[str, Any]] = []
        
        # 统计信息
        self.start_time = datetime.now()
        self.last_count_time = None
        
        logger.info("物体计数器初始化完成")
    
    def update_counts(self, objects: List[DetectedObject], roi_manager, current_frame: int = 0) -> Dict[str, int]:
        """
        更新计数统计
        
        Args:
            objects: 当前帧的所有跟踪物体
            roi_manager: ROI管理器实例
            current_frame: 当前帧编号
            
        Returns:
            当前计数统计
        """
        new_counts = 0
        cooldown_frames = self.cooldown_frames
        default_min_frames = self.default_min_frames
        
        for obj in objects:
            # 跳过桶本身
            if obj.class_id == self.bin_class_id:
                continue
            
            # 检查物体是否在ROI中
            in_roi = roi_manager.is_object_in_roi(obj)
            
            # 记录ROI进入状态
            if in_roi and obj.roi_entry_frame == -1:
                obj.roi_entry_frame = current_frame
                obj.in_roi = True
            elif not in_roi:
                obj.roi_entry_frame = -1
                obj.in_roi = False
            
            # 获取该类别特定的最小帧数要求
            class_name = obj.get_class_name()
            min_roi_frames = self.class_specific_min_frames.get(
                class_name, default_min_frames
            )
            
            # 计数条件检查
            should_count = (
                obj.id not in self.counted_objects and  # 未被计数
                not obj.counted and                     # 对象状态未计数
                in_roi and                             # 当前在ROI中
                obj.roi_entry_frame != -1 and         # 有ROI进入记录
                (current_frame - obj.roi_entry_frame) >= min_roi_frames  # 在ROI中足够时间
            )
            
            # 冷却期检查：如果之前计数过，检查是否已过冷却期
            if should_count and obj.counted_frame != -1:
                if (current_frame - obj.counted_frame) < cooldown_frames:
                    should_count = False
            
            if should_count:
                # 检查时间窗口限制（已注释）
                # if not self._check_time_window_limit():
                #     logger.debug(f"物体 {obj.id} ({obj.get_class_name()}) 被时间窗口限制阻止计数")
                #     continue
                
                # 标记为已计数
                self._count_object(obj, current_frame)
                new_counts += 1
        
        # 移除新增计数日志以精简输出
        
        return self.get_current_counts()
    
    # def _check_time_window_limit(self) -> bool:
    #     """
    #     检查时间窗口限制（已注释）
    #     
    #     Returns:
    #         bool: True表示可以计数，False表示被时间窗口限制
    #     """
    #     current_time = datetime.now()
    #     
    #     # 清理超过时间窗口的记录
    #     cutoff_time = current_time.timestamp() - self.time_window_seconds
    #     self.recent_counts = [t for t in self.recent_counts if t > cutoff_time]
    #     
    #     # 检查是否超过时间窗口内的最大数量
    #     if len(self.recent_counts) >= self.max_objects_per_window:
    #         logger.debug(f"时间窗口限制：{self.time_window_seconds}秒内已有{len(self.recent_counts)}个新增，超过限制{self.max_objects_per_window}")
    #         return False
    #     
    #     return True
    
    def _count_object(self, obj: DetectedObject, current_frame: int = 0):
        """
        对单个物体进行计数
        
        Args:
            obj: 要计数的物体
        """
        # 添加到已计数集合
        self.counted_objects.add(obj.id)
        obj.counted = True
        obj.in_roi = True
        obj.counted_frame = current_frame  # 记录计数帧
        
        # 更新总计数
        self.total_count += 1
        
        # 更新类别计数
        class_name = obj.get_class_name()
        if class_name in self.class_counts:
            self.class_counts[class_name] += 1
        else:
            # 处理未知类别
            self.class_counts[class_name] = 1
        
        # 记录计数历史
        count_record = {
            "timestamp": datetime.now().isoformat(),
            "object_id": int(obj.id),
            "class_id": int(obj.class_id),
            "class_name": class_name,
            "confidence": float(obj.confidence),
            "center": [int(obj.center[0]), int(obj.center[1])] if obj.center else None,
            "bbox": [float(x) for x in obj.bbox.tolist()] if obj.bbox is not None else None
        }
        self.counting_history.append(count_record)
        
        # 更新最后计数时间
        self.last_count_time = datetime.now()
        
        # 记录到时间窗口（已注释）
        # self.recent_counts.append(self.last_count_time.timestamp())
        
        logger.info(f"物体已计数: ID={obj.id}, 类别={class_name}, "
                   f"总数={self.total_count}, {class_name}数量={self.class_counts[class_name]}")
    
    def get_current_counts(self) -> Dict[str, int]:
        """
        获取当前计数统计
        
        Returns:
            计数统计字典
        """
        return {
            "total": self.total_count,
            "by_class": self.class_counts.copy()
        }
    
    def get_detailed_statistics(self) -> Dict[str, Any]:
        """
        获取详细的统计信息
        
        Returns:
            详细统计信息
        """
        current_time = datetime.now()
        processing_duration = current_time - self.start_time
        
        # 计算各类别比例
        class_percentages = {}
        if self.total_count > 0:
            for class_name, count in self.class_counts.items():
                if count > 0:
                    class_percentages[class_name] = round((count / self.total_count) * 100, 2)
        
        # 计算计数频率
        counting_rate = 0
        if self.last_count_time and processing_duration.total_seconds() > 0:
            counting_rate = round(self.total_count / processing_duration.total_seconds() * 60, 2)  # 每分钟
        
        statistics = {
            "summary": {
                "total_objects": self.total_count,
                "unique_objects_counted": len(self.counted_objects),
                "class_distribution": self.class_counts.copy(),
                "class_percentages": class_percentages
            },
            "timing": {
                "start_time": self.start_time.isoformat(),
                "last_count_time": self.last_count_time.isoformat() if self.last_count_time else None,
                "processing_duration_seconds": round(processing_duration.total_seconds(), 2),
                "counting_rate_per_minute": counting_rate
            },
            "details": {
                "counting_history": self.counting_history.copy(),
                "counted_object_ids": list(self.counted_objects)
            }
        }
        
        return statistics
    
    def export_statistics(self, output_path: str = None) -> str:
        """
        导出统计信息到JSON文件
        
        Args:
            output_path: 输出文件路径，如果为None则使用默认路径
            
        Returns:
            输出文件路径
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir.mkdir(parents=True, exist_ok=True)
            output_path = self.output_dir / f"counting_statistics_{timestamp}.json"
        
        statistics = self.get_detailed_statistics()
        
        # 添加配置信息
        statistics["configuration"] = {
            "class_names": self.class_names,
            "bin_class_id": self.bin_class_id
        }
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(statistics, f, indent=2, ensure_ascii=False)
            
            logger.info(f"统计信息已导出到: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"导出统计信息失败: {e}")
            raise
    
    def generate_summary_report(self) -> str:
        """
        生成文本格式的摘要报告
        
        Returns:
            摘要报告字符串
        """
        stats = self.get_detailed_statistics()
        
        report_lines = [
            "=" * 50,
            "垃圾桶物体投放计数统计报告",
            "=" * 50,
            "",
            "总体统计:",
            f"  总投放物体数量: {stats['summary']['total_objects']}",
            f"  处理时间: {stats['timing']['processing_duration_seconds']} 秒",
            ""
        ]
        
        if stats['summary']['total_objects'] > 0:
            report_lines.extend([
                "各类别统计:",
                "-" * 30
            ])
            
            for class_name, count in stats['summary']['class_distribution'].items():
                if count > 0:
                    percentage = stats['summary']['class_percentages'].get(class_name, 0)
                    report_lines.append(f"  {class_name}: {count} 个 ({percentage}%)")
            
            report_lines.extend([
                "",
                "投放频率:",
                f"  平均投放频率: {stats['timing']['counting_rate_per_minute']} 个/分钟"
            ])
        else:
            report_lines.append("未检测到任何物体投放")
        
        report_lines.extend([
            "",
            "配置信息:",
            f"  计数冷却期: {self.cooldown_frames} 帧",
            "",
            "=" * 50
        ])
        
        return "\n".join(report_lines)
    
    def reset_counts(self):
        """重置所有计数"""
        self.total_count = 0
        
        # 重置类别计数
        for class_name in self.class_counts:
            self.class_counts[class_name] = 0
        
        self.counted_objects.clear()
        self.counting_history.clear()
        
        self.start_time = datetime.now()
        self.last_count_time = None
        
        logger.info("计数器已重置")
    
    def is_object_counted(self, obj_id: int) -> bool:
        """
        检查物体是否已被计数
        
        Args:
            obj_id: 物体ID
            
        Returns:
            True如果已计数，否则False
        """
        return obj_id in self.counted_objects
    
    def get_class_count(self, class_name: str) -> int:
        """
        获取指定类别的计数
        
        Args:
            class_name: 类别名称
            
        Returns:
            该类别的计数
        """
        return self.class_counts.get(class_name, 0)
    
    def get_counting_history(self) -> List[Dict[str, Any]]:
        """
        获取计数历史记录
        
        Returns:
            计数历史记录列表
        """
        return self.counting_history.copy()
    
    def undo_last_count(self) -> bool:
        """
        撤销最后一次计数（调试用）
        
        Returns:
            True如果撤销成功，否则False
        """
        if not self.counting_history:
            logger.warning("没有可撤销的计数记录")
            return False
        
        # 获取最后一次计数记录
        last_record = self.counting_history.pop()
        obj_id = last_record["object_id"]
        class_name = last_record["class_name"]
        
        # 撤销计数
        if obj_id in self.counted_objects:
            self.counted_objects.remove(obj_id)
        
        self.total_count = max(0, self.total_count - 1)
        
        if class_name in self.class_counts:
            self.class_counts[class_name] = max(0, self.class_counts[class_name] - 1)
        
        logger.info(f"已撤销物体 {obj_id} ({class_name}) 的计数")
        return True
