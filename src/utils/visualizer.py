"""
可视化模块
整合计数和体积信息的可视化
"""

import cv2
import numpy as np
from pathlib import Path
from loguru import logger
from typing import List, Dict, Any, Optional
import matplotlib.pyplot as plt

# seaborn是可选的，用于美化图表
try:
    import seaborn as sns
    sns.set_style("whitegrid")
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


class Visualizer:
    """可视化器类"""
    
    def __init__(self, config):
        """
        初始化可视化器
        
        Args:
            config: Config实例
        """
        self.config = config
        self.output_dir = config.output_dir
        self.visualized_frames_dir = config.output_dir / "visualized_frames"
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.visualized_frames_dir.mkdir(parents=True, exist_ok=True)
    
    def draw_frame(
        self,
        image: np.ndarray,
        tracked_objects: List,
        roi_manager,
        counter,
        volume_info: Dict[str, float],
        frame_id: str
    ) -> np.ndarray:
        """
        绘制单帧的完整可视化
        
        Args:
            image: 输入图像
            tracked_objects: 跟踪的物体列表
            roi_manager: ROI管理器
            counter: 计数器
            volume_info: 体积信息字典
            frame_id: 帧ID
            
        Returns:
            可视化后的图像
        """
        result = image.copy()
        
        # 1. 绘制ROI区域
        if self.config.show_roi:
            result = self._draw_roi(result, roi_manager)
        
        # 2. 绘制检测物体
        if self.config.show_detection_boxes:
            result = self._draw_objects(result, tracked_objects, roi_manager)
        
        # 3. 绘制轨迹
        if self.config.show_trajectories:
            result = self._draw_trajectories(result, tracked_objects)
        
        # 4. 绘制信息面板（左上角）
        result = self._draw_info_panel(result, counter, volume_info, frame_id)
        
        return result
    
    def _draw_roi(self, image: np.ndarray, roi_manager) -> np.ndarray:
        """绘制ROI区域"""
        if roi_manager.roi_mask is None:
            return image
        
        try:
            result = image.copy()
            
            # 创建彩色ROI覆盖层
            roi_color = (0, 255, 0)  # 绿色
            alpha = self.config.roi_alpha
            
            colored_mask = np.zeros_like(image)
            colored_mask[roi_manager.roi_mask > 0] = roi_color
            
            result = cv2.addWeighted(result, 1 - alpha, colored_mask, alpha, 0)
            
            # 绘制轮廓
            if roi_manager.roi_contour is not None:
                cv2.drawContours(result, [roi_manager.roi_contour], -1, roi_color, 2)
            
            return result
        except Exception as e:
            logger.warning(f"绘制ROI失败: {e}")
            return image
    
    def _draw_objects(self, image: np.ndarray, tracked_objects: List, roi_manager) -> np.ndarray:
        """绘制检测物体"""
        result = image.copy()
        
        for obj in tracked_objects:
            # 跳过bin本身
            if obj.class_id == self.config.bin_class_id:
                continue
            
            # 获取颜色
            color = self.config.get_class_color(obj.class_id)
            
            # 绘制边界框
            if obj.bbox is not None:
                x1, y1, x2, y2 = obj.bbox.astype(int)
                
                # 根据是否在ROI内选择线条样式
                thickness = 3 if obj.in_roi else 2
                cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)
                
                # 绘制标签
                label = f"ID{obj.id}:{obj.get_class_name()}"
                if obj.counted:
                    label += " ✓"
                
                # 标签背景
                (text_width, text_height), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                cv2.rectangle(
                    result,
                    (x1, y1 - text_height - 8),
                    (x1 + text_width + 4, y1),
                    color,
                    -1
                )
                
                # 标签文字
                cv2.putText(
                    result,
                    label,
                    (x1 + 2, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )
            
            # 绘制掩码（可选）
            if obj.mask is not None and len(obj.mask.shape) == 2:
                try:
                    mask_resized = cv2.resize(
                        obj.mask.astype(np.uint8),
                        (image.shape[1], image.shape[0])
                    )
                    colored_mask = np.zeros_like(result)
                    colored_mask[mask_resized > 0] = color
                    result = cv2.addWeighted(result, 1, colored_mask, self.config.mask_alpha, 0)
                except:
                    pass
        
        return result
    
    def _draw_trajectories(self, image: np.ndarray, tracked_objects: List) -> np.ndarray:
        """绘制物体轨迹"""
        result = image.copy()
        
        for obj in tracked_objects:
            if obj.class_id == self.config.bin_class_id:
                continue
            
            color = self.config.get_class_color(obj.class_id)
            
            # 绘制轨迹线
            if len(obj.trajectory) > 1:
                for i in range(len(obj.trajectory) - 1):
                    cv2.line(
                        result,
                        obj.trajectory[i],
                        obj.trajectory[i + 1],
                        color,
                        2
                    )
            
            # 绘制当前中心点
            if obj.center:
                cv2.circle(result, obj.center, 5, color, -1)
        
        return result
    
    def _draw_info_panel(
        self,
        image: np.ndarray,
        counter,
        volume_info: Dict[str, float],
        frame_id: str
    ) -> np.ndarray:
        """
        绘制信息面板（左上角）
        
        面板布局:
        ┌─────────────────────┐
        │ Frame: 000123       │
        │ ─────────────────── │
        │ 📦 Total: 5 objects │
        │  • brick: 2         │
        │  • wood: 3          │
        │ ─────────────────── │
        │ 📊 Volume: 23.5 L   │
        │    Fill: 35.2%      │
        │    Baseline: 5.2 L  │
        └─────────────────────┘
        """
        result = image.copy()
        
        # 面板参数
        panel_x = 10
        panel_y = 10
        line_height = 30
        padding = 15
        
        # 获取计数信息
        counts = counter.get_current_counts()
        total_count = counts['total']
        class_counts = counts['by_class']
        
        # 构建文本行 - 使用英文类别名称避免中文字符问题
        lines = []
        lines.append(f"Frame: {frame_id}")
        lines.append("-" * 20)  # 使用ASCII字符
        lines.append(f"Total: {total_count} objects")
        
        # 类别名称映射（中文->英文）
        class_name_mapping = {
            '砖块': 'brick',
            '木材': 'wood', 
            '纸板': 'cardboard',
            '塑料': 'plastic',
            '金属': 'metal',
            '玻璃': 'glass',
            '其他': 'other'
        }
        
        for class_name, count in class_counts.items():
            if count > 0:
                # 使用英文类别名称
                english_name = class_name_mapping.get(class_name, class_name)
                lines.append(f"  {english_name}: {count}")
        
        lines.append("-" * 20)  # 使用ASCII字符
        # 显示相对体积（相对于基准的增量）
        relative_volume = max(0.0, volume_info.get('current_volume', 0) - volume_info.get('baseline_volume', 0))
        lines.append(f"Volume: {relative_volume:.1f} L")
        lines.append(f"  Fill: {volume_info.get('fill_percentage', 0):.1f}%")
        lines.append(f"  Baseline: 0.0 L")  # 固定显示为0.0L
        
        # 计算面板尺寸
        max_text_width = 0
        for line in lines:
            (text_width, _), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            max_text_width = max(max_text_width, text_width)
        
        panel_width = max_text_width + 2 * padding
        panel_height = len(lines) * line_height + 2 * padding
        
        # 绘制半透明背景
        overlay = result.copy()
        cv2.rectangle(
            overlay,
            (panel_x, panel_y),
            (panel_x + panel_width, panel_y + panel_height),
            (0, 0, 0),
            -1
        )
        result = cv2.addWeighted(result, 0.3, overlay, 0.7, 0)
        
        # 绘制文本
        y_offset = panel_y + padding + 20
        for line in lines:
            if "-" in line:
                # 分隔线
                y_offset += line_height // 2
            else:
                # 普通文本
                cv2.putText(
                    result,
                    line,
                    (panel_x + padding, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )
                y_offset += line_height
        
        return result
    
    def save_frame(self, image: np.ndarray, frame_id: str) -> str:
        """保存可视化帧"""
        try:
            output_path = self.visualized_frames_dir / f"{frame_id}.png"
            cv2.imwrite(str(output_path), image)
            return str(output_path)
        except Exception as e:
            logger.error(f"保存帧失败: {e}")
            return ""
    
    def create_video(self, frame_paths: List[str], output_name: str = "analysis_video.mp4") -> str:
        """创建分析视频"""
        if not frame_paths:
            logger.warning("没有帧可以创建视频")
            return ""
        
        try:
            logger.info(f"正在创建分析视频...")
            
            # 读取第一帧获取尺寸
            first_frame = cv2.imread(frame_paths[0])
            height, width = first_frame.shape[:2]
            
            # 创建视频写入器
            output_path = self.output_dir / output_name
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(
                str(output_path),
                fourcc,
                self.config.video_fps,
                (width, height)
            )
            
            # 写入所有帧
            for frame_path in frame_paths:
                frame = cv2.imread(frame_path)
                if frame is not None:
                    video_writer.write(frame)
            
            video_writer.release()
            logger.info(f"视频创建完成: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"创建视频失败: {e}")
            return ""
    
    def plot_statistics(
        self,
        frame_ids: List[str],
        count_history: List[int],
        volume_history: List[float]
    ) -> str:
        """
        绘制统计图表
        
        Args:
            frame_ids: 帧ID列表
            count_history: 计数历史
            volume_history: 体积历史
            
        Returns:
            图表文件路径
        """
        try:
            logger.info("正在生成统计图表...")
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # 绘制计数曲线
            ax1.plot(range(len(count_history)), count_history, 'b-', linewidth=2)
            ax1.set_xlabel('Frame Index')
            ax1.set_ylabel('Cumulative Count')
            ax1.set_title('Object Counting Over Time')
            ax1.grid(True, alpha=0.3)
            
            # 绘制体积曲线
            ax2.plot(range(len(volume_history)), volume_history, 'r-', linewidth=2)
            ax2.set_xlabel('Frame Index')
            ax2.set_ylabel('Volume (L)')
            ax2.set_title('Volume Estimation Over Time')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            output_path = self.output_dir / "statistics_chart.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"统计图表已保存: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"生成统计图表失败: {e}")
            return ""
