"""
整合处理器
将物体检测计数和体积估测整合在一起
"""

import cv2
import numpy as np
from pathlib import Path
from loguru import logger
from typing import List, Dict, Any
from tqdm import tqdm
import glob
import json

# 导入自定义模块
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import DetectionModel, DetectedObject, VolumeModel
from core.object_tracker import ObjectTracker
from core.roi_manager import ROIManager
from core.counter import ObjectCounter
from core.volume_estimator import VolumeEstimator
from utils import Visualizer


class IntegratedProcessor:
    """整合处理器类"""
    
    def __init__(self, config):
        """
        初始化整合处理器
        
        Args:
            config: Config实例
        """
        self.config = config
        
        # 初始化检测模型
        logger.info("正在初始化YOLO检测模型...")
        self.detection_model = DetectionModel(
            model_path=str(config.yolo_model_path),
            device=config.device,
            confidence_threshold=config.confidence_threshold
        )
        
        # 初始化体积模型
        logger.info("正在初始化体积估测模型...")
        self.volume_model = VolumeModel(
            checkpoint_path=str(config.volume_model_path),
            device=config.device,
            input_size=config.volume_input_size,
            baseline_volume=0.0
        )
        
        # 初始化核心模块
        logger.info("正在初始化核心模块...")
        self.tracker = ObjectTracker(
            max_distance=config.max_distance,
            iou_threshold=config.iou_threshold,
            disappear_frames=config.disappear_frames,
            use_iou_matching=config.use_iou_matching
        )
        self.roi_manager = ROIManager(
            bin_class_id=config.bin_class_id,
            overlap_threshold=config.overlap_threshold,
            use_center_point=config.use_center_point,
            roi_alpha=config.roi_alpha,
            line_thickness=2
        )
        self.counter = ObjectCounter(
            class_names=config.class_names,
            output_dir=str(config.output_dir),
            bin_class_id=config.bin_class_id
        )
        self.volume_estimator = VolumeEstimator(
            self.volume_model,
            str(config.baseline_image)
        )
        self.visualizer = Visualizer(config)
        
        # 处理状态
        self.current_frame_idx = 0
        self.total_frames = 0
        self.frame_results = []
        self.saved_frame_paths = []
        
        # 历史记录
        self.count_history = []
        self.volume_history = []
        
        logger.info("整合处理器初始化完成")
    
    def process_sequence(self, input_dir: str = None) -> Dict[str, Any]:
        """
        处理图像序列
        
        Args:
            input_dir: 输入目录，如果为None则使用配置中的路径
            
        Returns:
            处理结果字典
        """
        if input_dir is None:
            input_dir = self.config.input_dir
        
        # 获取图像文件列表
        image_files = self._get_image_files(input_dir)
        if not image_files:
            raise ValueError(f"在目录 {input_dir} 中未找到图像文件")
        
        self.total_frames = len(image_files)
        logger.info(f"开始处理 {self.total_frames} 帧图像")
        
        # 重置状态
        self._reset_state()
        
        # 处理每一帧
        with tqdm(total=self.total_frames, desc="Processing") as pbar:
            for frame_idx, image_file in enumerate(image_files):
                try:
                    result = self._process_single_frame(image_file, frame_idx)
                    self.frame_results.append(result)
                    pbar.update(1)
                except Exception as e:
                    logger.error(f"处理帧 {frame_idx} 失败: {e}")
                    continue
        
        # 生成最终结果
        final_results = self._generate_final_results()
        
        logger.info(f"处理完成! 共处理 {self.total_frames} 帧")
        return final_results
    
    def _get_image_files(self, input_dir: str) -> List[str]:
        """获取图像文件列表"""
        input_path = Path(input_dir)
        if not input_path.exists():
            raise FileNotFoundError(f"输入目录不存在: {input_dir}")
        
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
        image_files = []
        
        for ext in image_extensions:
            files = glob.glob(str(input_path / ext))
            files.extend(glob.glob(str(input_path / ext.upper())))
            image_files.extend(files)
        
        # 去重 - Windows下大小写不敏感导致重复
        image_files = list(set(image_files))
        
        image_files.sort()
        logger.info(f"找到 {len(image_files)} 个图像文件")
        return image_files
    
    def _reset_state(self):
        """重置处理状态"""
        self.current_frame_idx = 0
        self.frame_results.clear()
        self.saved_frame_paths.clear()
        self.count_history.clear()
        self.volume_history.clear()
        
        self.tracker.reset()
        self.counter.reset_counts()
        self.roi_manager.reset_roi()
        self.volume_estimator.reset()
    
    def _process_single_frame(self, image_file: str, frame_idx: int) -> Dict[str, Any]:
        """
        处理单个帧
        
        Args:
            image_file: 图像文件路径
            frame_idx: 帧索引
            
        Returns:
            帧处理结果
        """
        self.current_frame_idx = frame_idx
        
        # 读取图像
        image = cv2.imread(image_file)
        if image is None:
            raise ValueError(f"无法读取图像: {image_file}")
        
        image_shape = image.shape[:2]
        
        # 提取帧ID
        frame_id = self._extract_frame_id(image_file)
        
        # 1. YOLO检测
        detection_results = self.detection_model.predict(image)
        
        # 2. 更新ROI区域
        self._update_roi(detection_results, image_shape)
        
        # 3. 更新物体跟踪
        tracked_objects = self._update_tracking(detection_results)
        
        # 4. 获取当前计数（返回新计数的物体列表）
        previous_count = self.counter.get_current_counts()['total']
        current_counts = self.counter.update_counts(tracked_objects, self.roi_manager, frame_idx)
        new_count = current_counts['total']
        
        # 5. 体积估测（每帧都进行估测）
        volume_info = self.volume_estimator.estimate_volume(
            image,
            bin_mask=None  # 使用完整图像，不裁剪
        )
        
        # 记录入桶事件（如果发生）
        if new_count > previous_count:
            logger.info(f"帧 {frame_idx}: 检测到入桶事件，物体总数从 {previous_count} 增加到 {new_count}")
        
        # 记录历史
        self.count_history.append(new_count)
        self.volume_history.append(volume_info.get('current_volume', 0.0))
        
        # 6. 可视化
        visualized_image = self.visualizer.draw_frame(
            image,
            tracked_objects,
            self.roi_manager,
            self.counter,
            volume_info,
            frame_id
        )
        
        # 7. 保存可视化帧
        saved_path = None
        if self.config.save_visualized_frames:
            saved_path = self.visualizer.save_frame(visualized_image, frame_id)
            self.saved_frame_paths.append(saved_path)
        
        # 8. 准备帧结果
        frame_result = {
            "frame_index": frame_idx,
            "frame_id": frame_id,
            "image_file": image_file,
            "detection_count": len(detection_results["boxes"]),
            "tracked_objects": len(tracked_objects),
            "cumulative_count": new_count,
            "current_counts": current_counts.copy(),
            "volume_info": volume_info.copy(),
            "saved_path": saved_path
        }
        
        return frame_result
    
    def _update_roi(self, detection_results: Dict, image_shape: tuple):
        """更新ROI区域"""
        # 需要调整roi_manager以适配新的接口
        # 将detection_results转换为roi_manager期望的格式
        
        # 找到bin的mask
        bin_mask = None
        for i, class_id in enumerate(detection_results["classes"]):
            if class_id == self.config.bin_class_id:
                mask = detection_results["masks"][i]
                if mask is not None:
                    # 确保mask尺寸正确
                    if len(mask.shape) == 2:
                        if mask.shape != image_shape:
                            mask = cv2.resize(
                                mask.astype(np.uint8),
                                (image_shape[1], image_shape[0])
                            )
                        bin_mask = mask
                        break
        
        if bin_mask is not None:
            self.roi_manager.roi_mask = bin_mask
            self.roi_manager._calculate_roi_properties(image_shape)
    
    def _update_tracking(self, detection_results: Dict) -> List[DetectedObject]:
        """更新物体跟踪"""
        # 创建DetectedObject列表
        detected_objects = []
        
        for i in range(len(detection_results["boxes"])):
            obj = DetectedObject(
                obj_id=-1,  # 临时ID
                class_id=detection_results["classes"][i],
                bbox=detection_results["boxes"][i],
                mask=detection_results["masks"][i],
                confidence=detection_results["confidences"][i],
                class_names=self.config.class_names
            )
            detected_objects.append(obj)
        
        # 转换为tracker期望的格式
        tracker_input = {
            "boxes": detection_results["boxes"],
            "masks": detection_results["masks"],
            "classes": detection_results["classes"],
            "confidences": detection_results["confidences"]
        }
        
        tracked_objects = self.tracker.update(tracker_input)
        
        # 更新类别名称
        for obj in tracked_objects:
            obj.class_names = self.config.class_names
        
        return tracked_objects
    
    def _extract_frame_id(self, image_file: str) -> str:
        """从文件名提取帧ID"""
        import re
        filename = Path(image_file).stem
        match = re.search(r'frame_(\d+)', filename)
        if match:
            return f"frame_{match.group(1)}"
        return filename
    
    def _generate_final_results(self) -> Dict[str, Any]:
        """生成最终结果"""
        logger.info("正在生成最终结果...")
        
        # 获取计数统计
        counting_stats = self.counter.get_detailed_statistics()
        
        # 生成摘要报告
        summary_report = self._generate_summary_report(counting_stats)
        
        # 保存摘要报告
        report_file = self.config.output_dir / "summary_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(summary_report)
        
        # 创建统计图表
        chart_file = self.visualizer.plot_statistics(
            [r["frame_id"] for r in self.frame_results],
            self.count_history,
            self.volume_history
        )
        
        # 创建视频
        video_file = None
        if self.config.save_video and self.saved_frame_paths:
            video_file = self.visualizer.create_video(self.saved_frame_paths)
        
        # 准备最终结果
        final_results = {
            "processing_summary": {
                "total_frames": self.total_frames,
                "processed_frames": len(self.frame_results)
            },
            "counting_results": counting_stats,
            "volume_summary": {
                "baseline_volume": self.volume_estimator.baseline_volume,
                "final_volume": self.volume_history[-1] if self.volume_history else 0.0,
                "max_volume": max(self.volume_history) if self.volume_history else 0.0,
                "avg_volume": np.mean(self.volume_history) if self.volume_history else 0.0
            },
            "output_files": {
                "summary_report": str(report_file),
                "statistics_chart": chart_file,
                "analysis_video": video_file,
                "visualized_frames_dir": str(self.visualizer.visualized_frames_dir)
            },
            "frame_details": self.frame_results
        }
        
        # 保存完整结果JSON
        results_file = self.config.output_dir / "complete_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"最终结果已保存到: {results_file}")
        
        # 打印摘要
        self._print_summary(final_results)
        
        return final_results
    
    def _generate_summary_report(self, counting_stats: Dict) -> str:
        """生成摘要报告文本"""
        lines = []
        lines.append("=" * 60)
        lines.append("Mix System - 处理结果报告")
        lines.append("=" * 60)
        lines.append("")
        
        lines.append("## 计数统计")
        lines.append(f"总计数: {counting_stats['summary']['total_objects']} 个物体")
        lines.append("")
        
        lines.append("### 各类别分布:")
        for class_name, count in counting_stats['summary']['class_distribution'].items():
            if count > 0:
                percentage = counting_stats['summary']['class_percentages'].get(class_name, 0)
                lines.append(f"  {class_name}: {count} 个 ({percentage:.1f}%)")
        lines.append("")
        
        lines.append("## 体积统计")
        lines.append(f"空桶基准: {self.volume_estimator.baseline_volume:.2f} L")
        if self.volume_history:
            lines.append(f"最终体积: {self.volume_history[-1]:.2f} L")
            lines.append(f"最大体积: {max(self.volume_history):.2f} L")
            lines.append(f"平均体积: {np.mean(self.volume_history):.2f} L")
        lines.append("")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def _print_summary(self, results: Dict):
        """打印处理摘要"""
        print("\n" + "=" * 60)
        print("处理完成! 结果摘要:")
        print("=" * 60)
        print(f"总帧数: {results['processing_summary']['total_frames']}")
        print(f"成功处理: {results['processing_summary']['processed_frames']}")
        print("")
        
        print("计数统计:")
        print(f"  总计数: {results['counting_results']['summary']['total_objects']}")
        for class_name, count in results['counting_results']['summary']['class_distribution'].items():
            if count > 0:
                percentage = results['counting_results']['summary']['class_percentages'].get(class_name, 0)
                print(f"    {class_name}: {count} 个 ({percentage:.1f}%)")
        print("")
        
        print("体积统计:")
        vs = results['volume_summary']
        print(f"  基准体积: {vs['baseline_volume']:.2f} L")
        print(f"  最终体积: {vs['final_volume']:.2f} L")
        print(f"  最大体积: {vs['max_volume']:.2f} L")
        print("")
        
        print("输出文件:")
        for key, path in results['output_files'].items():
            if path:
                print(f"  {key}: {path}")
        print("=" * 60)
