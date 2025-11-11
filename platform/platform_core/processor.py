"""
处理器模块
封装MIx系统的处理逻辑，支持摄像头和数据集两种模式
"""

import cv2
import numpy as np
from pathlib import Path
from loguru import logger
from typing import Dict, Any, List, Optional, Callable
import glob
import json
from datetime import datetime
import sys

# 导入MIx模块
MIX_PATH = Path(__file__).parent.parent.parent / "MIx"
sys.path.insert(0, str(MIX_PATH))

from models import DetectionModel, DetectedObject, VolumeModel
from core.object_tracker import ObjectTracker
from core.roi_manager import ROIManager
from core.counter import ObjectCounter
from core.volume_estimator import VolumeEstimator
from utils import Visualizer


class MixProcessor:
    """MIx系统处理器基类"""
    
    def __init__(self, config_dict: Dict[str, Any], output_dir: str = None):
        """
        初始化处理器
        
        Args:
            config_dict: 配置字典
            output_dir: 输出目录
        """
        self.config = config_dict
        self.output_dir = Path(output_dir) if output_dir else Path("results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 解析配置
        self._parse_config()
        
        # 初始化模型和核心模块
        self._init_models()
        self._init_core_modules()
        
        # 处理状态
        self.is_processing = False
        self.frame_count = 0
        self.results = []
        
        # 历史记录
        self.count_history = []
        self.volume_history = []
        
        logger.info("MixProcessor初始化完成")
    
    def _parse_config(self):
        """解析配置"""
        # 模型路径（相对于MIx目录）
        mix_dir = Path(__file__).parent.parent.parent / "MIx"
        self.yolo_model_path = mix_dir / self.config['models']['yolo_model_path']
        self.volume_model_path = mix_dir / self.config['models']['volume_model_path']
        self.baseline_image = mix_dir / self.config['volume']['baseline_frame']
        
        # 设备
        self.device = self.config['models']['device']
        
        # 检测参数
        self.confidence_threshold = self.config['detection']['confidence_threshold']
        self.bin_class_id = self.config['detection']['bin_class_id']
        
        # 跟踪参数
        self.max_distance = self.config['tracking']['max_distance']
        self.iou_threshold = self.config['tracking']['iou_threshold']
        self.disappear_frames = self.config['tracking']['disappear_frames']
        self.use_iou_matching = self.config['tracking']['use_iou_matching']
        
        # 计数参数
        self.overlap_threshold = self.config['counting']['overlap_threshold']
        self.use_center_point = self.config['counting']['use_center_point']
        
        # 体积参数
        self.volume_input_size = self.config['volume']['input_size']
        self.max_volume = self.config['volume']['max_volume']
        
        # 可视化参数
        self.roi_alpha = self.config['visualization']['roi_alpha']
        
        # 类别信息
        self.class_names = self.config['classes']['names']
        self.class_colors = self.config['classes']['colors']
    
    def _init_models(self):
        """初始化模型"""
        logger.info("正在初始化模型...")
        
        # YOLO检测模型
        self.detection_model = DetectionModel(
            model_path=str(self.yolo_model_path),
            device=self.device,
            confidence_threshold=self.confidence_threshold
        )
        
        # 体积估测模型
        self.volume_model = VolumeModel(
            checkpoint_path=str(self.volume_model_path),
            device=self.device,
            input_size=self.volume_input_size,
            baseline_volume=0.0
        )
        
        logger.info("模型初始化完成")
    
    def _init_core_modules(self):
        """初始化核心模块"""
        logger.info("正在初始化核心模块...")
        
        # 物体跟踪器
        self.tracker = ObjectTracker(
            max_distance=self.max_distance,
            iou_threshold=self.iou_threshold,
            disappear_frames=self.disappear_frames,
            use_iou_matching=self.use_iou_matching
        )
        
        # 调试：记录tracker配置
        logger.info(f"ObjectTracker配置: max_distance={self.max_distance}, "
                   f"iou_threshold={self.iou_threshold}, "
                   f"disappear_frames={self.disappear_frames}, "
                   f"use_iou_matching={self.use_iou_matching}")
        
        # ROI管理器
        self.roi_manager = ROIManager(
            bin_class_id=self.bin_class_id,
            overlap_threshold=self.overlap_threshold,
            use_center_point=self.use_center_point,
            roi_alpha=self.roi_alpha,
            line_thickness=2
        )
        
        # 计数器
        self.counter = ObjectCounter(
            class_names=self.class_names,
            output_dir=str(self.output_dir),
            bin_class_id=self.bin_class_id
        )
        
        # 体积估测器
        self.volume_estimator = VolumeEstimator(
            self.volume_model,
            str(self.baseline_image)
        )
        
        # 可视化器（使用简化配置）
        class SimpleConfig:
            def __init__(self, config_dict, output_dir):
                self.output_dir = Path(output_dir)
                
                # 输出配置
                self.save_visualized_frames = config_dict.get('output', {}).get('save_visualized_frames', True)
                self.save_video = config_dict.get('output', {}).get('save_video', True)
                self.video_fps = config_dict.get('output', {}).get('video_fps', 10)
                
                # 检测配置
                self.bin_class_id = config_dict.get('detection', {}).get('bin_class_id', 0)
                self.confidence_threshold = config_dict.get('detection', {}).get('confidence_threshold', 0.5)
                
                # 可视化配置
                viz_config = config_dict.get('visualization', {})
                self.show_detection_boxes = viz_config.get('show_detection_boxes', True)
                self.show_roi = viz_config.get('show_roi', True)
                self.show_trajectories = viz_config.get('show_trajectories', True)
                self.show_volume_info = viz_config.get('show_volume_info', True)
                self.mask_alpha = viz_config.get('mask_alpha', 0.3)
                self.roi_alpha = viz_config.get('roi_alpha', 0.2)
                
                # 类别配置
                self.class_names = config_dict.get('classes', {}).get('names', [])
                self.class_colors = config_dict.get('classes', {}).get('colors', [])
                
                # 体积配置
                self.max_volume = config_dict.get('volume', {}).get('max_volume', 100.0)
            
            def get_class_color(self, class_id):
                """获取类别颜色"""
                if 0 <= class_id < len(self.class_colors):
                    return tuple(self.class_colors[class_id])
                # 默认颜色
                return (255, 255, 255)
        
        simple_config = SimpleConfig(self.config, self.output_dir)
        self.visualizer = Visualizer(simple_config)
        
        logger.info("核心模块初始化完成")
    
    def reset(self):
        """重置处理器状态"""
        self.frame_count = 0
        self.results.clear()
        self.count_history.clear()
        self.volume_history.clear()
        
        self.tracker.reset()
        self.counter.reset_counts()
        self.roi_manager.reset_roi()
        self.volume_estimator.reset()
        
        # 验证重置
        logger.info(f"处理器已重置 - tracker.current_frame={self.tracker.current_frame}, "
                   f"tracker.next_id={self.tracker.next_id}, "
                   f"counted_objects={len(self.counter.counted_objects)}")
    
    def process_frame(self, frame: np.ndarray, frame_id: str = None, do_visualization: bool = False) -> Dict[str, Any]:
        """
        处理单帧图像
        
        Args:
            frame: 输入图像 (BGR格式)
            frame_id: 帧ID
            do_visualization: 是否执行可视化（默认False以提高性能）
            
        Returns:
            处理结果字典
        """
        # 在处理开始时就增加帧计数，确保与tracker同步
        self.frame_count += 1
        
        if frame_id is None:
            frame_id = f"frame_{self.frame_count:06d}"
        
        image_shape = frame.shape[:2]
        
        # 1. YOLO检测
        detection_results = self.detection_model.predict(frame)
        
        # 2. 更新ROI
        self._update_roi(detection_results, image_shape)
        
        # 3. 更新跟踪
        tracked_objects = self._update_tracking(detection_results)
        
        # 4. 更新计数 - 使用frame_count保持一致性
        previous_count = self.counter.get_current_counts()['total']
        
        # 调试日志：检查帧号同步和跟踪状态
        logger.debug(f"帧 {self.frame_count}: "
                    f"tracker.current_frame={self.tracker.current_frame}, "
                    f"检测到{len(detection_results['boxes'])}个物体, "
                    f"跟踪{len(tracked_objects)}个物体, "
                    f"已计数{len(self.counter.counted_objects)}个")
        
        current_counts = self.counter.update_counts(
            tracked_objects, self.roi_manager, self.frame_count
        )
        new_count = current_counts['total']
        
        # 如果有新计数，记录详情
        if new_count > previous_count:
            newly_counted = [obj for obj in tracked_objects if obj.counted and obj.counted_frame == self.frame_count]
            logger.info(f"🎯 帧 {self.frame_count}: 新增计数 {new_count - previous_count}, "
                       f"ID={[obj.id for obj in newly_counted]}, "
                       f"类别={[obj.get_class_name() for obj in newly_counted]}")
        
        # 5. 体积估测
        volume_info = self.volume_estimator.estimate_volume(frame, bin_mask=None)
        
        # 记录历史
        self.count_history.append(new_count)
        self.volume_history.append(volume_info.get('current_volume', 0.0))
        
        # 6. 可视化 - 只在需要时执行（性能优化）
        visualized_frame = None
        if do_visualization:
            visualized_frame = self.visualizer.draw_frame(
                frame,
                tracked_objects,
                self.roi_manager,
                self.counter,
                volume_info,
                frame_id
            )
        
        # 7. 准备结果
        result = {
            "frame_id": frame_id,
            "frame_index": self.frame_count,
            "detection_count": len(detection_results["boxes"]),
            "tracked_objects": len(tracked_objects),
            "cumulative_count": new_count,
            "current_counts": current_counts.copy(),
            "volume_info": volume_info.copy(),
            "visualized_frame": visualized_frame,
            "has_new_object": new_count > previous_count,
            # 保存原始数据用于延迟可视化
            "_raw_frame": frame if not do_visualization else None,
            "_tracked_objects": tracked_objects if not do_visualization else None,
        }
        
        self.results.append(result)
        
        return result
    
    def _update_roi(self, detection_results: Dict, image_shape: tuple):
        """更新ROI区域"""
        bin_mask = None
        for i, class_id in enumerate(detection_results["classes"]):
            if class_id == self.bin_class_id:
                mask = detection_results["masks"][i]
                if mask is not None:
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
        """更新物体跟踪（与Mix版本完全一致）"""
        # 创建DetectedObject列表（与Mix一致）
        detected_objects = []
        
        for i in range(len(detection_results["boxes"])):
            obj = DetectedObject(
                obj_id=-1,  # 临时ID
                class_id=detection_results["classes"][i],
                bbox=detection_results["boxes"][i],
                mask=detection_results["masks"][i] if i < len(detection_results["masks"]) else None,
                confidence=detection_results["confidences"][i],
                class_names=self.class_names
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
            obj.class_names = self.class_names
        
        return tracked_objects
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        counting_stats = self.counter.get_detailed_statistics()
        
        return {
            "total_frames": self.frame_count,
            "counting": counting_stats,
            "volume": {
                "baseline": self.volume_estimator.baseline_volume,
                "current": self.volume_history[-1] if self.volume_history else 0.0,
                "max": max(self.volume_history) if self.volume_history else 0.0,
                "average": np.mean(self.volume_history) if self.volume_history else 0.0
            }
        }


class CameraProcessor(MixProcessor):
    """摄像头实时处理器"""
    
    def __init__(self, config_dict: Dict[str, Any], camera_index: int = 0, output_dir: str = None):
        """
        初始化摄像头处理器
        
        Args:
            config_dict: 配置字典
            camera_index: 摄像头索引
            output_dir: 输出目录
        """
        super().__init__(config_dict, output_dir)
        
        self.camera_index = camera_index
        self.camera = None
        self.is_camera_opened = False
        
        # 摄像头配置
        camera_config = config_dict.get('camera', {})
        self.camera_width = camera_config.get('width', 1280)
        self.camera_height = camera_config.get('height', 720)
        self.camera_fps = camera_config.get('fps', 30)
        
        logger.info(f"CameraProcessor初始化完成 (摄像头索引: {camera_index})")
    
    def open_camera(self) -> bool:
        """
        打开摄像头
        
        Returns:
            是否成功打开
        """
        try:
            self.camera = cv2.VideoCapture(self.camera_index)
            
            if not self.camera.isOpened():
                logger.error(f"无法打开摄像头 {self.camera_index}")
                return False
            
            # 设置摄像头参数
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
            self.camera.set(cv2.CAP_PROP_FPS, self.camera_fps)
            
            self.is_camera_opened = True
            logger.info(f"摄像头 {self.camera_index} 已打开")
            return True
            
        except Exception as e:
            logger.error(f"打开摄像头失败: {e}")
            return False
    
    def close_camera(self):
        """关闭摄像头"""
        if self.camera:
            self.camera.release()
            self.is_camera_opened = False
            logger.info("摄像头已关闭")
    
    def read_frame(self) -> Optional[np.ndarray]:
        """
        从摄像头读取一帧
        
        Returns:
            图像帧，失败返回None
        """
        if not self.is_camera_opened:
            return None
        
        ret, frame = self.camera.read()
        if not ret:
            logger.warning("读取摄像头帧失败")
            return None
        
        return frame
    
    def start_processing(self, callback: Callable = None, stop_event=None):
        """
        开始实时处理
        
        Args:
            callback: 每帧处理完成后的回调函数 callback(result)
            stop_event: 停止事件（threading.Event）
        """
        if not self.is_camera_opened:
            if not self.open_camera():
                return
        
        self.is_processing = True
        self.reset()
        
        logger.info("开始摄像头实时处理")
        
        try:
            while self.is_processing:
                # 检查停止事件
                if stop_event and stop_event.is_set():
                    break
                
                # 读取帧
                frame = self.read_frame()
                if frame is None:
                    break
                
                # 处理帧 - 摄像头模式需要实时可视化
                result = self.process_frame(frame, do_visualization=True)
                
                # 回调
                if callback:
                    callback(result)
        
        except Exception as e:
            logger.error(f"摄像头处理出错: {e}")
            
        finally:
            self.is_processing = False
            logger.info("摄像头处理已停止")
    
    def stop_processing(self):
        """停止处理"""
        self.is_processing = False


class DatasetProcessor(MixProcessor):
    """数据集批处理器"""
    
    def __init__(self, config_dict: Dict[str, Any], input_dir: str, output_dir: str = None):
        """
        初始化数据集处理器
        
        Args:
            config_dict: 配置字典
            input_dir: 输入目录
            output_dir: 输出目录
        """
        super().__init__(config_dict, output_dir)
        
        self.input_dir = Path(input_dir)
        self.image_files = []
        
        logger.info(f"DatasetProcessor初始化完成 (输入: {input_dir})")
    
    def load_images(self) -> bool:
        """
        加载图像文件列表
        
        Returns:
            是否成功加载
        """
        if not self.input_dir.exists():
            logger.error(f"输入目录不存在: {self.input_dir}")
            return False
        
        # 支持的图像格式
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
        self.image_files = []
        
        for ext in image_extensions:
            files = glob.glob(str(self.input_dir / ext))
            files.extend(glob.glob(str(self.input_dir / ext.upper())))
            self.image_files.extend(files)
        
        # 去除重复文件（在Windows等不区分大小写的文件系统上，*.png和*.PNG会匹配相同文件）
        self.image_files = list(set(self.image_files))
        self.image_files.sort()
        
        if not self.image_files:
            logger.error(f"在 {self.input_dir} 中未找到图像文件")
            return False
        
        logger.info(f"找到 {len(self.image_files)} 个图像文件")
        return True
    
    def process_dataset(self, progress_callback: Callable = None, stop_event=None) -> Dict[str, Any]:
        """
        处理整个数据集
        
        Args:
            progress_callback: 进度回调函数 callback(current, total, result)
            stop_event: 停止事件
            
        Returns:
            最终结果
        """
        if not self.image_files:
            if not self.load_images():
                return {}
        
        self.is_processing = True
        self.reset()
        
        total = len(self.image_files)
        logger.info(f"开始处理数据集 ({total} 张图像)")
        
        try:
            # 处理所有帧（生成可视化帧但不发送到UI，与Mix版本一致）
            logger.info("=" * 60)
            logger.info("Processing frames with visualization (saving to disk)")
            logger.info("=" * 60)
            
            # 判断是否需要保存可视化
            save_visualization = self.config['output'].get('save_visualized_frames', True)
            
            for idx, image_file in enumerate(self.image_files):
                # 检查停止事件
                if stop_event and stop_event.is_set():
                    logger.info("处理被用户中断")
                    break
                
                # 读取图像
                frame = cv2.imread(image_file)
                if frame is None:
                    logger.warning(f"无法读取图像: {image_file}")
                    continue
                
                # 提取帧ID
                frame_id = Path(image_file).stem
                
                # 处理帧 - 生成可视化（与Mix一致）
                result = self.process_frame(frame, frame_id, do_visualization=True)
                
                # 保存可视化帧到磁盘
                if save_visualization and result['visualized_frame'] is not None:
                    self.visualizer.save_frame(result['visualized_frame'], frame_id)
                
                # 进度回调（发送完整结果包括可视化帧）
                if progress_callback:
                    progress_callback(idx + 1, total, result)
        
        except Exception as e:
            logger.error(f"数据集处理出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        finally:
            self.is_processing = False
        
        # 生成最终结果
        final_results = self._generate_final_results()
        logger.info("=" * 60)
        logger.info("数据集处理完成")
        logger.info("=" * 60)
        
        return final_results
    
    def _generate_final_results(self) -> Dict[str, Any]:
        """生成最终结果"""
        # 获取统计信息
        stats = self.get_statistics()
        
        # 保存统计图表
        if len(self.count_history) > 0:
            frame_ids = [r["frame_id"] for r in self.results]
            self.visualizer.plot_statistics(
                frame_ids,
                self.count_history,
                self.volume_history
            )
        
        # 生成视频
        if self.config['output'].get('save_video', True):
            saved_frames = list(self.visualizer.visualized_frames_dir.glob("*.png"))
            if saved_frames:
                saved_frames.sort()
                # MIx的create_video不接受fps参数，使用默认fps
                self.visualizer.create_video([str(f) for f in saved_frames])
        
        # 保存完整结果
        results_file = self.output_dir / "complete_results.json"
        final_results = {
            "processing_summary": {
                "total_frames": self.frame_count,
                "input_directory": str(self.input_dir),
                "output_directory": str(self.output_dir),
                "timestamp": datetime.now().isoformat()
            },
            "statistics": stats,
            "frame_details": [
                {k: v for k, v in r.items() if k != 'visualized_frame'}
                for r in self.results
            ]
        }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"最终结果已保存: {results_file}")
        
        return final_results
    
    def stop_processing(self):
        """停止处理"""
        self.is_processing = False

