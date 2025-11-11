"""
配置管理模块
"""

from pathlib import Path
from typing import Dict, Any
import yaml


class Config:
    """Mix系统配置类"""
    
    def __init__(self, config_path: str = None):
        """
        初始化配置
        
        Args:
            config_path: 配置文件路径（可选）
        """
        # 项目根目录
        self.project_root = Path(__file__).parent.parent
        
        # 默认配置
        self._set_default_config()
        
        # 如果提供了配置文件，加载它
        if config_path and Path(config_path).exists():
            self.load_from_file(config_path)
    
    def _set_default_config(self):
        """设置默认配置"""
        # 模型配置
        self.yolo_model_path = self.project_root / "checkpoints" / "last.pt"
        self.volume_model_path = self.project_root / "checkpoints" / "best_strong_training.pth"
        self.device = "auto"  # "auto", "cuda", "cpu"
        
        # 输入输出配置
        self.input_dir = self.project_root / "datasets" / "604sequence"
        self.output_dir = self.project_root / "results"
        self.baseline_image = self.project_root / "baseline" / "frame_000009.png"
        
        # 检测配置
        self.confidence_threshold = 0.5
        self.bin_class_id = 0
        
        # 跟踪配置
        self.max_distance = 50
        self.disappear_frames = 5
        self.iou_threshold = 0.3
        self.use_iou_matching = True
        
        # 计数配置
        self.overlap_threshold = 0.1
        self.use_center_point = True
        
        # 体积估测配置
        self.volume_input_size = 448  # 强化版（暂用，基础版训练中）
        self.max_volume = 100.0  # 最大容量(L)
        self.estimate_on_counting = True  # 仅在入桶事件时估测
        
        # 可视化配置
        self.show_detection_boxes = True
        self.show_roi = True
        self.show_trajectories = True
        self.show_volume_info = True
        self.mask_alpha = 0.3
        self.roi_alpha = 0.2
        
        # 输出配置
        self.save_visualized_frames = True
        self.save_video = True
        self.video_fps = 10
        
        # 类别配置
        self.class_names = [
            "bin",         # 0
            "plastic bag", # 1  
            "brick",       # 2
            "wood",        # 3
            "pipe",        # 4
            "bottle",      # 5
            "cardboard",   # 6
        ]
        
        self.class_colors = [
            (0, 255, 0),      # bin - 绿色
            (255, 0, 0),      # plastic bag - 红色
            (139, 69, 19),    # brick - 棕色
            (0, 100, 0),      # wood - 深绿色
            (255, 255, 0),    # pipe - 黄色
            (0, 0, 255),      # bottle - 蓝色
            (255, 165, 0),    # cardboard - 橙色
        ]
    
    def load_from_file(self, config_path: str):
        """从YAML文件加载配置"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        # 更新配置
        if "models" in config_dict:
            models_config = config_dict["models"]
            if "yolo_model_path" in models_config:
                self.yolo_model_path = Path(models_config["yolo_model_path"])
            if "volume_model_path" in models_config:
                self.volume_model_path = Path(models_config["volume_model_path"])
            if "device" in models_config:
                self.device = models_config["device"]
        
        if "input" in config_dict:
            input_config = config_dict["input"]
            if "image_dir" in input_config:
                self.input_dir = Path(input_config["image_dir"])
        
        if "output" in config_dict:
            output_config = config_dict["output"]
            if "output_dir" in output_config:
                self.output_dir = Path(output_config["output_dir"])
            if "save_visualized_frames" in output_config:
                self.save_visualized_frames = output_config["save_visualized_frames"]
            if "save_video" in output_config:
                self.save_video = output_config["save_video"]
            if "video_fps" in output_config:
                self.video_fps = output_config["video_fps"]
        
        if "volume" in config_dict:
            volume_config = config_dict["volume"]
            if "input_size" in volume_config:
                self.volume_input_size = volume_config["input_size"]
            if "baseline_frame" in volume_config:
                # 可以指定不同的基准帧
                pass
    
    def save_to_file(self, config_path: str):
        """保存配置到YAML文件"""
        config_dict = {
            "models": {
                "yolo_model_path": str(self.yolo_model_path),
                "volume_model_path": str(self.volume_model_path),
                "device": self.device
            },
            "input": {
                "image_dir": str(self.input_dir)
            },
            "output": {
                "output_dir": str(self.output_dir),
                "save_visualized_frames": self.save_visualized_frames,
                "save_video": self.save_video,
                "video_fps": self.video_fps
            },
            "detection": {
                "confidence_threshold": self.confidence_threshold,
                "bin_class_id": self.bin_class_id
            },
            "tracking": {
                "max_distance": self.max_distance,
                "disappear_frames": self.disappear_frames,
                "iou_threshold": self.iou_threshold
            },
            "counting": {
                "overlap_threshold": self.overlap_threshold,
                "use_center_point": self.use_center_point
            },
            "volume": {
                "input_size": self.volume_input_size,
                "max_volume": self.max_volume,
                "estimate_on_counting": self.estimate_on_counting
            },
            "visualization": {
                "show_detection_boxes": self.show_detection_boxes,
                "show_roi": self.show_roi,
                "show_trajectories": self.show_trajectories,
                "show_volume_info": self.show_volume_info
            }
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
    
    def get_class_name(self, class_id: int) -> str:
        """获取类别名称"""
        if 0 <= class_id < len(self.class_names):
            return self.class_names[class_id]
        return f"class_{class_id}"
    
    def get_class_color(self, class_id: int) -> tuple:
        """获取类别颜色"""
        if 0 <= class_id < len(self.class_colors):
            return self.class_colors[class_id]
        return (128, 128, 128)
