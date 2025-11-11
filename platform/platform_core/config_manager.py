"""
配置管理器
管理应用程序的所有配置参数
"""

import yaml
from pathlib import Path
from typing import Dict, Any
from loguru import logger


class ConfigManager:
    """配置管理器类"""
    
    # 默认配置
    DEFAULT_CONFIG = {
        "models": {
            "yolo_model_path": "checkpoints/last.pt",
            "volume_model_path": "checkpoints/best_strong_training.pth",
            "device": "auto"
        },
        "detection": {
            "confidence_threshold": 0.5,
            "bin_class_id": 0
        },
        "tracking": {
            "max_distance": 50,
            "disappear_frames": 5,
            "iou_threshold": 0.3,
            "use_iou_matching": True
        },
        "counting": {
            "overlap_threshold": 0.1,
            "use_center_point": True
        },
        "volume": {
            "input_size": 448,
            "max_volume": 100.0,
            "estimate_every_frame": True,
            "baseline_frame": "baseline/frame_000009.png"
        },
        "visualization": {
            "show_detection_boxes": True,
            "show_roi": True,
            "show_trajectories": True,
            "show_volume_info": True,
            "mask_alpha": 0.3,
            "roi_alpha": 0.2
        },
        "camera": {
            "camera_index": 0,
            "width": 1280,
            "height": 720,
            "fps": 30
        },
        "output": {
            "save_visualized_frames": True,
            "save_video": True,
            "video_fps": 10
        },
        "classes": {
            "names": [
                "bin", "plastic bag", "brick", "wood", 
                "pipe", "bottle", "cardboard"
            ],
            "colors": [
                [0, 255, 0],    # bin - 绿色
                [255, 0, 0],    # plastic bag - 红色
                [139, 69, 19],  # brick - 棕色
                [0, 100, 0],    # wood - 深绿色
                [255, 255, 0],  # pipe - 黄色
                [0, 0, 255],    # bottle - 蓝色
                [255, 165, 0]   # cardboard - 橙色
            ]
        }
    }
    
    def __init__(self, config_file: str = None):
        """
        初始化配置管理器
        
        Args:
            config_file: 配置文件路径，如果为None则使用默认配置
        """
        self.config_file = Path(config_file) if config_file else None
        self.config = self.DEFAULT_CONFIG.copy()
        
        # 如果提供了配置文件，加载它
        if self.config_file and self.config_file.exists():
            self.load_config(self.config_file)
    
    def load_config(self, config_file: str) -> bool:
        """
        从YAML文件加载配置
        
        Args:
            config_file: 配置文件路径
            
        Returns:
            是否加载成功
        """
        try:
            config_path = Path(config_file)
            if not config_path.exists():
                logger.warning(f"配置文件不存在: {config_file}")
                return False
            
            with open(config_path, 'r', encoding='utf-8') as f:
                loaded_config = yaml.safe_load(f)
            
            # 合并配置（保留默认值）
            self._merge_config(loaded_config)
            
            self.config_file = config_path
            logger.info(f"成功加载配置文件: {config_file}")
            return True
            
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            return False
    
    def save_config(self, config_file: str = None) -> bool:
        """
        保存配置到YAML文件
        
        Args:
            config_file: 配置文件路径，如果为None则使用当前配置文件
            
        Returns:
            是否保存成功
        """
        try:
            save_path = Path(config_file) if config_file else self.config_file
            if not save_path:
                logger.error("未指定配置文件路径")
                return False
            
            # 确保目录存在
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(save_path, 'w', encoding='utf-8') as f:
                yaml.safe_dump(self.config, f, allow_unicode=True, default_flow_style=False)
            
            logger.info(f"配置已保存到: {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"保存配置文件失败: {e}")
            return False
    
    def _merge_config(self, loaded_config: Dict[str, Any]):
        """
        合并加载的配置到默认配置中
        
        Args:
            loaded_config: 加载的配置字典
        """
        for key, value in loaded_config.items():
            if key in self.config and isinstance(value, dict):
                self.config[key].update(value)
            else:
                self.config[key] = value
    
    def get(self, key: str, default=None) -> Any:
        """
        获取配置项（支持点号分隔的嵌套键）
        
        Args:
            key: 配置键，支持 "models.device" 这样的嵌套键
            default: 默认值
            
        Returns:
            配置值
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """
        设置配置项（支持点号分隔的嵌套键）
        
        Args:
            key: 配置键
            value: 配置值
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def reset_to_default(self):
        """重置为默认配置"""
        self.config = self.DEFAULT_CONFIG.copy()
        logger.info("配置已重置为默认值")
    
    def get_model_paths(self) -> Dict[str, str]:
        """获取模型路径配置"""
        return {
            'yolo': self.get('models.yolo_model_path'),
            'volume': self.get('models.volume_model_path'),
            'baseline': self.get('volume.baseline_frame')
        }
    
    def get_camera_config(self) -> Dict[str, Any]:
        """获取摄像头配置"""
        return self.config.get('camera', {})
    
    def get_processing_params(self) -> Dict[str, Any]:
        """获取处理参数"""
        return {
            'detection': self.config.get('detection', {}),
            'tracking': self.config.get('tracking', {}),
            'counting': self.config.get('counting', {}),
            'volume': self.config.get('volume', {}),
            'visualization': self.config.get('visualization', {})
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return self.config.copy()

