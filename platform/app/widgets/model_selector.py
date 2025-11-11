"""
模型选择器组件
"""

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QLineEdit, QFileDialog, QGroupBox)
from PyQt6.QtCore import pyqtSignal
from pathlib import Path


class ModelSelector(QWidget):
    """模型选择器组件"""
    
    # 信号：当模型路径改变时发出
    model_changed = pyqtSignal(str, str)  # (model_type, path)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
    
    def init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        
        # YOLO模型
        yolo_group = QGroupBox("YOLO Detection Model")
        yolo_layout = QHBoxLayout()
        
        self.yolo_path_edit = QLineEdit()
        self.yolo_path_edit.setPlaceholderText(" Select the YOLO model file (.pt)")
        self.yolo_path_edit.setReadOnly(True)
        
        self.yolo_browse_btn = QPushButton("Browse...")
        self.yolo_browse_btn.clicked.connect(self.browse_yolo_model)
        
        yolo_layout.addWidget(self.yolo_path_edit)
        yolo_layout.addWidget(self.yolo_browse_btn)
        yolo_group.setLayout(yolo_layout)
        
        # 体积模型
        volume_group = QGroupBox("Volume Estimation Model")
        volume_layout = QHBoxLayout()
        
        self.volume_path_edit = QLineEdit()
        self.volume_path_edit.setPlaceholderText("Select the volumetric model file (.pth)")
        self.volume_path_edit.setReadOnly(True)
        
        self.volume_browse_btn = QPushButton("Browse...")
        self.volume_browse_btn.clicked.connect(self.browse_volume_model)
        
        volume_layout.addWidget(self.volume_path_edit)
        volume_layout.addWidget(self.volume_browse_btn)
        volume_group.setLayout(volume_layout)
        
        # 基准图像
        baseline_group = QGroupBox("Empty Barrel Reference Image")
        baseline_layout = QHBoxLayout()
        
        self.baseline_path_edit = QLineEdit()
        self.baseline_path_edit.setPlaceholderText("Select the reference image file")
        self.baseline_path_edit.setReadOnly(True)
        
        self.baseline_browse_btn = QPushButton("Browse...")
        self.baseline_browse_btn.clicked.connect(self.browse_baseline_image)
        
        baseline_layout.addWidget(self.baseline_path_edit)
        baseline_layout.addWidget(self.baseline_browse_btn)
        baseline_group.setLayout(baseline_layout)
        
        # 添加到主布局
        layout.addWidget(yolo_group)
        layout.addWidget(volume_group)
        layout.addWidget(baseline_group)
    
    def browse_yolo_model(self):
        """浏览YOLO模型"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择YOLO模型文件",
            str(Path.home()),
            "模型文件 (*.pt);;所有文件 (*.*)"
        )
        
        if file_path:
            self.yolo_path_edit.setText(file_path)
            self.model_changed.emit("yolo", file_path)
    
    def browse_volume_model(self):
        """浏览体积模型"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择体积模型文件",
            str(Path.home()),
            "模型文件 (*.pth *.pt);;所有文件 (*.*)"
        )
        
        if file_path:
            self.volume_path_edit.setText(file_path)
            self.model_changed.emit("volume", file_path)
    
    def browse_baseline_image(self):
        """浏览基准图像"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择基准图像",
            str(Path.home()),
            "图像文件 (*.png *.jpg *.jpeg);;所有文件 (*.*)"
        )
        
        if file_path:
            self.baseline_path_edit.setText(file_path)
            self.model_changed.emit("baseline", file_path)
    
    def set_yolo_path(self, path: str):
        """设置YOLO模型路径"""
        self.yolo_path_edit.setText(path)
    
    def set_volume_path(self, path: str):
        """设置体积模型路径"""
        self.volume_path_edit.setText(path)
    
    def set_baseline_path(self, path: str):
        """设置基准图像路径"""
        self.baseline_path_edit.setText(path)
    
    def get_model_paths(self) -> dict:
        """获取所有模型路径"""
        return {
            'yolo': self.yolo_path_edit.text(),
            'volume': self.volume_path_edit.text(),
            'baseline': self.baseline_path_edit.text()
        }

