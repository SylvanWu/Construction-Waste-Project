"""
数据集浏览器组件
"""

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QLineEdit, QFileDialog, QGroupBox)
from PyQt6.QtCore import pyqtSignal
from pathlib import Path


class DatasetBrowser(QWidget):
    """数据集浏览器组件"""
    
    # 信号：路径改变
    path_changed = pyqtSignal(str, str)  # (path_type, path)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
    
    def init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        
        # 输入目录
        input_group = QGroupBox("Input Directory")
        input_layout = QHBoxLayout()
        
        self.input_path_edit = QLineEdit()
        self.input_path_edit.setPlaceholderText("Select the folder containing the image sequence")
        self.input_path_edit.setReadOnly(True)
        
        self.input_browse_btn = QPushButton("Browse...")
        self.input_browse_btn.clicked.connect(self.browse_input_dir)
        
        input_layout.addWidget(self.input_path_edit)
        input_layout.addWidget(self.input_browse_btn)
        input_group.setLayout(input_layout)
        
        # 输出目录
        output_group = QGroupBox("Output Directory")
        output_layout = QHBoxLayout()
        
        self.output_path_edit = QLineEdit()
        self.output_path_edit.setPlaceholderText("Select the output folder for the results")
        self.output_path_edit.setReadOnly(True)
        
        self.output_browse_btn = QPushButton("Browse...")
        self.output_browse_btn.clicked.connect(self.browse_output_dir)
        
        output_layout.addWidget(self.output_path_edit)
        output_layout.addWidget(self.output_browse_btn)
        output_group.setLayout(output_layout)
        
        # 添加到主布局
        layout.addWidget(input_group)
        layout.addWidget(output_group)
    
    def browse_input_dir(self):
        """浏览输入目录"""
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "选择输入目录",
            str(Path.home())
        )
        
        if dir_path:
            self.input_path_edit.setText(dir_path)
            self.path_changed.emit("input", dir_path)
    
    def browse_output_dir(self):
        """浏览输出目录"""
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "选择输出目录",
            str(Path.home())
        )
        
        if dir_path:
            self.output_path_edit.setText(dir_path)
            self.path_changed.emit("output", dir_path)
    
    def set_input_path(self, path: str):
        """设置输入路径"""
        self.input_path_edit.setText(path)
    
    def set_output_path(self, path: str):
        """设置输出路径"""
        self.output_path_edit.setText(path)
    
    def get_paths(self) -> dict:
        """获取路径"""
        return {
            'input': self.input_path_edit.text(),
            'output': self.output_path_edit.text()
        }

