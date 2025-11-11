"""
参数设置面板
"""

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QDoubleSpinBox, QSpinBox, QComboBox, QGroupBox, QCheckBox)
from PyQt6.QtCore import pyqtSignal


class ParamPanel(QWidget):
    """参数设置面板"""
    
    # 信号：参数改变
    param_changed = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
    
    def init_ui(self):
        """Initialize UI"""
        layout = QVBoxLayout(self)
        
        # Basic Parameters
        basic_group = QGroupBox("Basic Parameters")
        basic_layout = QVBoxLayout()
        
        # Device Selection
        device_layout = QHBoxLayout()
        device_layout.addWidget(QLabel("Compute Device:"))
        self.device_combo = QComboBox()
        self.device_combo.addItems(["Auto", "CUDA (GPU)", "CPU"])
        self.device_combo.setToolTip("Select computing device")
        self.device_combo.currentTextChanged.connect(self.emit_params)
        device_layout.addWidget(self.device_combo)
        device_layout.addStretch()
        
        # Confidence Threshold
        conf_layout = QHBoxLayout()
        conf_layout.addWidget(QLabel("Detection Confidence:"))
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.0, 1.0)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(0.5)
        self.conf_spin.setDecimals(2)
        self.conf_spin.setToolTip("Minimum confidence for detection")
        self.conf_spin.valueChanged.connect(self.emit_params)
        conf_layout.addWidget(self.conf_spin)
        conf_layout.addStretch()
        
        basic_layout.addLayout(device_layout)
        basic_layout.addLayout(conf_layout)
        basic_group.setLayout(basic_layout)
        
        # Tracking Parameters
        tracking_group = QGroupBox("Tracking Parameters")
        tracking_layout = QVBoxLayout()
        
        # Max Distance
        dist_layout = QHBoxLayout()
        dist_layout.addWidget(QLabel("Max Distance (px):"))
        self.max_dist_spin = QSpinBox()
        self.max_dist_spin.setRange(1, 100)
        self.max_dist_spin.setValue(50)
        self.max_dist_spin.setToolTip("Maximum distance for object matching")
        self.max_dist_spin.valueChanged.connect(self.emit_params)
        dist_layout.addWidget(self.max_dist_spin)
        dist_layout.addStretch()
        
        # Disappear Frames
        disappear_layout = QHBoxLayout()
        disappear_layout.addWidget(QLabel("Disappear Frames:"))
        self.disappear_spin = QSpinBox()
        self.disappear_spin.setRange(1, 100)
        self.disappear_spin.setValue(5)
        self.disappear_spin.setToolTip("Frames before object is removed from tracking")
        self.disappear_spin.valueChanged.connect(self.emit_params)
        disappear_layout.addWidget(self.disappear_spin)
        disappear_layout.addStretch()
        
        # IoU Threshold
        iou_layout = QHBoxLayout()
        iou_layout.addWidget(QLabel("IoU Threshold:"))
        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0.0, 1.0)
        self.iou_spin.setSingleStep(0.05)
        self.iou_spin.setValue(0.3)
        self.iou_spin.setDecimals(2)
        self.iou_spin.setToolTip("Intersection over Union threshold for matching")
        self.iou_spin.valueChanged.connect(self.emit_params)
        iou_layout.addWidget(self.iou_spin)
        iou_layout.addStretch()
        
        # Use IoU Matching
        use_iou_layout = QHBoxLayout()
        self.use_iou_check = QCheckBox("Use IoU Matching")
        self.use_iou_check.setChecked(True)
        self.use_iou_check.setToolTip("Enable IoU-based object matching")
        self.use_iou_check.stateChanged.connect(self.emit_params)
        use_iou_layout.addWidget(self.use_iou_check)
        use_iou_layout.addStretch()
        
        tracking_layout.addLayout(dist_layout)
        tracking_layout.addLayout(disappear_layout)
        tracking_layout.addLayout(iou_layout)
        tracking_layout.addLayout(use_iou_layout)
        tracking_group.setLayout(tracking_layout)
        
        # ROI & Counting Parameters
        roi_group = QGroupBox("ROI & Counting Parameters")
        roi_layout = QVBoxLayout()
        
        # Overlap Threshold
        overlap_layout = QHBoxLayout()
        overlap_layout.addWidget(QLabel("Overlap Threshold:"))
        self.overlap_spin = QDoubleSpinBox()
        self.overlap_spin.setRange(0.0, 1.0)
        self.overlap_spin.setSingleStep(0.05)
        self.overlap_spin.setValue(0.1)
        self.overlap_spin.setDecimals(2)
        self.overlap_spin.setToolTip("Minimum overlap ratio for counting")
        self.overlap_spin.valueChanged.connect(self.emit_params)
        overlap_layout.addWidget(self.overlap_spin)
        overlap_layout.addStretch()
        
        # Use Center Point
        center_point_layout = QHBoxLayout()
        self.use_center_check = QCheckBox("Use Center Point Verification")
        self.use_center_check.setChecked(True)
        self.use_center_check.setToolTip("Verify object center is in ROI")
        self.use_center_check.stateChanged.connect(self.emit_params)
        center_point_layout.addWidget(self.use_center_check)
        center_point_layout.addStretch()
        
        roi_layout.addLayout(overlap_layout)
        roi_layout.addLayout(center_point_layout)
        roi_group.setLayout(roi_layout)
        
        # Volume Estimation Parameters
        volume_group = QGroupBox("Volume Estimation Parameters")
        volume_layout = QVBoxLayout()
        
        # Input Size
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("Input Size:"))
        self.input_size_combo = QComboBox()
        self.input_size_combo.addItems(["224 (Fast)", "448 (Accurate)"])
        self.input_size_combo.setCurrentIndex(1)
        self.input_size_combo.setToolTip("Model input resolution")
        self.input_size_combo.currentTextChanged.connect(self.emit_params)
        size_layout.addWidget(self.input_size_combo)
        size_layout.addStretch()
        
        # Max Capacity
        capacity_layout = QHBoxLayout()
        capacity_layout.addWidget(QLabel("Max Capacity (L):"))
        self.max_volume_spin = QDoubleSpinBox()
        self.max_volume_spin.setRange(1.0, 1000.0)
        self.max_volume_spin.setValue(100.0)
        self.max_volume_spin.setToolTip("Maximum bin capacity in liters")
        self.max_volume_spin.valueChanged.connect(self.emit_params)
        capacity_layout.addWidget(self.max_volume_spin)
        capacity_layout.addStretch()
        
        volume_layout.addLayout(size_layout)
        volume_layout.addLayout(capacity_layout)
        volume_group.setLayout(volume_layout)
        
        # Visualization Options
        viz_group = QGroupBox("Visualization Options")
        viz_layout = QVBoxLayout()
        
        self.show_boxes_check = QCheckBox("Show Detection Boxes")
        self.show_boxes_check.setChecked(True)
        self.show_boxes_check.stateChanged.connect(self.emit_params)
        
        self.show_roi_check = QCheckBox("Show ROI Region")
        self.show_roi_check.setChecked(True)
        self.show_roi_check.stateChanged.connect(self.emit_params)
        
        self.show_trajectory_check = QCheckBox("Show Trajectories")
        self.show_trajectory_check.setChecked(True)
        self.show_trajectory_check.stateChanged.connect(self.emit_params)
        
        viz_layout.addWidget(self.show_boxes_check)
        viz_layout.addWidget(self.show_roi_check)
        viz_layout.addWidget(self.show_trajectory_check)
        viz_group.setLayout(viz_layout)
        
        # Add to main layout
        layout.addWidget(basic_group)
        layout.addWidget(tracking_group)
        layout.addWidget(roi_group)
        layout.addWidget(volume_group)
        layout.addWidget(viz_group)
        layout.addStretch()
    
    def emit_params(self):
        """Emit parameters changed signal"""
        params = self.get_params()
        self.param_changed.emit(params)
    
    def get_params(self) -> dict:
        """Get all parameters"""
        # Parse device
        device_text = self.device_combo.currentText()
        if "CUDA" in device_text:
            device = "cuda"
        elif "CPU" in device_text:
            device = "cpu"
        else:
            device = "auto"
        
        # Parse input size
        size_text = self.input_size_combo.currentText()
        input_size = 224 if "224" in size_text else 448
        
        return {
            'device': device,
            'confidence_threshold': self.conf_spin.value(),
            'max_distance': self.max_dist_spin.value(),
            'disappear_frames': self.disappear_spin.value(),
            'iou_threshold': self.iou_spin.value(),
            'use_iou_matching': self.use_iou_check.isChecked(),
            'overlap_threshold': self.overlap_spin.value(),
            'use_center_point': self.use_center_check.isChecked(),
            'volume_input_size': input_size,
            'max_volume': self.max_volume_spin.value(),
            'show_detection_boxes': self.show_boxes_check.isChecked(),
            'show_roi': self.show_roi_check.isChecked(),
            'show_trajectories': self.show_trajectory_check.isChecked()
        }
    
    def set_params(self, params: dict):
        """Set parameters"""
        if 'device' in params:
            device = params['device']
            if device == 'cuda':
                self.device_combo.setCurrentText("CUDA (GPU)")
            elif device == 'cpu':
                self.device_combo.setCurrentText("CPU")
            else:
                self.device_combo.setCurrentText("Auto")
        
        if 'confidence_threshold' in params:
            self.conf_spin.setValue(params['confidence_threshold'])
        
        if 'max_distance' in params:
            self.max_dist_spin.setValue(params['max_distance'])
        
        if 'disappear_frames' in params:
            self.disappear_spin.setValue(params['disappear_frames'])
        
        if 'iou_threshold' in params:
            self.iou_spin.setValue(params['iou_threshold'])
        
        if 'use_iou_matching' in params:
            self.use_iou_check.setChecked(params['use_iou_matching'])
        
        if 'overlap_threshold' in params:
            self.overlap_spin.setValue(params['overlap_threshold'])
        
        if 'use_center_point' in params:
            self.use_center_check.setChecked(params['use_center_point'])
        
        if 'volume_input_size' in params:
            size = params['volume_input_size']
            self.input_size_combo.setCurrentIndex(0 if size == 224 else 1)
        
        if 'max_volume' in params:
            self.max_volume_spin.setValue(params['max_volume'])

