"""
Camera Selector Component
"""

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QComboBox, QSpinBox, QGroupBox, QPushButton, QMessageBox)
from PyQt6.QtCore import pyqtSignal
import cv2


class CameraSelector(QWidget):
    """Camera Selector Component"""
    
    # Signals
    settings_changed = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.available_cameras = []
        self.init_ui()
        self.detect_cameras()
    
    def init_ui(self):
        """Initialize UI"""
        layout = QVBoxLayout(self)
        
        # Camera Settings
        camera_group = QGroupBox("Camera Settings")
        camera_layout = QVBoxLayout()
        
        # Camera Device List
        device_layout = QHBoxLayout()
        device_layout.addWidget(QLabel("Camera Device:"))
        self.camera_combo = QComboBox()
        self.camera_combo.setMinimumWidth(200)
        self.camera_combo.currentIndexChanged.connect(self.emit_settings)
        device_layout.addWidget(self.camera_combo)
        
        # Refresh button
        self.refresh_btn = QPushButton("🔄 Refresh")
        self.refresh_btn.setMaximumWidth(100)
        self.refresh_btn.clicked.connect(self.detect_cameras)
        self.refresh_btn.setToolTip("Refresh available cameras")
        device_layout.addWidget(self.refresh_btn)
        device_layout.addStretch()
        
        # Camera Index (backup)
        index_layout = QHBoxLayout()
        index_layout.addWidget(QLabel("Camera Index:"))
        self.camera_index_spin = QSpinBox()
        self.camera_index_spin.setRange(0, 10)
        self.camera_index_spin.setValue(0)
        self.camera_index_spin.setToolTip("Manual camera index selection")
        self.camera_index_spin.valueChanged.connect(self.emit_settings)
        index_layout.addWidget(self.camera_index_spin)
        index_layout.addStretch()
        
        # Resolution
        resolution_layout = QHBoxLayout()
        resolution_layout.addWidget(QLabel("Resolution:"))
        
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems([
            "640x480 (VGA)",
            "1280x720 (HD)",
            "1920x1080 (Full HD)",
            "2560x1440 (2K)",
            "3840x2160 (4K)"
        ])
        self.resolution_combo.setCurrentIndex(1)  # Default 720p
        self.resolution_combo.setToolTip("Camera resolution")
        self.resolution_combo.currentTextChanged.connect(self.emit_settings)
        resolution_layout.addWidget(self.resolution_combo)
        resolution_layout.addStretch()
        
        # FPS
        fps_layout = QHBoxLayout()
        fps_layout.addWidget(QLabel("Frame Rate (FPS):"))
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 120)
        self.fps_spin.setValue(30)
        self.fps_spin.setToolTip("Camera frame rate")
        self.fps_spin.valueChanged.connect(self.emit_settings)
        fps_layout.addWidget(self.fps_spin)
        fps_layout.addStretch()
        
        camera_layout.addLayout(device_layout)
        camera_layout.addLayout(index_layout)
        camera_layout.addLayout(resolution_layout)
        camera_layout.addLayout(fps_layout)
        
        # Camera Control Buttons
        control_layout = QHBoxLayout()
        
        self.test_btn = QPushButton("Test Camera")
        self.test_btn.clicked.connect(self.test_camera)
        self.test_btn.setToolTip("Test selected camera")
        control_layout.addWidget(self.test_btn)
        
        self.snapshot_btn = QPushButton("Take Snapshot")
        self.snapshot_btn.setEnabled(False)
        self.snapshot_btn.setToolTip("Take a snapshot (during processing)")
        control_layout.addWidget(self.snapshot_btn)
        
        camera_layout.addLayout(control_layout)
        
        camera_group.setLayout(camera_layout)
        
        layout.addWidget(camera_group)
    
    def detect_cameras(self):
        """Detect available cameras"""
        self.camera_combo.clear()
        self.available_cameras = []
        
        # Try to detect cameras (0-5)
        for i in range(6):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # Use DirectShow on Windows for faster detection
            if cap.isOpened():
                # Try to read a frame to verify it's working
                ret, _ = cap.read()
                if ret:
                    self.available_cameras.append(i)
                    self.camera_combo.addItem(f"Camera {i}")
                cap.release()
        
        if not self.available_cameras:
            self.camera_combo.addItem("No cameras detected")
            QMessageBox.warning(self, "No Cameras", "No cameras detected. Please check connections.")
        else:
            # Sync with spin box
            if self.available_cameras:
                self.camera_index_spin.setValue(self.available_cameras[0])
    
    def test_camera(self):
        """Test selected camera"""
        camera_index = self.get_camera_index()
        
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            QMessageBox.critical(self, "Error", f"Cannot open camera {camera_index}")
            return
        
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            QMessageBox.information(self, "Success", 
                                   f"Camera {camera_index} is working!\n"
                                   f"Resolution: {frame.shape[1]}x{frame.shape[0]}")
        else:
            QMessageBox.critical(self, "Error", f"Failed to read from camera {camera_index}")
    
    def get_camera_index(self) -> int:
        """Get current camera index"""
        # Prefer combo box selection if available
        if self.available_cameras and self.camera_combo.currentIndex() >= 0:
            idx = self.camera_combo.currentIndex()
            if idx < len(self.available_cameras):
                return self.available_cameras[idx]
        # Fallback to spin box
        return self.camera_index_spin.value()
    
    def emit_settings(self):
        """Emit settings changed signal"""
        settings = self.get_settings()
        self.settings_changed.emit(settings)
    
    def get_settings(self) -> dict:
        """Get camera settings"""
        resolution_text = self.resolution_combo.currentText()
        # Extract resolution from text like "1280x720 (HD)"
        resolution = resolution_text.split('(')[0].strip()
        width, height = map(int, resolution.split('x'))
        
        return {
            'camera_index': self.get_camera_index(),
            'width': width,
            'height': height,
            'fps': self.fps_spin.value()
        }
    
    def set_camera_index(self, index: int):
        """Set camera index"""
        self.camera_index_spin.setValue(index)

