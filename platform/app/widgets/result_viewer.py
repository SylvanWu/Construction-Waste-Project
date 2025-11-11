"""
Result Viewer Component
"""

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QTextEdit, QGroupBox)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QImage
import numpy as np
import cv2


class ResultViewer(QWidget):
    """Result Viewer Component"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
    
    def init_ui(self):
        """Initialize UI"""
        layout = QVBoxLayout(self)
        
        # Image display area (placeholder - no real-time visualization)
        image_group = QGroupBox("Visualization Area")
        image_layout = QVBoxLayout()
        
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(640, 480)
        self.image_label.setStyleSheet("QLabel { background-color: #2b2b2b; color: #888; font-size: 14px; }")
        self.image_label.setText("Waiting for processing...\n\n"
                                 "Real-time visualization will be displayed here.\n"
                                 "Video and visualized frames will also be saved to output directory.")
        self.image_label.setWordWrap(True)
        
        image_layout.addWidget(self.image_label)
        image_group.setLayout(image_layout)
        
        # Statistics area (real-time updates)
        stats_group = QGroupBox("Real-time Statistics")
        stats_layout = QVBoxLayout()
        
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setMaximumHeight(180)
        self.stats_text.setStyleSheet("QTextEdit { background-color: #2b2b2b; color: #00ff00; font-family: Consolas; font-size: 13px; }")
        
        stats_layout.addWidget(self.stats_text)
        stats_group.setLayout(stats_layout)
        
        # Add to main layout
        layout.addWidget(image_group, stretch=3)
        layout.addWidget(stats_group, stretch=1)
    
    def start_processing(self):
        """Processing started - update status"""
        self.image_label.setText("Processing...\n\n"
                                 "Real-time visualization will appear here.\n"
                                 "Statistics will be updated below.\n\n"
                                 "Video will be created after processing completes.")
        self.stats_text.clear()
    
    def update_image(self, image: np.ndarray):
        """
        Update displayed image (real-time visualization)
        
        Args:
            image: Image in numpy array format (BGR)
        """
        try:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Get image dimensions
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            
            # Create QImage
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            
            # Scale to fit label while maintaining aspect ratio
            pixmap = QPixmap.fromImage(qt_image)
            scaled_pixmap = pixmap.scaled(
                self.image_label.size(), 
                Qt.AspectRatioMode.KeepAspectRatio, 
                Qt.TransformationMode.SmoothTransformation
            )
            
            # Update label
            self.image_label.setPixmap(scaled_pixmap)
            
        except Exception as e:
            print(f"Failed to update image: {e}")
    
    def update_stats(self, result: dict):
        """
        Update statistics information (real-time)
        
        Args:
            result: Processing result dictionary
        """
        try:
            stats_text = []
            stats_text.append(f"Frame ID: {result.get('frame_id', 'N/A')}")
            stats_text.append(f"Frame Index: {result.get('frame_index', 0)}")
            stats_text.append("")
            
            # Count information
            counts = result.get('current_counts', {})
            stats_text.append(f"Total Count: {counts.get('total', 0)} objects")
            
            class_counts = counts.get('class_counts', {})
            if class_counts:
                stats_text.append("By Class:")
                for class_name, count in class_counts.items():
                    if count > 0:
                        stats_text.append(f"  {class_name}: {count}")
            
            stats_text.append("")
            
            # Volume information
            volume_info = result.get('volume_info', {})
            current_vol = volume_info.get('current_volume', 0.0)
            baseline_vol = volume_info.get('baseline_volume', 0.0)
            fill_pct = volume_info.get('fill_percentage', 0.0)
            
            stats_text.append(f"Current Volume: {current_vol:.2f} L")
            stats_text.append(f"Baseline Volume: {baseline_vol:.2f} L")
            stats_text.append(f"Fill Percentage: {fill_pct:.1f}%")
            
            # New object notification
            if result.get('has_new_object', False):
                stats_text.append("")
                stats_text.append("⚠ New object detected in bin!")
            
            self.stats_text.setPlainText("\n".join(stats_text))
            
        except Exception as e:
            print(f"Failed to update statistics: {e}")
    
    def processing_complete(self):
        """Processing completed"""
        self.image_label.setText("Processing Complete!\n\n"
                                "Visualizations have been saved.\n\n"
                                "Check the output directory for:\n"
                                "• Visualized frames\n"
                                "• Analysis video\n"
                                "• Statistics chart\n"
                                "• Complete results (JSON)")
    
    def clear(self):
        """Clear display"""
        self.image_label.setText("Waiting for processing...\n\n"
                                 "Real-time visualization will be displayed here.\n"
                                 "Video and visualized frames will also be saved to output directory.")
        self.stats_text.clear()

