"""
主窗口
"""

from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QTabWidget, QPushButton, QProgressBar, QLabel,
                             QMessageBox, QSplitter, QTextEdit, QMenuBar, QMenu, QFileDialog,
                             QStackedWidget)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QAction
from pathlib import Path
import sys
import threading

# 导入自定义组件
from .widgets import (ModelSelector, DatasetBrowser, CameraSelector,
                     ParamPanel, ResultViewer)

# 导入platform核心模块
from platform_core import ConfigManager, CameraProcessor, DatasetProcessor
from loguru import logger


class ProcessingThread(QThread):
    """处理线程"""
    
    # 信号
    progress_updated = pyqtSignal(int, int, dict)  # (current, total, result)
    finished = pyqtSignal(dict)  # final_results
    error_occurred = pyqtSignal(str)  # error_message
    frame_processed = pyqtSignal(dict)  # result
    
    def __init__(self, processor, mode='dataset'):
        super().__init__()
        self.processor = processor
        self.mode = mode
        self.stop_event = threading.Event()
    
    def run(self):
        """运行处理"""
        try:
            if self.mode == 'dataset':
                # 数据集模式
                def progress_callback(current, total, result):
                    self.progress_updated.emit(current, total, result)
                    self.frame_processed.emit(result)
                
                results = self.processor.process_dataset(
                    progress_callback=progress_callback,
                    stop_event=self.stop_event
                )
                self.finished.emit(results)
            
            elif self.mode == 'camera':
                # 摄像头模式
                def frame_callback(result):
                    self.frame_processed.emit(result)
                
                self.processor.start_processing(
                    callback=frame_callback,
                    stop_event=self.stop_event
                )
                self.finished.emit({})
        
        except Exception as e:
            logger.error(f"处理线程出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.error_occurred.emit(str(e))
    
    def stop(self):
        """停止处理"""
        self.stop_event.set()
        if self.mode == 'camera' and self.processor:
            self.processor.stop_processing()


class MainWindow(QMainWindow):
    """主窗口类"""
    
    def __init__(self):
        super().__init__()
        
        # 配置管理器
        self.config_manager = ConfigManager()
        
        # 处理器
        self.processor = None
        self.processing_thread = None
        
        # 初始化UI
        self.init_ui()
        
        # 加载默认配置
        self.load_default_models()
        
        logger.info("主窗口初始化完成")
    
    def init_ui(self):
        """Initialize UI"""
        self.setWindowTitle("Construction Waste Monitor")
        self.setGeometry(100, 100, 1600, 900)
        
        # Create menu bar
        self.create_menus()
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout - Vertical to accommodate top tab bar
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Top: Tab navigation bar
        self.tab_bar = self.create_tab_bar()
        main_layout.addWidget(self.tab_bar)
        
        # Content area with splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left: Settings panel (enlarged)
        left_panel = self.create_left_panel()
        
        # Right: Result viewer
        right_panel = self.create_right_panel()
        
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        # Increase left panel proportion from 1:2 to 2:3
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        
        main_layout.addWidget(splitter)
        
        # Status bar
        self.statusBar().showMessage("Ready")
    
    def create_menus(self):
        """Create menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        load_config_action = QAction("Load Configuration", self)
        load_config_action.triggered.connect(self.load_config)
        file_menu.addAction(load_config_action)
        
        save_config_action = QAction("Save Configuration", self)
        save_config_action.triggered.connect(self.save_config)
        file_menu.addAction(save_config_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        
        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def create_tab_bar(self) -> QWidget:
        """Create horizontal tab navigation bar at top"""
        tab_bar_widget = QWidget()
        tab_bar_widget.setObjectName("TabBar")
        tab_bar_widget.setFixedHeight(50)
        tab_bar_layout = QHBoxLayout(tab_bar_widget)
        tab_bar_layout.setContentsMargins(10, 5, 10, 5)
        tab_bar_layout.setSpacing(5)
        
        # Tab buttons
        self.tab_buttons = []
        tab_names = ["Model Settings", "Dataset Processing", "Camera Processing", "Parameters"]
        
        for i, name in enumerate(tab_names):
            btn = QPushButton(name)
            btn.setCheckable(True)
            btn.setMinimumWidth(150)
            btn.setFixedHeight(40)
            btn.clicked.connect(lambda checked, idx=i: self.switch_tab(idx))
            self.tab_buttons.append(btn)
            tab_bar_layout.addWidget(btn)
        
        # Set first button as active
        self.tab_buttons[0].setChecked(True)
        
        tab_bar_layout.addStretch()
        
        # Style the tab bar
        tab_bar_widget.setStyleSheet("""
            #TabBar {
                background-color: #2b2b2b;
                border-bottom: 2px solid #4CAF50;
            }
            QPushButton {
                background-color: #3c3c3c;
                color: #ffffff;
                border: none;
                border-radius: 5px;
                padding: 8px 20px;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #4a4a4a;
            }
            QPushButton:checked {
                background-color: #4CAF50;
            }
        """)
        
        return tab_bar_widget
    
    def switch_tab(self, index: int):
        """Switch to specified tab"""
        # Update button states
        for i, btn in enumerate(self.tab_buttons):
            btn.setChecked(i == index)
        
        # Switch stacked widget
        self.content_stack.setCurrentIndex(index)
    
    def create_left_panel(self) -> QWidget:
        """Create left settings panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Stacked widget for different content pages
        self.content_stack = QStackedWidget()
        
        # Model Settings page
        model_page = QWidget()
        model_layout = QVBoxLayout(model_page)
        self.model_selector = ModelSelector()
        model_layout.addWidget(self.model_selector)
        model_layout.addStretch()
        self.content_stack.addWidget(model_page)
        
        # Dataset Processing page
        dataset_page = QWidget()
        dataset_layout = QVBoxLayout(dataset_page)
        self.dataset_browser = DatasetBrowser()
        dataset_layout.addWidget(self.dataset_browser)
        dataset_layout.addStretch()
        self.content_stack.addWidget(dataset_page)
        
        # Camera Processing page
        camera_page = QWidget()
        camera_layout = QVBoxLayout(camera_page)
        self.camera_selector = CameraSelector()
        camera_layout.addWidget(self.camera_selector)
        camera_layout.addStretch()
        self.content_stack.addWidget(camera_page)
        
        # Parameters page
        param_page = QWidget()
        param_layout = QVBoxLayout(param_page)
        self.param_panel = ParamPanel()
        param_layout.addWidget(self.param_panel)
        self.content_stack.addWidget(param_page)
        
        layout.addWidget(self.content_stack)
        
        # Control buttons
        control_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("Start Processing")
        self.start_btn.clicked.connect(self.start_processing)
        self.start_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 12px; font-size: 14px; }")
        
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_processing)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("QPushButton { background-color: #f44336; color: white; font-weight: bold; padding: 12px; font-size: 14px; }")
        
        control_layout.addWidget(self.start_btn)
        control_layout.addWidget(self.stop_btn)
        
        layout.addLayout(control_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMinimumHeight(25)
        layout.addWidget(self.progress_bar)
        
        # Log area
        log_label = QLabel("Processing Log:")
        log_label.setStyleSheet("font-weight: bold; font-size: 13px;")
        layout.addWidget(log_label)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(180)
        self.log_text.setStyleSheet("QTextEdit { background-color: #1e1e1e; color: #d4d4d4; font-family: Consolas; font-size: 12px; }")
        layout.addWidget(self.log_text)
        
        return panel
    
    def create_right_panel(self) -> QWidget:
        """Create right result panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Result viewer
        self.result_viewer = ResultViewer()
        layout.addWidget(self.result_viewer)
        
        return panel
    
    def load_default_models(self):
        """Load default model paths"""
        mix_dir = Path(__file__).parent.parent.parent / "MIx"
        
        # Set default model paths
        yolo_path = mix_dir / "checkpoints" / "last.pt"
        volume_path = mix_dir / "checkpoints" / "best_strong_training.pth"
        baseline_path = mix_dir / "baseline" / "frame_000009.png"
        
        if yolo_path.exists():
            self.model_selector.set_yolo_path(str(yolo_path))
            self.log("Loaded YOLO model: " + str(yolo_path))
        
        if volume_path.exists():
            self.model_selector.set_volume_path(str(volume_path))
            self.log("Loaded volume model: " + str(volume_path))
        
        if baseline_path.exists():
            self.model_selector.set_baseline_path(str(baseline_path))
            self.log("Loaded baseline image: " + str(baseline_path))
    
    def start_processing(self):
        """Start processing"""
        # Get current tab
        current_tab = self.content_stack.currentIndex()
        
        # Get configuration
        config = self.prepare_config()
        
        if current_tab == 1:  # Dataset processing
            self.start_dataset_processing(config)
        elif current_tab == 2:  # Camera processing
            self.start_camera_processing(config)
        else:
            QMessageBox.warning(self, "Warning", "Please select 'Dataset Processing' or 'Camera Processing' tab")
    
    def prepare_config(self) -> dict:
        """Prepare configuration dictionary"""
        config = self.config_manager.to_dict()
        
        # Update model paths
        model_paths = self.model_selector.get_model_paths()
        config['models']['yolo_model_path'] = model_paths['yolo']
        config['models']['volume_model_path'] = model_paths['volume']
        config['volume']['baseline_frame'] = model_paths['baseline']
        
        # Update parameters
        params = self.param_panel.get_params()
        config['models']['device'] = params['device']
        config['detection']['confidence_threshold'] = params['confidence_threshold']
        config['tracking']['max_distance'] = params['max_distance']
        config['tracking']['disappear_frames'] = params['disappear_frames']
        config['tracking']['iou_threshold'] = params['iou_threshold']
        config['tracking']['use_iou_matching'] = params['use_iou_matching']
        config['counting']['overlap_threshold'] = params['overlap_threshold']
        config['counting']['use_center_point'] = params['use_center_point']
        config['volume']['input_size'] = params['volume_input_size']
        config['volume']['max_volume'] = params['max_volume']
        config['visualization']['show_detection_boxes'] = params['show_detection_boxes']
        config['visualization']['show_roi'] = params['show_roi']
        config['visualization']['show_trajectories'] = params['show_trajectories']
        
        # Update camera settings
        camera_settings = self.camera_selector.get_settings()
        config['camera'].update(camera_settings)
        
        return config
    
    def start_dataset_processing(self, config: dict):
        """Start dataset processing"""
        paths = self.dataset_browser.get_paths()
        
        if not paths['input']:
            QMessageBox.warning(self, "Warning", "Please select input directory")
            return
        
        if not paths['output']:
            QMessageBox.warning(self, "Warning", "Please select output directory")
            return
        
        self.log("Starting dataset processing...")
        self.log(f"Input directory: {paths['input']}")
        self.log(f"Output directory: {paths['output']}")
        
        try:
            # Create processor
            self.processor = DatasetProcessor(
                config_dict=config,
                input_dir=paths['input'],
                output_dir=paths['output']
            )
            
            # Load images
            if not self.processor.load_images():
                QMessageBox.critical(self, "Error", "Unable to load image files")
                return
            
            # Create processing thread
            self.processing_thread = ProcessingThread(self.processor, mode='dataset')
            self.processing_thread.progress_updated.connect(self.on_progress_updated)
            self.processing_thread.frame_processed.connect(self.on_frame_processed)
            self.processing_thread.finished.connect(self.on_processing_finished)
            self.processing_thread.error_occurred.connect(self.on_error)
            
            # Show progress bar
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            
            # Update button state
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            
            # Update result viewer - processing started
            self.result_viewer.start_processing()
            
            # Start thread
            self.processing_thread.start()
            
        except Exception as e:
            self.log(f"Failed to start processing: {e}")
            QMessageBox.critical(self, "Error", f"Failed to start processing: {e}")
    
    def start_camera_processing(self, config: dict):
        """Start camera processing"""
        camera_settings = self.camera_selector.get_settings()
        
        self.log("Starting camera processing...")
        self.log(f"Camera index: {camera_settings['camera_index']}")
        
        try:
            # Create processor
            self.processor = CameraProcessor(
                config_dict=config,
                camera_index=camera_settings['camera_index'],
                output_dir="camera_results"
            )
            
            # Open camera
            if not self.processor.open_camera():
                QMessageBox.critical(self, "Error", "Unable to open camera")
                return
            
            # Create processing thread
            self.processing_thread = ProcessingThread(self.processor, mode='camera')
            self.processing_thread.frame_processed.connect(self.on_frame_processed)
            self.processing_thread.finished.connect(self.on_processing_finished)
            self.processing_thread.error_occurred.connect(self.on_error)
            
            # Update button state
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            
            # Start thread
            self.processing_thread.start()
            
        except Exception as e:
            self.log(f"Failed to start camera: {e}")
            QMessageBox.critical(self, "Error", f"Failed to start camera: {e}")
    
    def stop_processing(self):
        """Stop processing"""
        if self.processing_thread:
            self.log("Stopping processing...")
            self.processing_thread.stop()
            self.processing_thread.wait()
            
            if self.processor and hasattr(self.processor, 'close_camera'):
                self.processor.close_camera()
            
            self.log("Processing stopped")
    
    def on_progress_updated(self, current: int, total: int, result: dict):
        """Progress update"""
        progress = int((current / total) * 100)
        self.progress_bar.setValue(progress)
        self.statusBar().showMessage(f"Processing: {current}/{total} ({progress}%)")
    
    def on_frame_processed(self, result: dict):
        """Frame processed"""
        # Update result viewer (both image and statistics)
        if result.get('visualized_frame') is not None:
            self.result_viewer.update_image(result['visualized_frame'])
        self.result_viewer.update_stats(result)
    
    def on_processing_finished(self, results: dict):
        """Processing finished"""
        self.log("Processing complete!")
        
        # Hide progress bar
        self.progress_bar.setVisible(False)
        
        # Update button state
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        
        # Show statistics
        if results and 'statistics' in results:
            stats = results['statistics']
            self.log(f"Total frames: {stats.get('total_frames', 0)}")
            counting = stats.get('counting', {}).get('summary', {})
            self.log(f"Total count: {counting.get('total_objects', 0)}")
        
        self.statusBar().showMessage("Processing complete")
        
        # Update result viewer - processing complete
        self.result_viewer.processing_complete()
        
        QMessageBox.information(self, "Complete", "Processing complete!")
    
    def on_error(self, error_msg: str):
        """Error handling"""
        self.log(f"Error: {error_msg}")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "Error", error_msg)
    
    def log(self, message: str):
        """Add log message"""
        self.log_text.append(message)
        logger.info(message)
    
    def load_config(self):
        """Load configuration file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Configuration",
            str(Path.home()),
            "Config Files (*.yaml *.yml);;All Files (*.*)"
        )
        
        if file_path:
            if self.config_manager.load_config(file_path):
                self.log(f"Configuration loaded: {file_path}")
                QMessageBox.information(self, "Success", "Configuration loaded successfully")
    
    def save_config(self):
        """Save configuration file"""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Configuration",
            str(Path.home() / "construction_waste_monitor_config.yaml"),
            "Config Files (*.yaml *.yml);;All Files (*.*)"
        )
        
        if file_path:
            # Update configuration
            config = self.prepare_config()
            self.config_manager.config = config
            
            if self.config_manager.save_config(file_path):
                self.log(f"Configuration saved: {file_path}")
                QMessageBox.information(self, "Success", "Configuration saved successfully")
    
    def show_about(self):
        """Show about dialog"""
        about_text = """
        <h2>Construction Waste Monitor</h2>
        <p><b>Version:</b> 1.0.0</p>
        <p><b>Author:</b> Sylvan</p>
        <p><b>Date:</b> October 2025</p>
        <br>
        <p>Desktop application for construction waste monitoring and volume estimation</p>
        <p>Integrating YOLO object detection and deep learning volume estimation</p>
        <br>
        <p>Tech Stack: PyQt6, PyTorch, YOLO11, OpenCV</p>
        """
        
        QMessageBox.about(self, "About", about_text)
    
    def closeEvent(self, event):
        """Close event"""
        # Stop processing if running
        if self.processing_thread and self.processing_thread.isRunning():
            reply = QMessageBox.question(
                self,
                "Confirm Exit",
                "Processing is in progress. Are you sure you want to exit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                self.stop_processing()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()

