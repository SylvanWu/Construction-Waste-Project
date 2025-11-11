# Construction Waste Monitor - Desktop Client

Desktop application for construction waste monitoring and volume estimation

## 📦 Features

- ✅ **Real-time Camera Processing**: Supports real-time video stream object detection and volume estimation
- ✅ **Dataset Batch Processing**: Supports batch processing of image sequences
- ✅ **Model Management**: Built-in models with support for custom model replacement
- ✅ **Result Visualization**: Real-time display of detection results, counting statistics, and volume information
- ✅ **One-Click Installation**: Standalone installer, no environment configuration required

## 🚀 Quick Start

### Method 1: Use Standalone Installer (Recommended)

1. Download `ConstructionWasteMonitor-Setup.exe`
2. Double-click to install
3. Run desktop shortcut

### Method 2: Run from Source

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run application
python main.py
```

## 💻 System Requirements

- **Operating System**: Windows 10/11 (64-bit)
- **Memory**: At least 8GB RAM
- **Graphics**: NVIDIA GPU with CUDA support (recommended), or use CPU mode
- **Storage**: At least 5GB available space

## 📖 Usage Guide

### 1. Model Settings

- **YOLO Detection Model**: For object detection and bin area segmentation
- **Volume Estimation Model**: For estimating object volume in bin
- Click "Select File" to replace with custom models

### 2. Processing Modes

#### Camera Mode
1. Select "Camera" tab
2. Select camera device
3. Click "Start Processing"
4. View detection results in real-time

#### Dataset Mode
1. Select "Dataset" tab
2. Select input image directory
3. Select output results directory
4. Click "Start Processing"
5. View processing progress and results

### 3. Parameter Adjustment

Click "Advanced Settings" to adjust:
- Detection confidence threshold
- Tracking parameters
- Counting parameters
- Volume estimation parameters

### 4. View Results

After processing completes, view:
- Visualized frame images
- Statistical charts
- Detailed reports
- Export video

## 🏗️ Project Structure

```
platform/
├── app/                       # UI application layer
│   ├── main_window.py         # Main window
│   └── widgets/               # UI components
│       ├── model_selector.py      # Model selector
│       ├── dataset_browser.py     # Dataset browser
│       ├── camera_selector.py     # Camera selector
│       ├── param_panel.py         # Parameter panel
│       └── result_viewer.py       # Result viewer
├── platform_core/             # Core business logic
│   ├── config_manager.py      # Configuration management
│   └── processor.py           # Processor
├── resources/                 # Resource files
│   ├── styles/main.qss        # Stylesheet
│   └── config/                # Configuration files
├── build/                     # Packaging configuration
│   ├── construction_waste_monitor.spec
│   └── build_windows.bat
├── logs/                      # Log files
├── main.py                    # Main entry point
├── requirements.txt           # Dependency list
├── README.md                  # Project documentation
├── ARCHITECTURE.md            # Architecture and call relationships
└── PIPELINE.md                # Core processing pipeline
```

## 📚 Documentation

- **README.md**: Project overview and quick start
- **ARCHITECTURE.md**: Detailed architecture design and call relationship analysis
- **PIPELINE.md**: Core processing pipeline and algorithm flow

## 🔧 Development Guide

### Package as standalone exe

```bash
# Windows
cd build
build_windows.bat
```

Generated installer located in `dist/` directory.

## ❓ FAQ

### 1. CUDA-related errors
If encountering CUDA errors, change device to "CPU" in settings.

### 2. Cannot open camera
Check if camera is occupied by other programs, or try changing camera index.

### 3. Model loading failed
Ensure model files exist and format is correct.

## 📝 Version Information

- **Version**: v1.0.0
- **Author**: Yue Wu
- **Date**: November 2025

---

For detailed usage, see main project [README.md](../README.md)
