# AI-Driven In-bin Waste Monitoring

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![YOLOv11](https://img.shields.io/badge/YOLOv11-Ultralytics-00FFFF.svg)](https://github.com/ultralytics/ultralytics)

> **AI-driven waste identification and real-time monitoring system for zero-waste construction using depth-enhanced computer vision.**

This repository contains the official implementation of the Master's thesis project:

**"AI-Driven Waste Identification and Real-Time Monitoring for Zero-Waste Construction: A Depth-Enhanced Computer Vision Approach"**  
*Yue Wu, The University of Auckland, 2025*

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Datasets](#datasets)
- [Models](#models)
- [Experiments](#experiments)
- [Citation](#citation)
- [License](#license)

---

## 🌟 Overview

**Construction Waste Project** is an integrated computer vision system that monitors construction waste disposal in real-time by:
- **Detecting & Segmenting** 7 classes of waste objects using YOLOv11
- **Tracking** objects across frames with multi-object tracking (center distance + IoU + ReID)
- **Counting Events** using a Finite-State Machine (FSM) with dwell/cooldown logic
- **Estimating Volume** via teacher-student distillation (RGB-D → RGB-only)

**Key Innovation**: Depth-enhanced volume estimation that works with RGB-only cameras at inference time (distilled from RGB-D teacher).

---

## ✨ Features

### Core Capabilities

- ✅ **Instance Segmentation**: YOLOv11n-seg for 7 waste classes (mAP50: 82.1%)
- ✅ **Multi-Object Tracking**: Hybrid approach (center + IoU + ReID) with ID persistence
- ✅ **Event Detection**: FSM-based in-bin counting (F1: 59.46% on 34 GT events)
- ✅ **Volume Estimation**: ResNet-18 student model (MAE: 2.84L, R²: 0.91)
- ✅ **Real-Time Performance**: 15.2 FPS on GTX 1080 Ti (full pipeline)
- ✅ **Edge Deployment Ready**: 4.2 GB VRAM, 3.8 GB RAM

### Supported Classes

| ID | Class       | Color      | Description                    |
|----|-------------|------------|--------------------------------|
| 0  | bin         | Green      | Waste collection container     |
| 1  | plastic bag | Red        | Plastic packaging materials    |
| 2  | brick       | Brown      | Construction bricks/blocks     |
| 3  | wood        | Dark Green | Timber/wooden materials        |
| 4  | pipe        | Yellow     | Metal/plastic pipes            |
| 5  | bottle      | Blue       | Beverage bottles (plastic/glass)|
| 6  | cardboard   | Orange     | Cardboard packaging            |

---

## 🏗️ System Architecture

```
Input RGB Frames
       │
       ▼
┌──────────────────┐
│ YOLOv11 Detector │  ← Instance Segmentation
│  (mAP50: 82.1%)  │
└────────┬─────────┘
         │ Masks + Boxes
         ▼
┌──────────────────┐
│ Multi-Object     │  ← Center + IoU + ReID
│ Tracker (MOT)    │     (IDF1: varies by class)
└────────┬─────────┘
         │ Tracked Objects
         ├─────────────────┬─────────────────┐
         ▼                 ▼                 ▼
┌─────────────┐   ┌─────────────┐   ┌──────────────┐
│ FSM Counter │   │ Volume Est. │   │ Visualizer   │
│ (F1: 59.46%)│   │ (MAE: 2.84L)│   │ (Overlay)    │
└─────────────┘   └─────────────┘   └──────────────┘
         │                 │                 │
         └────────┬────────┴─────────────────┘
                  ▼
         ┌──────────────────┐
         │  Output Results  │
         │ • Events (JSON)  │
         │ • Volume (JSON)  │
         │ • Video (MP4)    │
         │ • Frames (PNG)   │
         └──────────────────┘
```

**Key Components**:
1. **Detection**: YOLOv11n-seg (640×640 input, conf=0.25, IoU=0.7)
2. **Tracking**: Hybrid MOT (max_distance=20px, disappear_frames=15)
3. **Event FSM**: dwell=4 frames, cooldown=15 frames, IoU_entry=0.3
4. **Volume**: ResNet-18 (448×448 input, distilled from RGB-D teacher)

---

## 🚀 Installation

### Prerequisites

- **Python**: 3.10+
- **GPU**: NVIDIA with 6GB+ VRAM (GTX 1080 Ti recommended)
- **CUDA**: 11.8+ (for PyTorch 2.0)
- **RAM**: 8GB minimum, 16GB recommended

### Step 1: Clone Repository

```bash
git clone https://github.com/SylvanWu/Construction-Waste-Project.git
cd Construction-Waste-Project
```

### Step 2: Create Virtual Environment

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n mixtogether python=3.10
conda activate mixtogether
```

### Step 3: Install Dependencies

**For Command-Line Use:**

```bash
pip install -r requirements.txt
```

**For Desktop GUI (Additional):**

```bash
# Install GUI framework
pip install PyQt6>=6.5.0
```

> **Note**: The GUI application (`platform/`) requires PyQt6. For command-line usage only, you can skip this step.

### Step 4: Download Models

**Models are NOT included in this repository.** Download them separately:

#### Option A: From Zenodo (DOI)
```bash
# Download pre-trained models
wget https://doi.org/10.5281/zenodo.XXXXXX -O models.zip
unzip models.zip
```

#### Option B: Manual Download
1. **YOLOv11 Segmentation Model** (`yolov11_seg_best.pt`, ~45 MB)
   - Download from: [Zenodo Link]
   - Place in: `checkpoints/yolov11_seg_best.pt`

2. **Volume Estimation Model** (`volume_student_resnet18.pth`, ~44 MB)
   - Download from: [Zenodo Link]
   - Place in: `checkpoints/volume_student_resnet18.pth`

3. **Baseline Image** (empty bin reference)
   - Download from: [Zenodo Link]
   - Place in: `baseline/frame_000009.png`

**Expected Directory Structure**:
```
MIxTogether/
├── checkpoints/
│   ├── yolov11_seg_best.pt         # YOLOv11 model
│   └── volume_student_resnet18.pth # Volume model
├── baseline/
│   └── frame_000009.png            # Empty bin reference
└── ...
```

### Step 5: Download Datasets (Optional)

Datasets are hosted on Roboflow Universe:

```bash
# Install roboflow CLI
pip install roboflow

# Download A-Val dataset (604 frames)
python scripts/download_data.py --dataset aval

# Download A-Det dataset (846 frames, training set)
python scripts/download_data.py --dataset adet
```

Or download manually from:
- **Dataset**: https://app.roboflow.com/aidriven-waste-identification-and-realtime-monitoring-for-zerowaste-construction-a-depthenhanced-com/construction-waste-project-gdkmi/1

---

## ⚡ Quick Start

**Two Ways to Use MIxTogether:**

1. **🖥️ Desktop GUI Application** (Recommended for users) - User-friendly interface with real-time visualization
2. **⌨️ Command-Line Interface** (For researchers/developers) - Full control and scriptable

---

### Option 1: Desktop GUI Application 🖥️

**Launch the GUI:**

```bash
# Install GUI dependencies
pip install PyQt6>=6.5.0

# Run desktop application
python platform/main.py
```

**Features:**
- ✨ **Real-time camera monitoring**
- 📁 **Batch dataset processing**
- 🎛️ **Interactive parameter tuning**
- 📊 **Live result visualization**
- 📹 **Video export**

See [`platform/README.md`](platform/README.md) for detailed GUI usage.

---

### Option 2: Command-Line Interface ⌨️

### Basic Usage

```bash
# Process image sequence with default settings
python main.py --input datasets/test_sequence --output results/

# Use custom configuration
python main.py --config configs/config.yaml

# Specify models explicitly
python main.py --input ./images \
               --yolo-model checkpoints/yolov11_seg_best.pt \
               --volume-model checkpoints/volume_student_resnet18.pth \
               --baseline-image baseline/frame_000009.png
```

### Advanced Options

```bash
# Disable video generation (faster)
python main.py --input ./images --no-video

# Use CPU only (no GPU)
python main.py --input ./images --device cpu

# Verbose logging
python main.py --input ./images --verbose

# Custom output directory
python main.py --input ./images --output ./custom_results
```

### Output Files

After processing, you will find:

```
results/
├── complete_results.json      # Full system output
├── counting_statistics.json   # Event detection summary
├── volume_predictions.csv     # Frame-by-frame volumes
├── summary_report.txt         # Human-readable summary
├── visualized_frames/         # Annotated frames (PNG)
│   ├── frame_000001_vis.png
│   └── ...
├── analysis_video.mp4         # Full visualization video
└── logs/
    └── processing.log         # Detailed logs
```

---

## 📊 Datasets

### Dataset A-Val (Validation Set)

- **Frames**: 604 RGB images (1280×720, 15 FPS)
- **Duration**: ~40 seconds
- **GT Events**: 34 manually annotated in-bin events
- **Purpose**: E1 (event detection) and E2 (volume estimation) evaluation
- **Download**: [Roboflow Universe Link]

### Dataset A-Det (Training Set)

- **Frames**: 846 RGB images with instance segmentation masks
- **Classes**: 7 (bin, plastic bag, brick, wood, pipe, bottle, cardboard)
- **Format**: YOLO segmentation format (polygon masks)
- **Purpose**: Training YOLOv11 segmentation model
- **Download**: [Roboflow Universe Link]

### Data Format

**YOLO Segmentation Labels** (`labels/frame_*.txt`):
```
class_id x1 y1 x2 y2 x3 y3 ... xn yn  # Normalized polygon coordinates
```

**Ground Truth Events** (`gt_events.json`):
```json
[
  {
    "frame": 42,
    "class": "wood",
    "event_id": 1,
    "timestamp": 2.8
  },
  ...
]
```

---

## 🤖 Models

### 1. YOLOv11 Segmentation

| Parameter | Value |
|-----------|-------|
| Architecture | YOLOv11n-seg |
| Input Size | 640×640 |
| Classes | 7 |
| mAP50 | 82.1% |
| mAP50-95 | 65.3% |
| FPS (GTX 1080 Ti) | 45.2 |
| VRAM | 2.1 GB |

**Training**:
- Dataset: A-Det (846 frames)
- Epochs: 100
- Batch Size: 16
- Optimizer: SGD (lr=0.01, momentum=0.9)
- Augmentations: HSV, fliplr, mosaic, translate, scale

### 2. Volume Estimation (ResNet-18 Student)

| Parameter | Value |
|-----------|-------|
| Architecture | ResNet-18 |
| Input Size | 448×448 |
| Output | Volume (liters) |
| MAE | 2.84 L |
| MAPE | 25.39% |
| R² | 0.91 |
| FPS (GTX 1080 Ti) | 66.7 |
| VRAM | 1.8 GB |

**Training (Teacher-Student Distillation)**:
- Teacher: RGB-D depth analyzer (Method-A)
- Student: ResNet-18 (RGB-only)
- Dataset: 1933 frames from Method-A collection
- Epochs: 80 (early stopping at 15)
- Optimizer: AdamW (lr=4×10⁻⁴, wd=1×10⁻⁴)
- Loss: 0.3×L1 + 0.7×Huber(δ=5.0)
- Scheduler: CosineAnnealingWarmRestarts (T₀=10, Tₘᵤₗₜ=2)

---

## 🧪 Experiments

### E1: Event Detection

Evaluate frame-level in-bin counting:

```bash
python scripts/eval_e1.py \
    --gt-events data/gt_events.json \
    --pred-events results/complete_results.json \
    --output results/e1_metrics.json
```

**Key Metrics**:
- **Precision**: 74.51%
- **Recall**: 49.37%
- **F1 Score**: 59.46%
- **Matching Window**: ±20 frames at 15 FPS

### E2: Volume Estimation

Evaluate volume prediction accuracy:

```bash
python scripts/eval_e2.py \
    --predictions results/volume_predictions.csv \
    --ground-truth data/volume_gt.csv \
    --output results/e2_metrics.json
```

**Key Metrics**:
- **MAE**: 2.84 L
- **MAPE**: 25.39%
- **MdAE**: 2.10 L
- **RMSE**: 4.12 L
- **R²**: 0.91

### E3: Runtime Performance

Benchmark system performance:

```bash
python scripts/eval_e3.py \
    --video data/test_video.mp4 \
    --model checkpoints/yolov11_seg_best.pt \
    --output results/e3_runtime.json
```

**Key Metrics** (GTX 1080 Ti):
- **FPS**: 15.2 (full pipeline)
- **VRAM**: 4.2 GB
- **RAM**: 3.8 GB
- **Alert Latency**: 0.27 s

---

## 📖 Configuration

Edit `configs/config.yaml` to customize system behavior:

```yaml
# Model paths
models:
  yolo_model_path: "checkpoints/yolov11_seg_best.pt"
  volume_model_path: "checkpoints/volume_student_resnet18.pth"
  device: "auto"  # auto, cuda, cpu

# Detection
detection:
  confidence_threshold: 0.25
  iou_threshold: 0.7

# Tracking
tracking:
  max_distance: 20        # pixels
  disappear_frames: 15
  iou_threshold: 0.6

# Event Detection (FSM)
counting:
  dwell_frames: 4         # minimum frames inside bin
  cooldown_frames: 15     # frames to wait after event
  overlap_threshold: 0.3  # IoU for "inside bin"

# Volume Estimation
volume:
  input_size: 448
  max_volume: 100.0       # liters
  baseline_frame: "baseline/frame_000009.png"
```

---

## 📈 Results Summary

| Experiment | Metric | Value | Notes |
|------------|--------|-------|-------|
| **E1: Event Detection** | F1 Score | 59.46% | 34 GT events, ±20 frame window |
| | Precision | 74.51% | Low false positives |
| | Recall | 49.37% | Misses due to occlusion |
| **E2: Volume Estimation** | MAE | 2.84 L | RGB-only student model |
| | MAPE | 25.39% | vs. teacher's ~15% |
| | R² | 0.91 | Strong correlation |
| **E3: Runtime** | FPS | 15.2 | Full pipeline on GTX 1080 Ti |
| | VRAM | 4.2 GB | Peak usage |
| | Alert Latency | 0.27 s | From entry to event trigger |

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository at https://github.com/SylvanWu/Construction-Waste-Project
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 Citation

If you use this code or datasets in your research, please cite:

```bibtex
@mastersthesis{wu2025mixtogether,
  title={AI-Driven Waste Identification and Real-Time Monitoring for Zero-Waste Construction: A Depth-Enhanced Computer Vision Approach},
  author={Wu, Yue},
  year={2025},
  school={The University of Auckland},
  supervisor={Patrice Delmas},
  note={GitHub: https://github.com/SylvanWu/Construction-Waste-Project}
}
```

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Models**: Creative Commons Attribution 4.0 (CC BY 4.0)  
**Code**: MIT License

---

## 🙏 Acknowledgments

- **Supervisor**: Dr. Patrice Delmas (The University of Auckland)
- **Frameworks**: [Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics), [PyTorch](https://pytorch.org/)
- **Datasets**: Hosted on [Roboflow Universe](https://universe.roboflow.com/)
- **Hardware**: NVIDIA GTX 1080 Ti provided by The University of Auckland

---

## 📧 Contact

**Yue Wu**  
Master's Student, The University of Auckland, New Zealand

For questions or collaborations, please open an issue or contact via GitHub.

---

## 🔗 Links

- **Paper (Thesis)**: Master's Thesis, The University of Auckland, 2025
- **Datasets**: [Roboflow Universe](https://app.roboflow.com/aidriven-waste-identification-and-realtime-monitoring-for-zerowaste-construction-a-depthenhanced-com/construction-waste-project-gdkmi/1)
- **GitHub Repository**: https://github.com/SylvanWu/Construction-Waste-Project

---

**Last Updated**: November 2025  
**Version**: 1.0.0

