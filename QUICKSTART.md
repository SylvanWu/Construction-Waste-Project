# Quick Start Guide

Get started with MIxTogether in two ways.

---

## 🖥️ Option 1: Desktop GUI Application (Recommended for Users)

### 1. Clone the Repository

```bash
git clone https://github.com/SylvanWu/Construction-Waste-Project.git
cd Construction-Waste-Project
```

### 2. Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install all dependencies (including GUI)
pip install -r requirements.txt
pip install PyQt6>=6.5.0
```

### 3. Download Models

Models are included in this repository under `checkpoints/`:
- `checkpoints/yolov11_seg_best.pth` (~45 MB)
- `checkpoints/volume_student_resnet18.pth` (~44 MB)
- `baseline/frame_000009.png` (~500 KB)

**If models are missing, download from:**
```
https://doi.org/10.5281/zenodo.XXXXXX
```

### 4. Launch Desktop Application

```bash
python platform/main.py
```

### 5. Using the GUI

1. **Select Mode**:
   - Camera Mode (real-time monitoring)
   - Dataset Mode (batch processing)

2. **Configure Models**:
   - YOLO Detection Model: `checkpoints/yolov11_seg_best.pth`
   - Volume Estimation Model: `checkpoints/volume_student_resnet18.pth`
   - Baseline Image: `baseline/frame_000009.png`

3. **Start Processing**:
   - Click "Start Processing" button
   - View real-time detection results, counting statistics, and volume information

4. **View Results**:
   - Visualized frames
   - Statistical charts
   - Export reports

See [`platform/README.md`](platform/README.md) for detailed GUI usage.

---

## ⌨️ Option 2: Command-Line Interface (Recommended for Researchers)

### 1. Clone and Install

```bash
git clone https://github.com/SylvanWu/Construction-Waste-Project.git
cd Construction-Waste-Project
pip install -r requirements.txt
```

### 2. Download Datasets (Optional)

**From Roboflow:**
```bash
# Install roboflow CLI
pip install roboflow

# Download A-Val dataset (604 frames)
python scripts/download_data.py --dataset aval

# Download A-Det dataset (846 frames, training set)
python scripts/download_data.py --dataset adet
```

Or download manually from:
- **A-Val**: https://universe.roboflow.com/your-workspace/mixtogether-aval
- **A-Det**: https://universe.roboflow.com/your-workspace/mixtogether-adet

### 3. Process Data

```bash
# Basic usage
python main.py --input datasets/A-Val/images --output results/

# Use custom configuration
python main.py --config configs/config.yaml --input ./images

# Disable video generation (faster)
python main.py --input ./images --no-video

# CPU mode (no GPU)
python main.py --input ./images --device cpu --verbose
```

### 4. View Results

```
results/
├── complete_results.json      # Complete output
├── counting_statistics.json   # Event statistics
├── volume_predictions.csv     # Volume predictions
├── visualized_frames/         # Annotated frames
└── analysis_video.mp4         # Video
```

---

## 🧪 Run Evaluation Experiments

### E1: Event Detection Evaluation

```bash
python scripts/eval_e1.py \
    --gt-events data/gt_events.json \
    --pred-events results/complete_results.json \
    --output results/e1_metrics.json
```

### E2: Volume Estimation Evaluation

```bash
python scripts/eval_e2.py \
    --predictions results/volume_predictions.csv \
    --ground-truth data/volume_gt.csv \
    --output results/e2_metrics.json
```

### E3: Performance Benchmarking

```bash
python scripts/eval_e3.py \
    --video data/test_video.mp4 \
    --model checkpoints/yolov11_seg_best.pth \
    --output results/e3_runtime.json
```

---

## ❓ FAQ

### Q1: No GPU Available?

Use CPU mode (slower):
```bash
python main.py --input ./images --device cpu
# Or select "Device: CPU" in GUI
```

### Q2: Out of Memory (CUDA OOM)

Adjust in `configs/config.yaml`:
```yaml
detection:
  batch_size: 1  # Reduce batch size
```

### Q3: Model Files Not Found

Ensure models are in correct locations:
```bash
ls -lh checkpoints/
# Should see: yolov11_seg_best.pth, volume_student_resnet18.pth
```

### Q4: GUI Launch Failed

Ensure PyQt6 is installed:
```bash
pip install PyQt6>=6.5.0
```

---

## 📚 More Resources

- **Full Documentation**: [README.md](README.md)
- **Installation Guide**: [docs/INSTALLATION.md](docs/INSTALLATION.md)
- **Usage Manual**: [docs/USAGE.md](docs/USAGE.md)
- **GUI Guide**: [platform/README.md](platform/README.md)
- **Contributing**: [CONTRIBUTING.md](CONTRIBUTING.md)

---

## 💬 Get Help

- **GitHub Issues**: https://github.com/SylvanWu/Construction-Waste-Project/issues
- **Email**: Yue Wu, The University of Auckland
- **Documentation**: https://github.com/SylvanWu/Construction-Waste-Project

---

**Enjoy using MIxTogether! 🎉**
