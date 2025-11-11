# Installation Guide

This guide provides detailed instructions for setting up the MIxTogether system.

---

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Environment Setup](#environment-setup)
3. [Installing Dependencies](#installing-dependencies)
4. [Downloading Models](#downloading-models)
5. [Downloading Datasets](#downloading-datasets)
6. [Verification](#verification)
7. [Troubleshooting](#troubleshooting)

---

## System Requirements

### Hardware

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **GPU** | NVIDIA with 6GB VRAM | GTX 1080 Ti (11GB) or RTX 3060 |
| **CPU** | 4-core processor | Intel i7 or AMD Ryzen 7 |
| **RAM** | 8 GB | 16 GB |
| **Storage** | 10 GB free | 50 GB free (with datasets) |

### Software

| Software | Version |
|----------|---------|
| **OS** | Windows 10/11, Ubuntu 20.04+, macOS 11+ |
| **Python** | 3.10 or 3.11 |
| **CUDA** | 11.8+ (for GPU support) |
| **cuDNN** | 8.6+ |

---

## Environment Setup

### Option 1: Using venv (Recommended for beginners)

```bash
# 1. Clone repository
git clone https://github.com/your-username/MIxTogether.git
cd MIxTogether

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# 4. Upgrade pip
python -m pip install --upgrade pip
```

### Option 2: Using Conda (Recommended for researchers)

```bash
# 1. Clone repository
git clone https://github.com/your-username/MIxTogether.git
cd MIxTogether

# 2. Create conda environment
conda create -n mixtogether python=3.10 -y
conda activate mixtogether

# 3. Install PyTorch with CUDA support
conda install pytorch==2.0.1 torchvision==0.15.2 pytorch-cuda=11.8 -c pytorch -c nvidia
```

---

## Installing Dependencies

### Step 1: Install Core Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- PyTorch 2.0.1 (with CUDA 11.8 support)
- Ultralytics YOLOv11
- OpenCV
- NumPy, SciPy, Pandas
- Matplotlib, Seaborn
- Loguru, PyYAML

### Step 2: Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
python -c "import ultralytics; print(f'Ultralytics: {ultralytics.__version__}')"
```

**Expected Output**:
```
PyTorch: 2.0.1+cu118
CUDA Available: True
Ultralytics: 8.0.196
```

---

## Downloading Models

Models are hosted on Zenodo and are **NOT included** in the GitHub repository.

### Step 1: Download from Zenodo

Visit the Zenodo repository:
```
https://doi.org/10.5281/zenodo.XXXXXX
```

Or use `wget`:
```bash
# Download models package
wget https://zenodo.org/record/XXXXXX/files/MIxTogether-Models-v1.0.zip -O models.zip

# Extract
unzip models.zip
```

### Step 2: Place Models in Correct Locations

```bash
# Create checkpoints directory
mkdir -p checkpoints
mkdir -p baseline

# Move models
mv yolov11_seg_best.pth checkpoints/
mv volume_student_resnet18.pth checkpoints/
mv frame_000009.png baseline/
```

### Step 3: Verify Model Files

```bash
ls -lh checkpoints/
# Expected output:
# yolov11_seg_best.pth         (~45 MB)
# volume_student_resnet18.pth  (~44 MB)

ls -lh baseline/
# Expected output:
# frame_000009.png             (~500 KB)
```

---

## Downloading Datasets

Datasets are hosted on Roboflow Universe.

### Option 1: Using Roboflow CLI (Recommended)

```bash
# Install roboflow
pip install roboflow

# Download A-Val dataset (604 frames)
python scripts/download_data.py --dataset aval --output datasets/A-Val

# Download A-Det dataset (846 frames, training set)
python scripts/download_data.py --dataset adet --output datasets/A-Det
```

### Option 2: Manual Download

1. Visit Roboflow Universe:
   - **A-Val**: https://universe.roboflow.com/your-workspace/mixtogether-aval
   - **A-Det**: https://universe.roboflow.com/your-workspace/mixtogether-adet

2. Download in YOLO format

3. Extract to:
   ```
   datasets/
   ├── A-Val/
   │   ├── images/
   │   └── labels/
   └── A-Det/
       ├── images/
       └── labels/
   ```

---

## Verification

### Step 1: Check Directory Structure

```bash
tree -L 2 MIxTogether/
```

**Expected Structure**:
```
MIxTogether/
├── src/
│   ├── detection/
│   ├── tracking/
│   ├── volume/
│   ├── event/
│   └── utils/
├── configs/
│   └── config.yaml
├── checkpoints/
│   ├── yolov11_seg_best.pth
│   └── volume_student_resnet18.pth
├── baseline/
│   └── frame_000009.png
├── datasets/
│   ├── A-Val/
│   └── A-Det/
├── scripts/
├── docs/
├── main.py
├── requirements.txt
├── README.md
└── LICENSE
```

### Step 2: Run Quick Test

```bash
# Test YOLO inference
python -c "
from ultralytics import YOLO
model = YOLO('checkpoints/yolov11_seg_best.pth')
print('✓ YOLO model loaded successfully')
"

# Test volume model
python -c "
import torch
from torchvision import models
model = models.resnet18()
model.load_state_dict(torch.load('checkpoints/volume_student_resnet18.pth'))
print('✓ Volume model loaded successfully')
"
```

### Step 3: Run Demo

```bash
# Process a small test sequence
python main.py --input datasets/A-Val/images --output test_output --verbose
```

---

## Troubleshooting

### Issue 1: CUDA Out of Memory

**Symptoms**:
```
RuntimeError: CUDA out of memory
```

**Solutions**:
1. Reduce batch size in config:
   ```yaml
   detection:
     batch_size: 1  # Default is 16
   ```

2. Use CPU inference:
   ```bash
   python main.py --input ./images --device cpu
   ```

3. Close other GPU applications

---

### Issue 2: Module Not Found

**Symptoms**:
```
ModuleNotFoundError: No module named 'ultralytics'
```

**Solutions**:
1. Ensure virtual environment is activated:
   ```bash
   # Check if (venv) or (mixtogether) appears in prompt
   which python  # Should point to venv/bin/python
   ```

2. Reinstall dependencies:
   ```bash
   pip install --force-reinstall -r requirements.txt
   ```

---

### Issue 3: Model Files Not Found

**Symptoms**:
```
FileNotFoundError: [Errno 2] No such file or directory: 'checkpoints/yolov11_seg_best.pth'
```

**Solutions**:
1. Verify model files exist:
   ```bash
   ls -lh checkpoints/
   ```

2. Check file names match exactly (case-sensitive):
   - `yolov11_seg_best.pth` (not `yolov11_seg_best.pt`)
   - `volume_student_resnet18.pth`

3. Re-download from Zenodo if missing

---

### Issue 4: CUDA Not Available

**Symptoms**:
```python
torch.cuda.is_available()  # Returns False
```

**Solutions**:
1. Check NVIDIA driver:
   ```bash
   nvidia-smi
   ```

2. Reinstall PyTorch with CUDA:
   ```bash
   pip uninstall torch torchvision
   pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
   ```

3. Verify CUDA version matches:
   ```bash
   nvcc --version  # Should be 11.8+
   ```

---

### Issue 5: Permission Denied (Linux/Mac)

**Symptoms**:
```
PermissionError: [Errno 13] Permission denied
```

**Solutions**:
1. Run with correct permissions:
   ```bash
   chmod +x scripts/*.py
   ```

2. Don't use `sudo` with pip (use virtual environment instead)

---

## Next Steps

After successful installation:

1. **Read the Quick Start guide**: [README.md](../README.md#quick-start)
2. **Configure the system**: Edit `configs/config.yaml`
3. **Run experiments**: See [EXPERIMENTS.md](EXPERIMENTS.md)
4. **Explore examples**: Check `examples/` directory

---

## Getting Help

If you encounter issues not covered here:

1. Check [GitHub Issues](https://github.com/your-username/MIxTogether/issues)
2. Review the [FAQ](FAQ.md)
3. Open a new issue with:
   - Full error message
   - Python/CUDA versions (`python --version`, `nvcc --version`)
   - OS and hardware specs
   - Steps to reproduce

---

**Last Updated**: November 2025

