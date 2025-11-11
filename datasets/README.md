# Datasets

This directory should contain the image datasets for training and evaluation.

**Datasets are NOT included in the GitHub repository due to size (>1GB).**

---

## Required Datasets

| Dataset | Frames | Size | Description | Download Link |
|---------|--------|------|-------------|---------------|
| **A-Val** | 604 | ~400 MB | Validation set for event detection & volume | [Roboflow Universe](https://universe.roboflow.com/your-workspace/mixtogether-aval) |
| **A-Det** | 846 | ~700 MB | Training set for segmentation | [Roboflow Universe](https://universe.roboflow.com/your-workspace/mixtogether-adet) |

---

## Download Instructions

### Option 1: Automatic Download (Recommended)

```bash
# From project root

# Download A-Val dataset
python scripts/download_data.py --dataset aval

# Download A-Det dataset
python scripts/download_data.py --dataset adet
```

This will download datasets in YOLO format and organize them automatically.

### Option 2: Manual Download from Roboflow

1. **Sign up/Login** to Roboflow Universe:
   ```
   https://universe.roboflow.com/
   ```

2. **Navigate to datasets**:
   - A-Val: https://universe.roboflow.com/your-workspace/mixtogether-aval
   - A-Det: https://universe.roboflow.com/your-workspace/mixtogether-adet

3. **Download in YOLO format**:
   - Click "Download" button
   - Select "YOLOv11" format
   - Download ZIP file

4. **Extract to this directory**:
   ```bash
   unzip mixtogether-aval.zip -d datasets/A-Val/
   unzip mixtogether-adet.zip -d datasets/A-Det/
   ```

---

## Expected Directory Structure

After downloading, your structure should look like:

```
datasets/
├── A-Val/                    # Validation set (604 frames)
│   ├── images/
│   │   ├── frame_000001.png
│   │   ├── frame_000002.png
│   │   └── ... (604 files)
│   ├── labels/               # YOLO format segmentation labels
│   │   ├── frame_000001.txt
│   │   └── ...
│   ├── data.yaml             # Dataset configuration
│   └── README.dataset.txt    # Roboflow metadata
│
├── A-Det/                    # Training set (846 frames)
│   ├── images/
│   │   └── ... (846 files)
│   ├── labels/
│   │   └── ... (846 files)
│   └── data.yaml
│
└── README.md                 # This file
```

---

## Dataset Details

### A-Val (Validation Set)

- **Purpose**: Evaluation for E1 (event detection) and E2 (volume estimation)
- **Frames**: 604 RGB images
- **Resolution**: 1280×720
- **FPS**: 15
- **Duration**: ~40 seconds
- **Camera**: Intel RealSense D435i (top-down view)
- **Environment**: Indoor, controlled lighting
- **GT Events**: 34 manually annotated in-bin events

**Class Distribution in GT Events**:
| Class | Count | Percentage |
|-------|-------|------------|
| plastic bag | 13 | 38.2% |
| wood | 8 | 23.5% |
| brick | 7 | 20.6% |
| cardboard | 4 | 11.8% |
| bottle | 2 | 5.9% |

### A-Det (Training Set)

- **Purpose**: Training YOLOv11 segmentation model
- **Frames**: 846 RGB images
- **Resolution**: 1280×720
- **Annotations**: Instance segmentation masks (YOLO polygon format)
- **Classes**: 7 (bin, plastic bag, brick, wood, pipe, bottle, cardboard)
- **Split**: 70% train, 15% val, 15% test

---

## Data Format

### YOLO Segmentation Labels

Each `.txt` file contains one line per object:

```
class_id x1 y1 x2 y2 x3 y3 ... xn yn
```

Where:
- `class_id`: Integer (0-6)
- `x1 y1 ... xn yn`: Normalized polygon coordinates (0-1)

**Example** (`frame_000042.txt`):
```
0 0.234 0.156 0.456 0.178 0.567 0.234 ...  # bin
3 0.450 0.300 0.480 0.320 0.490 0.340 ...  # wood
```

### data.yaml

Dataset configuration file:

```yaml
names:
  - bin
  - plastic bag
  - brick
  - wood
  - pipe
  - bottle
  - cardboard

nc: 7  # number of classes
train: images/train
val: images/val
test: images/test
```

---

## Ground Truth Events (A-Val)

For E1 evaluation, ground truth events are provided separately:

**File**: `data/gt_events.json` (download from Zenodo)

**Format**:
```json
[
  {
    "frame": 42,
    "class": "wood",
    "event_id": 1,
    "timestamp": 2.8,
    "annotator": "manual"
  },
  ...
]
```

---

## Verification

After downloading, verify the datasets:

```bash
# Check directory structure
tree datasets/ -L 2

# Count images
echo "A-Val images: $(ls -1 datasets/A-Val/images/*.png | wc -l)"
echo "A-Det images: $(ls -1 datasets/A-Det/images/*.png | wc -l)"

# Verify data.yaml exists
cat datasets/A-Val/data.yaml
```

**Expected Output**:
```
A-Val images: 604
A-Det images: 846
```

---

## Usage

### Quick Start

Process A-Val validation set:

```bash
python main.py --input datasets/A-Val/images --output results/aval_test
```

### Training (A-Det)

Train YOLOv11 on A-Det:

```bash
from ultralytics import YOLO

model = YOLO("yolov11n-seg.pt")  # Start from pretrained
model.train(
    data="datasets/A-Det/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16
)
```

---

## Data Collection Details

### Camera Setup

- **Model**: Intel RealSense D435i
- **Position**: Top-down view, ~1.5m height
- **FOV**: Wide enough to cover entire bin
- **Lighting**: Consistent indoor lighting

### Annotation Protocol

1. **Instance Segmentation**: Polygon masks drawn manually
2. **Event Labels**: Annotated by frame number when object fully inside bin
3. **Quality Control**: Double-checked by supervisor

---

## License

Datasets are licensed under **CC BY 4.0**.

You are free to:
- Share and adapt the datasets
- Use for commercial purposes

With attribution to:
```
Yue Wu, The University of Auckland, 2025
```

---

## Citation

If you use these datasets, please cite:

```bibtex
@dataset{wu2025mixtogether_data,
  author = {Wu, Yue},
  title = {MIxTogether In-bin Waste Monitoring Datasets},
  year = {2025},
  publisher = {Roboflow Universe},
  url = {https://universe.roboflow.com/your-workspace/mixtogether}
}
```

---

## Troubleshooting

### Issue: Download Script Fails

```bash
# Ensure roboflow is installed
pip install roboflow

# Set API key
export ROBOFLOW_API_KEY="your_key_here"  # Linux/Mac
$env:ROBOFLOW_API_KEY="your_key_here"    # Windows PowerShell
```

### Issue: Missing Labels

If images download but labels are missing:
1. Ensure you selected "YOLOv11" format (not "YOLO")
2. Re-download from Roboflow
3. Check `labels/` directory exists

---

**Last Updated**: November 2025

