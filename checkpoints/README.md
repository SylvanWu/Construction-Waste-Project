# Model Checkpoints

This directory should contain the pre-trained models.

**Models are NOT included in the GitHub repository due to file size limitations.**

---

## Required Files

| File | Size | Description | Download Link |
|------|------|-------------|---------------|
| `yolov11_seg_best.pth` | ~45 MB | YOLOv11 segmentation model | [Zenodo](https://doi.org/10.5281/zenodo.XXXXXX) |
| `volume_student_resnet18.pth` | ~44 MB | ResNet-18 volume estimation | [Zenodo](https://doi.org/10.5281/zenodo.XXXXXX) |

---

## Download Instructions

### Option 1: Automatic Download

```bash
# From project root
python scripts/download_models.py
```

### Option 2: Manual Download

1. Visit the Zenodo repository:
   ```
   https://doi.org/10.5281/zenodo.XXXXXX
   ```

2. Download the model files:
   - `yolov11_seg_best.pth`
   - `volume_student_resnet18.pth`

3. Place them in this directory:
   ```
   checkpoints/
   ├── yolov11_seg_best.pth
   └── volume_student_resnet18.pth
   ```

### Option 3: Using wget

```bash
cd checkpoints/

# YOLOv11 model
wget https://zenodo.org/record/XXXXXX/files/yolov11_seg_best.pth

# Volume model
wget https://zenodo.org/record/XXXXXX/files/volume_student_resnet18.pth
```

---

## Verification

After downloading, verify the files:

```bash
# Check file sizes
ls -lh checkpoints/

# Expected output:
# -rw-r--r-- 1 user user  45M Nov 12 00:00 yolov11_seg_best.pth
# -rw-r--r-- 1 user user  44M Nov 12 00:00 volume_student_resnet18.pth
```

Test loading:

```bash
python -c "
from ultralytics import YOLO
import torch

# Test YOLO
yolo = YOLO('checkpoints/yolov11_seg_best.pth')
print('✓ YOLO model loaded')

# Test Volume
model = torch.load('checkpoints/volume_student_resnet18.pth')
print('✓ Volume model loaded')
"
```

---

## Model Details

### YOLOv11 Segmentation

- **Architecture**: YOLOv11n-seg
- **Input**: 640×640 RGB
- **Output**: Instance segmentation masks + bounding boxes
- **Classes**: 7 (bin, plastic bag, brick, wood, pipe, bottle, cardboard)
- **mAP50**: 82.1%
- **Training Set**: A-Det (846 frames)

### Volume Estimation (ResNet-18)

- **Architecture**: ResNet-18 (pretrained on ImageNet)
- **Input**: 448×448 RGB
- **Output**: Volume (liters)
- **Training**: Teacher-student distillation from RGB-D
- **MAE**: 2.84 L
- **R²**: 0.91

---

## Troubleshooting

### Issue: File Not Found

If you get `FileNotFoundError`:

1. Ensure files are in the correct directory
2. Check file names match exactly (case-sensitive)
3. Verify downloads completed successfully

### Issue: Model Loading Error

If you get `RuntimeError: Error(s) in loading state_dict`:

1. Ensure you downloaded the correct version
2. Check PyTorch version matches (2.0.1)
3. Try re-downloading the models

---

## License

Models are licensed under **CC BY 4.0**.

You are free to:
- Share and adapt the models
- Use for commercial purposes

With attribution to:
```
Yue Wu, The University of Auckland, 2025
```

---

**Last Updated**: November 2025

