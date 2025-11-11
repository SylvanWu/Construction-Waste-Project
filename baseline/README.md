# Baseline Image

This directory should contain the baseline (empty bin) reference image for volume estimation.

---

## Required File

| File | Size | Description | Download Link |
|------|------|-------------|---------------|
| `frame_000009.png` | ~500 KB | Empty bin reference (1280×720) | [Zenodo](https://doi.org/10.5281/zenodo.XXXXXX) |

---

## Purpose

The baseline image is used by the volume estimation module to:
1. Establish the "zero fill level" reference
2. Compare current frames against empty bin state
3. Improve depth-based volume prediction accuracy

**This image must show the bin completely empty from the same camera angle used during inference.**

---

## Download Instructions

### Option 1: From Zenodo

```bash
cd baseline/
wget https://zenodo.org/record/XXXXXX/files/frame_000009.png
```

### Option 2: Manual Download

1. Visit: https://doi.org/10.5281/zenodo.XXXXXX
2. Download `frame_000009.png`
3. Place in this directory

---

## Verification

```bash
# Check file exists
ls -lh baseline/frame_000009.png

# Verify dimensions
file baseline/frame_000009.png
# Expected: PNG image data, 1280 x 720, 8-bit/color RGB

# Preview (requires ImageMagick or similar)
display baseline/frame_000009.png
```

---

## Creating Your Own Baseline

If you're using a different setup, create your own baseline image:

1. **Position your camera** at the monitoring location
2. **Ensure bin is completely empty**
3. **Capture a frame** (same resolution as inference)
4. **Save as PNG**:
   ```python
   import cv2
   frame = cv2.imread("your_frame.png")
   cv2.imwrite("baseline/frame_000009.png", frame)
   ```

5. **Update config**:
   ```yaml
   volume:
     baseline_frame: "baseline/frame_000009.png"
   ```

---

## Technical Requirements

- **Format**: PNG (lossless)
- **Resolution**: Must match inference frames (1280×720 recommended)
- **Color**: RGB (3 channels)
- **Content**: Empty bin, same camera angle/lighting as deployment
- **No**: People, moving objects, or temporary items

---

## Usage in Code

```python
from PIL import Image

# Load baseline
baseline = Image.open("baseline/frame_000009.png")

# Use in volume estimation
volume_estimator.set_baseline(baseline)
```

---

**Last Updated**: November 2025

