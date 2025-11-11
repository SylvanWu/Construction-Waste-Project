# Usage Guide

This guide explains how to use the MIxTogether system for in-bin waste monitoring.

---

## Quick Reference

```bash
# Basic usage
python main.py --input <image_dir> --output <output_dir>

# With custom config
python main.py --config configs/my_config.yaml

# Specify models
python main.py --input ./images \
               --yolo-model checkpoints/yolov11_seg_best.pth \
               --volume-model checkpoints/volume_student_resnet18.pth
```

---

## Command-Line Options

### Input/Output

| Option | Description | Default |
|--------|-------------|---------|
| `--input`, `-i` | Input image directory | (required) |
| `--output`, `-o` | Output directory | `results/` |

### Models

| Option | Description | Default |
|--------|-------------|---------|
| `--yolo-model` | Path to YOLOv11 model | `checkpoints/yolov11_seg_best.pth` |
| `--volume-model` | Path to volume model | `checkpoints/volume_student_resnet18.pth` |
| `--baseline-image` | Empty bin reference | `baseline/frame_000009.png` |

### Configuration

| Option | Description | Default |
|--------|-------------|---------|
| `--config`, `-c` | YAML config file | `configs/config.yaml` |
| `--device` | Device (auto/cuda/cpu) | `auto` |

### Processing

| Option | Description | Default |
|--------|-------------|---------|
| `--no-video` | Disable video generation | False |
| `--no-save-frames` | Don't save frames | False |

### Logging

| Option | Description | Default |
|--------|-------------|---------|
| `--verbose`, `-v` | Enable debug logging | False |

---

## Configuration File

Edit `configs/config.yaml` to customize system parameters:

### Detection Settings

```yaml
detection:
  confidence_threshold: 0.25  # Minimum confidence (0-1)
  iou_threshold: 0.7         # NMS IoU threshold
  bin_class_id: 0            # Class ID for bin
```

### Tracking Settings

```yaml
tracking:
  max_distance: 20           # Max center distance (pixels)
  disappear_frames: 15       # Frames before track removal
  iou_threshold: 0.6         # IoU matching threshold
  use_iou_matching: true     # Enable IoU-based matching
```

### Event Detection (FSM)

```yaml
counting:
  dwell_frames: 4            # Min frames inside bin
  cooldown_frames: 15        # Cooldown after event
  overlap_threshold: 0.3     # IoU for "inside bin"
  use_center_point: true     # Use center validation
```

### Volume Estimation

```yaml
volume:
  input_size: 448            # Model input size
  max_volume: 100.0          # Max bin capacity (L)
  estimate_every_frame: true # Estimate on all frames
  baseline_frame: "baseline/frame_000009.png"
```

---

## Output Files

After processing, the `results/` directory contains:

### 1. `complete_results.json`

Full system output with per-frame detections:

```json
{
  "frame_000001": {
    "frame_number": 1,
    "detections": [
      {
        "class": "wood",
        "confidence": 0.89,
        "bbox": [x1, y1, x2, y2],
        "track_id": 5,
        "in_bin": true
      }
    ],
    "volume": 12.34,
    "event": {
      "triggered": false
    }
  }
}
```

### 2. `counting_statistics.json`

Event counting summary:

```json
{
  "summary": {
    "total_objects": 34,
    "class_distribution": {
      "wood": 8,
      "brick": 7,
      "plastic_bag": 13,
      "bottle": 2,
      "cardboard": 4
    },
    "class_percentages": {
      "wood": 23.53,
      "brick": 20.59,
      ...
    }
  },
  "events": [
    {
      "event_id": 1,
      "frame": 42,
      "class": "wood",
      "timestamp": 2.8
    }
  ]
}
```

### 3. `volume_predictions.csv`

Frame-by-frame volume estimates:

```csv
frame,volume_liters,confidence
frame_000001,0.0,1.0
frame_000042,12.34,0.95
...
```

### 4. `summary_report.txt`

Human-readable summary:

```
MIxTogether Processing Summary
=====================================
Input: datasets/A-Val/images
Frames: 604
Duration: 40.27s

Detection Results:
  Total Detections: 1234
  Classes: 7

Counting Results:
  Total Events: 34
  wood: 8 (23.5%)
  brick: 7 (20.6%)
  ...

Volume Statistics:
  Mean: 12.45 L
  Max: 45.67 L
  Min: 0.00 L
```

### 5. `visualized_frames/`

Annotated frames with:
- Bounding boxes
- Segmentation masks
- Track IDs
- Volume overlay
- Event markers

### 6. `analysis_video.mp4`

Full video with all visualizations (10 FPS by default).

---

## Examples

### Example 1: Process Test Sequence

```bash
python main.py \
    --input datasets/A-Val/images \
    --output results/test_run \
    --verbose
```

### Example 2: Custom Configuration

Create `configs/my_config.yaml`:

```yaml
detection:
  confidence_threshold: 0.3  # Lower threshold

counting:
  dwell_frames: 5           # Stricter dwell
  cooldown_frames: 20       # Longer cooldown
```

Run:

```bash
python main.py --config configs/my_config.yaml --input ./images
```

### Example 3: CPU-Only Inference

```bash
python main.py --input ./images --device cpu --no-video
```

### Example 4: Fast Processing (No Visualization)

```bash
python main.py \
    --input ./images \
    --no-video \
    --no-save-frames
```

---

## Evaluation Scripts

### E1: Event Detection Evaluation

Compare system output with ground truth:

```bash
python scripts/eval_e1.py \
    --gt-events data/gt_events.json \
    --pred-events results/complete_results.json \
    --output results/e1_metrics.json \
    --window 20  # ±20 frame matching window
```

**Output** (`e1_metrics.json`):
```json
{
  "precision": 0.7451,
  "recall": 0.4937,
  "f1_score": 0.5946,
  "tp": 15,
  "fp": 5,
  "fn": 19,
  "per_class": {
    "wood": {"precision": 0.80, "recall": 0.62, "f1": 0.70},
    ...
  }
}
```

### E2: Volume Estimation Evaluation

Evaluate volume prediction accuracy:

```bash
python scripts/eval_e2.py \
    --predictions results/volume_predictions.csv \
    --ground-truth data/volume_gt.csv \
    --output results/e2_metrics.json
```

**Output** (`e2_metrics.json`):
```json
{
  "mae": 2.84,
  "mape": 25.39,
  "mdae": 2.10,
  "rmse": 4.12,
  "r_squared": 0.91
}
```

### E3: Runtime Benchmarking

Measure system performance:

```bash
python scripts/eval_e3.py \
    --video data/test_video.mp4 \
    --model checkpoints/yolov11_seg_best.pth \
    --output results/e3_runtime.json \
    --iterations 100
```

**Output** (`e3_runtime.json`):
```json
{
  "fps": 15.2,
  "vram_peak_gb": 4.2,
  "ram_peak_gb": 3.8,
  "alert_latency_ms": 270,
  "detection_ms": 22,
  "tracking_ms": 8,
  "volume_ms": 15
}
```

---

## Tips & Best Practices

### 1. Optimize for Speed

- Use `--no-save-frames` if you don't need visualizations
- Set `estimate_every_frame: false` to skip volume on some frames
- Reduce `confidence_threshold` to filter more detections

### 2. Improve Accuracy

- Adjust `dwell_frames` based on your camera FPS
- Tune `overlap_threshold` for your bin size
- Calibrate `baseline_image` for your specific setup

### 3. Batch Processing

Process multiple sequences:

```bash
#!/bin/bash
for dir in datasets/*/; do
    python main.py --input "$dir" --output "results/$(basename $dir)"
done
```

### 4. Monitor GPU Usage

```bash
# In another terminal
watch -n 1 nvidia-smi
```

---

## Troubleshooting

### Issue: Low FPS

**Solutions**:
- Reduce image resolution
- Disable video generation (`--no-video`)
- Use smaller model (if available)

### Issue: Missed Events

**Solutions**:
- Lower `confidence_threshold` (e.g., 0.2)
- Reduce `dwell_frames` (e.g., 3)
- Increase `disappear_frames` (e.g., 20)

### Issue: False Positives

**Solutions**:
- Increase `dwell_frames` (e.g., 6)
- Decrease `overlap_threshold` (e.g., 0.2)
- Increase `cooldown_frames` (e.g., 20)

---

## Advanced Usage

### Custom Model Integration

Replace YOLOv11 with your own model:

```python
# src/detection/custom_model.py
from ultralytics import YOLO

model = YOLO("path/to/your/model.pt")
results = model.predict(image, conf=0.25)
```

### Export to ONNX

For deployment optimization:

```bash
python scripts/export_onnx.py \
    --yolo-model checkpoints/yolov11_seg_best.pth \
    --volume-model checkpoints/volume_student_resnet18.pth \
    --output models_onnx/
```

---

## Next Steps

- **Experiments**: See [EXPERIMENTS.md](EXPERIMENTS.md)
- **API Reference**: See [API.md](API.md)
- **Contributing**: See [CONTRIBUTING.md](../CONTRIBUTING.md)

---

**Last Updated**: November 2025

