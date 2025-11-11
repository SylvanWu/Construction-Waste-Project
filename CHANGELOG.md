# Changelog

All notable changes to the MIxTogether project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] - 2025-11-12

### Added
- Initial release of MIxTogether system
- YOLOv11 segmentation for 7 waste classes
- Multi-object tracking (center + IoU + ReID)
- FSM-based event detection (dwell/cooldown logic)
- ResNet-18 volume estimation (teacher-student distillation)
- Full integration pipeline with real-time visualization
- E1/E2/E3 evaluation scripts
- Comprehensive documentation (README, INSTALLATION, USAGE)
- Dataset download scripts for Roboflow
- Configuration management with YAML
- Logging system with loguru
- Model checkpoints (hosted on Zenodo)
- Datasets (hosted on Roboflow Universe)

### Features
- **Detection**: mAP50 82.1%, mAP50-95 65.3%
- **Volume**: MAE 2.84L, R² 0.91
- **Events**: F1 59.46%, Precision 74.51%
- **Runtime**: 15.2 FPS on GTX 1080 Ti
- **Memory**: 4.2 GB VRAM, 3.8 GB RAM

### Documentation
- Installation guide with troubleshooting
- Usage guide with examples
- API reference
- Dataset documentation
- Model documentation

---

## [Unreleased]

### Planned
- [ ] ONNX export for deployment
- [ ] TensorRT optimization
- [ ] Raspberry Pi / Jetson support
- [ ] Cloud dashboard integration
- [ ] Multi-bin support
- [ ] Uncertainty quantification
- [ ] Active learning pipeline
- [ ] Real-time alerting system

---

## Version History

### v1.0.0 (2025-11-12)
Initial public release for Master's thesis

---

## Notes

For detailed experiment results, see:
- `artifacts/experiments/FINAL_EXPERIMENT_SUMMARY.md`
- Thesis document (submitted Nov 2025)

For dataset details, see:
- [Roboflow Universe](https://universe.roboflow.com/your-workspace/mixtogether)
- [Zenodo DOI](https://doi.org/10.5281/zenodo.XXXXXX)

