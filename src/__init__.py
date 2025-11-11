"""
MIxTogether: AI-Driven In-bin Waste Monitoring System

This package contains the core modules for:
- Object detection and segmentation (YOLOv11)
- Multi-object tracking (MOT)
- Event detection (FSM)
- Volume estimation (ResNet-18)

Author: Yue Wu
Institution: The University of Auckland
Year: 2025
"""

__version__ = "1.0.0"
__author__ = "Yue Wu"
__email__ = "your.email@auckland.ac.nz"
__license__ = "MIT"

from pathlib import Path

# Package root directory
PACKAGE_ROOT = Path(__file__).parent
PROJECT_ROOT = PACKAGE_ROOT.parent

# Default paths
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"
DEFAULT_CHECKPOINTS = PROJECT_ROOT / "checkpoints"
DEFAULT_BASELINE = PROJECT_ROOT / "baseline"

__all__ = [
    "__version__",
    "__author__",
    "PACKAGE_ROOT",
    "PROJECT_ROOT",
    "DEFAULT_CONFIG",
    "DEFAULT_CHECKPOINTS",
    "DEFAULT_BASELINE",
]
