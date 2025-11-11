"""
Volume Estimation Module: ResNet-18 Student Model

This module provides:
- RGB-only volume estimation (distilled from RGB-D teacher)
- ResNet-18 architecture
- Preprocessing for volume estimation
- Teacher-student training utilities
"""

__all__ = ["baseline_b1", "losses", "dataset", "transforms"]
