"""
Setup script for MIxTogether package

Install in development mode:
    pip install -e .

Install for production:
    pip install .
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
if readme_path.exists():
    with open(readme_path, "r", encoding="utf-8") as f:
        long_description = f.read()
else:
    long_description = "MIxTogether: AI-Driven In-bin Waste Monitoring System"

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
if requirements_path.exists():
    with open(requirements_path, "r", encoding="utf-8") as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]
else:
    requirements = [
        "torch>=2.0.1",
        "torchvision>=0.15.2",
        "ultralytics>=8.0.196",
        "opencv-python>=4.8.0",
        "numpy>=1.24.3",
        "scipy>=1.10.1",
        "pandas>=2.0.2",
        "matplotlib>=3.7.2",
        "seaborn>=0.12.2",
        "loguru>=0.7.0",
        "pyyaml>=6.0",
        "Pillow>=10.0.0",
        "tqdm>=4.65.0",
        "scikit-learn>=1.3.0",
        "h5py>=3.9.0",
    ]

setup(
    name="mixtogether",
    version="1.0.0",
    author="Yue Wu",
    author_email="your.email@auckland.ac.nz",
    description="AI-Driven In-bin Waste Monitoring for Zero-Waste Construction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/MIxTogether",
    project_urls={
        "Bug Tracker": "https://github.com/your-username/MIxTogether/issues",
        "Documentation": "https://github.com/your-username/MIxTogether/wiki",
        "Datasets": "https://universe.roboflow.com/your-workspace/mixtogether",
        "Models": "https://doi.org/10.5281/zenodo.XXXXXX",
    },
    packages=find_packages(exclude=["tests", "docs", "scripts", "datasets", "checkpoints", "baseline"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.1.0",
            "black>=23.3.0",
            "flake8>=6.0.0",
            "mypy>=1.3.0",
            "isort>=5.12.0",
        ],
        "export": [
            "onnx>=1.14.0",
            "onnxruntime-gpu>=1.15.1",
        ],
        "tracking": [
            "tensorboard>=2.13.0",
            "wandb>=0.15.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "mixtogether=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json"],
    },
    keywords=[
        "computer vision",
        "object detection",
        "waste monitoring",
        "construction",
        "YOLOv11",
        "depth estimation",
        "event detection",
        "multi-object tracking",
    ],
    license="MIT",
    zip_safe=False,
)

