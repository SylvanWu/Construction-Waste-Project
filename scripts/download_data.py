"""
Data Download Script
Downloads datasets from Roboflow Universe

Usage:
    python scripts/download_data.py --dataset aval
    python scripts/download_data.py --dataset adet --output custom_path/
"""

import argparse
import sys
from pathlib import Path

try:
    from roboflow import Roboflow
except ImportError:
    print("Error: roboflow package not installed")
    print("Install it with: pip install roboflow")
    sys.exit(1)


# Dataset configurations
DATASETS = {
    "dataset": {
        "workspace": "aidriven-waste-identification-and-realtime-monitoring-for-zerowaste-construction-a-depthenhanced-com",
        "project": "construction-waste-project-gdkmi",
        "version": 1,
        "description": "Construction Waste Dataset (validation and training sets)"
    }
}


def download_dataset(dataset_name: str, output_dir: Path, api_key: str = None):
    """
    Download dataset from Roboflow Universe
    
    Args:
        dataset_name: Dataset identifier ('aval' or 'adet')
        output_dir: Output directory path
        api_key: Roboflow API key (optional, uses environment variable if not provided)
    """
    if dataset_name not in DATASETS:
        print(f"Error: Unknown dataset '{dataset_name}'")
        print(f"Available datasets: {list(DATASETS.keys())}")
        sys.exit(1)
    
    config = DATASETS[dataset_name]
    
    print(f"Downloading dataset: {config['description']}")
    print(f"Workspace: {config['workspace']}")
    print(f"Project: {config['project']}")
    print(f"Version: {config['version']}")
    print()
    
    # Initialize Roboflow
    rf = Roboflow(api_key=api_key)
    
    # Get project
    project = rf.workspace(config["workspace"]).project(config["project"])
    dataset = project.version(config["version"])
    
    # Download in YOLO format
    print(f"Downloading to: {output_dir}")
    dataset.download(
        "yolov11",  # Format
        location=str(output_dir)
    )
    
    print(f"✓ Dataset downloaded successfully to {output_dir}")
    print()
    
    # Print directory structure
    print("Directory structure:")
    for item in output_dir.rglob("*"):
        if item.is_file():
            print(f"  {item.relative_to(output_dir)}")


def main():
    parser = argparse.ArgumentParser(
        description="Download MIxTogether datasets from Roboflow Universe",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download A-Val dataset
  python scripts/download_data.py --dataset aval

  # Download A-Det dataset to custom path
  python scripts/download_data.py --dataset adet --output my_data/

  # Use custom API key
  python scripts/download_data.py --dataset aval --api-key YOUR_KEY

Available datasets:
  aval - A-Val validation set (604 frames, 34 GT events)
  adet - A-Det training set (846 frames with masks)

Note: You need a Roboflow account and API key.
Get your API key from: https://app.roboflow.com/settings/api
        """
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=list(DATASETS.keys()),
        help="Dataset to download (aval or adet)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: datasets/<dataset_name>)"
    )
    
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Roboflow API key (optional, uses ROBOFLOW_API_KEY env var if not provided)"
    )
    
    args = parser.parse_args()
    
    # Set default output path
    if args.output is None:
        output_dir = Path("datasets") / args.dataset.upper().replace("_", "-")
    else:
        output_dir = Path(args.output)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Download dataset
    try:
        download_dataset(args.dataset, output_dir, api_key=args.api_key)
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print()
        print("Troubleshooting:")
        print("1. Check your API key is correct")
        print("2. Ensure you have access to the dataset")
        print("3. Check your internet connection")
        sys.exit(1)


if __name__ == "__main__":
    main()

