"""
MIxTogether: AI-Driven In-bin Waste Monitoring System
Main entry point for the integrated detection, tracking, volume estimation, and event counting system.

Usage:
    python main.py --input <image_dir> --output <output_dir>
    python main.py --config configs/config.yaml
    python main.py --help

Author: Yue Wu
Institution: The University of Auckland
Year: 2025
"""

import sys
import argparse
from pathlib import Path
from loguru import logger

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.config import Config
from src.utils.logger import setup_logger
from src.integrated_processor import IntegratedProcessor


def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="MIxTogether: AI-Driven In-bin Waste Monitoring System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example Usage:
  # Process image sequence with default settings
  python main.py --input datasets/test_sequence

  # Use custom configuration
  python main.py --config configs/custom_config.yaml

  # Specify models and output
  python main.py --input ./images --output ./results \\
                 --yolo-model models/yolo.pt \\
                 --volume-model models/volume.pth

  # Disable video generation
  python main.py --input ./images --no-video

  # Verbose logging
  python main.py --input ./images --verbose

For more information, see: https://github.com/your-username/MIxTogether
        """
    )
    
    # Input/Output
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=None,
        help="Input image directory (containing frame_*.png)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="results",
        help="Output directory for results (default: results/)"
    )
    
    # Configuration
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to YAML configuration file"
    )
    
    # Models
    parser.add_argument(
        "--yolo-model",
        type=str,
        default=None,
        help="Path to YOLOv11 segmentation model (.pt file)"
    )
    
    parser.add_argument(
        "--volume-model",
        type=str,
        default=None,
        help="Path to volume estimation model (.pth file)"
    )
    
    parser.add_argument(
        "--baseline-image",
        type=str,
        default=None,
        help="Path to baseline (empty bin) image for volume estimation"
    )
    
    # Processing Options
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Do not generate output video"
    )
    
    parser.add_argument(
        "--no-save-frames",
        action="store_true",
        help="Do not save visualized frames"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Device to use for inference (default: auto)"
    )
    
    # Logging
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging (DEBUG level)"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="MIxTogether v1.0.0"
    )
    
    return parser.parse_args()


def validate_inputs(args, config):
    """Validate required files and directories exist"""
    errors = []
    
    # Check input directory
    if not config.input_dir.exists():
        errors.append(f"Input directory does not exist: {config.input_dir}")
    
    # Check YOLO model
    if not config.yolo_model_path.exists():
        errors.append(f"YOLO model not found: {config.yolo_model_path}")
        logger.warning("Please download the model from: https://universe.roboflow.com/...")
    
    # Check volume model
    if not config.volume_model_path.exists():
        errors.append(f"Volume model not found: {config.volume_model_path}")
        logger.warning("Please download the model from: https://doi.org/10.5281/zenodo....")
    
    # Check baseline image
    if not config.baseline_image.exists():
        errors.append(f"Baseline image not found: {config.baseline_image}")
        logger.warning("Please provide a baseline (empty bin) image")
    
    if errors:
        logger.error("Validation failed:")
        for error in errors:
            logger.error(f"  - {error}")
        logger.info("\nPlease refer to README.md for setup instructions")
        sys.exit(1)


def print_header():
    """Print welcome banner"""
    print("=" * 80)
    print(" " * 25 + "MIxTogether System")
    print(" " * 15 + "AI-Driven In-bin Waste Monitoring")
    print(" " * 20 + "Detection | Tracking | Volume | Events")
    print("=" * 80)
    print()


def print_config_summary(config):
    """Print configuration summary"""
    logger.info("Configuration Summary:")
    logger.info(f"  Input:          {config.input_dir}")
    logger.info(f"  Output:         {config.output_dir}")
    logger.info(f"  YOLO Model:     {config.yolo_model_path}")
    logger.info(f"  Volume Model:   {config.volume_model_path}")
    logger.info(f"  Baseline Image: {config.baseline_image}")
    logger.info(f"  Device:         {config.device}")
    logger.info(f"  Save Video:     {config.save_video}")
    logger.info(f"  Save Frames:    {config.save_visualized_frames}")
    print()


def main():
    """Main execution function"""
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Setup logging
        log_dir = Path(args.output) / "logs"
        setup_logger(log_dir=log_dir, verbose=args.verbose)
        
        # Print header
        print_header()
        logger.info("Starting MIxTogether system...")
        
        # Load configuration
        if args.config:
            logger.info(f"Loading configuration from: {args.config}")
            config = Config(config_path=args.config)
        else:
            logger.info("Using default configuration")
            config = Config()
        
        # Override with command-line arguments
        if args.input:
            config.input_dir = Path(args.input)
        if args.output:
            config.output_dir = Path(args.output)
        if args.yolo_model:
            config.yolo_model_path = Path(args.yolo_model)
        if args.volume_model:
            config.volume_model_path = Path(args.volume_model)
        if args.baseline_image:
            config.baseline_image = Path(args.baseline_image)
        if args.device != "auto":
            config.device = args.device
        if args.no_video:
            config.save_video = False
        if args.no_save_frames:
            config.save_visualized_frames = False
        
        # Print configuration
        print_config_summary(config)
        
        # Validate inputs
        logger.info("Validating inputs...")
        validate_inputs(args, config)
        logger.info("✓ All required files found")
        print()
        
        # Create output directory
        config.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize processor
        logger.info("Initializing integrated processor...")
        processor = IntegratedProcessor(config)
        logger.info("✓ Processor initialized successfully")
        print()
        
        # Process image sequence
        logger.info("Processing image sequence...")
        logger.info("-" * 80)
        results = processor.process_sequence(str(config.input_dir))
        logger.info("-" * 80)
        
        # Print summary
        logger.info("Processing complete!")
        logger.info("Results saved to: {}".format(config.output_dir))
        
        if "counting_results" in results:
            summary = results["counting_results"]["summary"]
            logger.info(f"Total events detected: {summary.get('total_objects', 0)}")
            
            if "class_distribution" in summary:
                logger.info("Class distribution:")
                for class_name, count in summary["class_distribution"].items():
                    if count > 0:
                        logger.info(f"  {class_name}: {count}")
        
        logger.info("=" * 80)
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning("\nUser interrupted execution")
        return 1
        
    except Exception as e:
        logger.error(f"Execution failed: {e}")
        if args.verbose:
            import traceback
            logger.error("Detailed traceback:")
            logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())

