"""
Main Entry Point for Edge AI Object Detection Experiments

Command-line interface to run single experiments, full experiment suites,
real-time inference, and visualization generation.

Author: Edge AI Research Team
Date: 2024
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from model_loader import ModelLoader
from model_optimizer import ModelOptimizer
from evaluator import ModelEvaluator
from hardware_monitor import HardwareMonitor
from camera_capture import CameraCapture
from inference_with_boxes import InferenceWithBoxes
from main_experiment import ExperimentRunner
from visualization import ResultsVisualizer
from dataset_prepare import DatasetPreparer

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
    
    Returns:
        Configuration dictionary
    """
    path = Path(config_path)
    if not path.exists():
        logger.warning("Config file not found: %s, using defaults", config_path)
        return {}
    
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info("Configuration loaded from: %s", config_path)
    return config


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """
    Configure logging.
    
    Args:
        log_level: Logging level
        log_file: Optional log file path
    """
    # Ensure log directory exists
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='[%(asctime)s] %(levelname)s %(message)s',
        handlers=handlers
    )


def print_banner():
    """Print experiment banner."""
    print("=" * 70)
    print("Edge AI Object Detection Experiment")
    print("Raspberry Pi 4 - Lightweight CNN Evaluation")
    print("=" * 70)
    print()


def run_single_experiment(args, config):
    """
    Run a single model-optimization experiment.
    
    Args:
        args: Command-line arguments
        config: Configuration dictionary
    """
    logger.info("Running single experiment: %s + %s", args.model, args.optimization)
    
    # Initialize components
    runner = ExperimentRunner(config=config)
    
    # Run experiment
    result = runner.run_single_experiment(args.model, args.optimization)
    
    # Display results
    print("\n" + "=" * 70)
    print("Experiment Results:")
    print("=" * 70)
    print(f"Model:          {result.get('model', 'N/A')}")
    print(f"Optimization:   {result.get('optimization', 'N/A')}")
    print(f"FPS:            {result.get('fps', 0):.2f}")
    print(f"Latency:        {result.get('latency_ms', 0):.2f} ms")
    print(f"mAP:            {result.get('map_score', 0):.3f}")
    print(f"Model Size:     {result.get('model_size_mb', 0):.2f} MB")
    print(f"Energy:         {result.get('energy_joules', 0):.2f} Joules")
    print(f"CPU Usage:      {result.get('cpu_percent', 0):.1f}%")
    print(f"RAM Usage:      {result.get('ram_mb', 0):.1f} MB")
    print("=" * 70)
    
    # Save if requested
    if args.save_results:
        results_df = runner._save_results()
        print(f"\nResults saved to: {runner.output_dir / 'experiment_results.csv'}")


def run_full_experiment(args, config):
    """
    Run complete experiment suite.
    
    Args:
        args: Command-line arguments
        config: Configuration dictionary
    """
    logger.info("Running full experiment suite")
    
    # Initialize runner
    runner = ExperimentRunner(config=config)
    
    # Run all experiments
    df = runner.run_full_experiment_suite(save_results=args.save_results)
    
    print("\n" + "=" * 70)
    print("All Experiments Complete!")
    print("=" * 70)
    print(f"\nTotal experiments: {len(df)}")
    print(f"Results saved to: {runner.output_dir / 'experiment_results.csv'}")
    
    # Generate visualizations if requested
    if args.generate_graphs:
        print("\nGenerating visualizations...")
        visualizer = ResultsVisualizer(
            results_path=str(runner.output_dir / "experiment_results.csv")
        )
        graph_paths = visualizer.generate_all_graphs()
        print(f"Generated {len(graph_paths)} graphs in output/graphs/")


def run_realtime_inference(args, config):
    """
    Run real-time object detection.
    
    Args:
        args: Command-line arguments
        config: Configuration dictionary
    """
    logger.info("Starting real-time inference")
    
    # Load model
    loader = ModelLoader()
    
    if args.model == 'mobilenet':
        model = loader.load_mobilenet()
        model_type = 'tflite'
        input_size = tuple(config.get('models', {}).get('mobilenet', {}).get('input_size', [224, 224]))
    elif args.model == 'squeezenet':
        model = loader.load_squeezenet()
        model_type = 'tflite'
        input_size = tuple(config.get('models', {}).get('squeezenet', {}).get('input_size', [227, 227]))
    elif args.model == 'yolo':
        model = loader.load_yolo_lightweight()
        model_type = 'opencv'
        input_size = tuple(config.get('models', {}).get('yolo', {}).get('input_size', [416, 416]))
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    # Initialize inference
    inference = InferenceWithBoxes(
        model=model,
        model_type=model_type,
        confidence_threshold=config.get('models', {}).get('yolo', {}).get('confidence_threshold', 0.5),
        nms_threshold=config.get('models', {}).get('yolo', {}).get('nms_threshold', 0.4),
        input_size=input_size
    )
    
    # Run video stream
    print("\n" + "=" * 70)
    print("Real-Time Object Detection")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Optimization: {args.optimization}")
    print(f"Press 'q' to quit")
    print("=" * 70 + "\n")
    
    results = inference.process_video_stream(duration_seconds=60.0, save_output=False)
    
    print("\n" + "=" * 70)
    print("Inference Complete")
    print("=" * 70)
    print(f"Frames processed: {results.get('frames_processed', 0)}")
    print(f"Average FPS: {results.get('avg_fps', 0):.2f}")
    print(f"Average inference time: {results.get('avg_inference_time_ms', 0):.2f} ms")
    print("=" * 70)


def run_benchmark(args, config):
    """
    Run benchmark on specific model and optimization.
    
    Args:
        args: Command-line arguments
        config: Configuration dictionary
    """
    logger.info("Running benchmark: %s + %s", args.model, args.optimization)
    
    # Load model
    loader = ModelLoader()
    
    if args.model == 'mobilenet':
        model = loader.load_mobilenet()
        model_type = 'tflite'
    elif args.model == 'squeezenet':
        model = loader.load_squeezenet()
        model_type = 'tflite'
    elif args.model == 'yolo':
        model = loader.load_yolo_lightweight()
        model_type = 'opencv'
    else:
        model = {'type': args.model, 'is_placeholder': True}
        model_type = 'placeholder'
    
    # Run evaluation
    evaluator = ModelEvaluator()
    input_size = (224, 224) if args.model in ['mobilenet', 'squeezenet'] else (416, 416)
    
    results = evaluator.run_full_benchmark(
        model=model,
        model_name=args.model,
        optimization=args.optimization,
        input_size=input_size,
        model_type=model_type
    )
    
    # Display results
    print("\n" + "=" * 70)
    print("Benchmark Results")
    print("=" * 70)
    for key, value in results.items():
        if isinstance(value, float):
            print(f"{key:20s}: {value:.3f}")
        else:
            print(f"{key:20s}: {value}")
    print("=" * 70)


def prepare_data(args, config):
    """
    Prepare datasets and download models.
    
    Args:
        args: Command-line arguments
        config: Configuration dictionary
    """
    logger.info("Preparing datasets")
    
    preparer = DatasetPreparer()
    
    if args.download_models:
        print("\nDownloading pretrained models...")
        models = preparer.download_pretrained_models(force_download=args.force)
        print(f"\nDownloaded {len(models)} models")
    
    if args.dataset == 'cifar10' or args.dataset == 'all':
        print("\nPreparing CIFAR-10...")
        cifar_path = preparer.download_cifar10(force_download=args.force)
        print(f"CIFAR-10 ready at: {cifar_path}")
    
    if args.dataset == 'pascal_voc' or args.dataset == 'all':
        print("\nPreparing Pascal VOC...")
        voc_path = preparer.download_pascal_voc(force_download=args.force)
        print(f"Pascal VOC ready at: {voc_path}")
    
    # Display dataset info
    print("\n" + "=" * 70)
    print("Dataset Information")
    print("=" * 70)
    info = preparer.get_dataset_info()
    for dataset, details in info.items():
        print(f"\n{dataset.upper()}:")
        for key, value in details.items():
            print(f"  {key}: {value}")
    print("=" * 70)


def generate_visualizations(args, config):
    """
    Generate visualization graphs from results.
    
    Args:
        args: Command-line arguments
        config: Configuration dictionary
    """
    logger.info("Generating visualizations")
    
    results_path = args.results if hasattr(args, 'results') and args.results else "output/csv/experiment_results.csv"
    
    visualizer = ResultsVisualizer(results_path=results_path)
    graph_paths = visualizer.generate_all_graphs()
    
    print("\n" + "=" * 70)
    print("Visualizations Generated")
    print("=" * 70)
    for path in graph_paths:
        print(f"  ✓ {path}")
    print("=" * 70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Edge AI Object Detection Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run single experiment
  python main.py --model mobilenet --optimization baseline --mode single
  
  # Run full experiment suite
  python main.py --mode full_experiment
  
  # Real-time inference
  python main.py --model yolo --mode realtime
  
  # Generate visualizations
  python main.py --mode visualize --results output/csv/experiment_results.csv
  
  # Download models and datasets
  python main.py --mode prepare --dataset all --download-models
        """
    )
    
    # Mode selection
    parser.add_argument(
        '--mode', 
        type=str, 
        choices=['single', 'full_experiment', 'realtime', 'benchmark', 'prepare', 'visualize'],
        default='single',
        help='Operation mode (default: single)'
    )
    
    # Model selection
    parser.add_argument(
        '--model',
        type=str,
        choices=['mobilenet', 'squeezenet', 'yolo'],
        default='mobilenet',
        help='Model to use (default: mobilenet)'
    )
    
    # Optimization selection
    parser.add_argument(
        '--optimization',
        type=str,
        choices=['baseline', 'quantization', 'pruning', 'combined'],
        default='baseline',
        help='Optimization to apply (default: baseline)'
    )
    
    # Dataset selection
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['cifar10', 'pascal_voc', 'all'],
        default='cifar10',
        help='Dataset to prepare (default: cifar10)'
    )
    
    # Optional flags
    parser.add_argument(
        '--save_results',
        action='store_true',
        help='Save results to CSV'
    )
    
    parser.add_argument(
        '--generate_graphs',
        action='store_true',
        help='Generate visualization graphs after experiments'
    )
    
    parser.add_argument(
        '--download-models',
        action='store_true',
        help='Download pretrained models'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-download/re-generate'
    )
    
    parser.add_argument(
        '--results',
        type=str,
        default='output/csv/experiment_results.csv',
        help='Path to results CSV for visualization'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose/debug logging'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else config.get('logging', {}).get('level', 'INFO')
    log_file = config.get('logging', {}).get('file', 'output/logs/experiment.log')
    setup_logging(log_level, log_file)
    
    # Print banner
    print_banner()
    
    try:
        # Execute based on mode
        if args.mode == 'single':
            run_single_experiment(args, config)
        
        elif args.mode == 'full_experiment':
            run_full_experiment(args, config)
        
        elif args.mode == 'realtime':
            run_realtime_inference(args, config)
        
        elif args.mode == 'benchmark':
            run_benchmark(args, config)
        
        elif args.mode == 'prepare':
            prepare_data(args, config)
        
        elif args.mode == 'visualize':
            generate_visualizations(args, config)
        
        else:
            parser.print_help()
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
    
    except Exception as e:
        logger.error("Fatal error: %s", str(e), exc_info=True)
        print(f"\n✗ Error: {e}")
        print("Check logs for details: output/logs/experiment.log")
        sys.exit(1)


if __name__ == "__main__":
    main()
