"""
Main Experiment Orchestrator for Edge AI Object Detection

Runs complete experiment suite: 3 models × 4 optimizations = 12 experiments
Collects all metrics and saves results to CSV.

Author: Edge AI Research Team
Date: 2024
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from tqdm import tqdm

from model_loader import ModelLoader
from model_optimizer import ModelOptimizer
from evaluator import ModelEvaluator
from hardware_monitor import HardwareMonitor

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """
    Orchestrate complete experiment suite.
    
    Systematically tests all model-optimization combinations and
    collects comprehensive performance metrics.
    """
    
    MODELS = ['mobilenet', 'squeezenet', 'yolo']
    OPTIMIZATIONS = ['baseline', 'quantization', 'pruning', 'combined']
    
    MODEL_INPUT_SIZES = {
        'mobilenet': (224, 224),
        'squeezenet': (227, 227),
        'yolo': (416, 416)
    }
    
    MODEL_TYPES = {
        'mobilenet': 'tflite',
        'squeezenet': 'tflite',
        'yolo': 'opencv'
    }

    MODEL_PATHS = {
        'mobilenet': {
            'tflite': 'mobilenet_v2.tflite',
            'h5': 'mobilenet_v2.h5'
        },
        'squeezenet': {
            'tflite': 'squeezenet.tflite',
            'h5': 'squeezenet.h5'
        },
        'yolo': {
            'tflite': 'yolo.tflite',
            'h5': 'yolo.h5'
        }
    }
    
    def __init__(self, output_dir: str = "output/csv", models_dir: str = "models",
                 config: Optional[Dict] = None):
        """
        Initialize experiment runner.
        
        Args:
            output_dir: Directory to save results
            models_dir: Directory containing models
            config: Configuration dictionary
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.models_dir = models_dir
        self.config = config or {}
        
        # Initialize components
        self.model_loader = ModelLoader(models_dir=models_dir)
        self.model_optimizer = ModelOptimizer()
        self.hardware_monitor = HardwareMonitor()
        self.evaluator = ModelEvaluator(hardware_monitor=self.hardware_monitor)
        
        # Results storage
        self.results: List[Dict] = []
        
        logger.info("ExperimentRunner initialized")
        logger.info("Models: %s", self.MODELS)
        logger.info("Optimizations: %s", self.OPTIMIZATIONS)
    
    def run_single_experiment(self, model_name: str, optimization: str) -> Dict:
        """
        Run a single model-optimization experiment.
        
        Args:
            model_name: Name of model to test
            optimization: Optimization to apply
        
        Returns:
            Dictionary with experiment results
        """
        logger.info("=" * 70)
        logger.info("EXPERIMENT: %s + %s", model_name.upper(), optimization.upper())
        logger.info("=" * 70)
        
        start_time = time.time()
        
        try:
            # Step 1: Load model
            logger.info("[Step 1/4] Loading %s model...", model_name)
            model = self._load_model(model_name, optimization)
            
            # Step 2: Apply optimization if needed
            logger.info("[Step 2/4] Applying %s optimization...", optimization)
            model, model_path = self._apply_optimization(model, model_name, optimization)
            
            # Step 3: Determine model type and input size
            model_type = self.MODEL_TYPES.get(model_name, 'placeholder')
            input_size = self.MODEL_INPUT_SIZES.get(model_name, (224, 224))
            
            # Check if model is placeholder
            if isinstance(model, dict) and model.get('is_placeholder', False):
                model_type = 'placeholder'
                logger.warning("Using placeholder model - results will be simulated")
            
            # Step 4: Run benchmark
            logger.info("[Step 3/4] Running benchmark...")
            results = self.evaluator.run_full_benchmark(
                model=model,
                model_name=model_name,
                optimization=optimization,
                model_path=model_path,
                input_size=input_size,
                model_type=model_type
            )
            
            # Step 5: Collect hardware metrics
            logger.info("[Step 4/4] Collecting hardware metrics...")
            hardware_metrics = self.hardware_monitor.get_average_metrics()
            results.update(hardware_metrics)
            
            # Add timestamp
            results['timestamp'] = time.strftime("%Y-%m-%d %H:%M:%S")
            results['experiment_duration_s'] = time.time() - start_time
            
            logger.info("=" * 70)
            logger.info("EXPERIMENT COMPLETE: %s + %s", model_name, optimization)
            logger.info("  FPS: %.2f | Latency: %.2f ms | mAP: %.3f",
                       results.get('fps', 0), results.get('latency_ms', 0), results.get('map_score', 0))
            logger.info("=" * 70)
            
            return results
            
        except Exception as e:
            logger.error("Experiment failed: %s", str(e))
            
            # Return partial results with error
            error_result = {
                'model': model_name,
                'optimization': optimization,
                'error': str(e),
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
            }
            return error_result
    
    def _load_model(self, model_name: str, optimization: str) -> Any:
        """Load model based on name."""
        if model_name == 'mobilenet':
            return self.model_loader.load_mobilenet()
        elif model_name == 'squeezenet':
            return self.model_loader.load_squeezenet()
        elif model_name == 'yolo':
            return self.model_loader.load_yolo_lightweight()
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    def _apply_optimization(self, model: Any, model_name: str, optimization: str):
        """
        Apply optimization to model.
        
        Returns:
            Tuple of (optimized_model, model_path)
        """
        model_paths = self.MODEL_PATHS.get(model_name, {})
        tflite_path = f"{self.models_dir}/{model_paths.get('tflite', f'{model_name}.tflite')}"
        h5_path = f"{self.models_dir}/{model_paths.get('h5', f'{model_name}.h5')}"

        if optimization == 'baseline':
            # No optimization
            return model, tflite_path
        
        elif optimization == 'quantization':
            # Apply quantization
            try:
                optimized_path = self.model_optimizer.apply_quantization(
                    tflite_path,
                    precision='int8'
                )
                return model, optimized_path
            except Exception as e:
                logger.warning("Quantization failed (%s), using baseline", str(e))
                return model, tflite_path
        
        elif optimization == 'pruning':
            # Apply pruning (requires Keras model)
            try:
                optimized_path = self.model_optimizer.apply_pruning(
                    model,
                    sparsity=0.5,
                    output_path=f"{self.models_dir}/{model_name}_pruned.h5"
                )
                return optimized_path, f"{self.models_dir}/{model_name}_pruned.h5"
            except Exception as e:
                logger.warning("Pruning failed (%s), using baseline", str(e))
                return model, h5_path
        
        elif optimization == 'combined':
            # Apply combined optimization
            try:
                optimized_path = self.model_optimizer.apply_combined(
                    h5_path,
                    sparsity=0.5,
                    precision='int8'
                )
                return model, optimized_path
            except Exception as e:
                logger.warning("Combined optimization failed (%s), using baseline", str(e))
                return model, h5_path
        
        else:
            raise ValueError(f"Unknown optimization: {optimization}")
    
    def run_full_experiment_suite(self, models: Optional[List[str]] = None,
                                  optimizations: Optional[List[str]] = None,
                                  save_results: bool = True) -> pd.DataFrame:
        """
        Run complete experiment suite (all model-optimization combinations).
        
        Args:
            models: List of models to test (None = all)
            optimizations: List of optimizations to test (None = all)
            save_results: Whether to save results to CSV
        
        Returns:
            DataFrame with all results
        """
        if models is None:
            models = self.MODELS
        if optimizations is None:
            optimizations = self.OPTIMIZATIONS
        
        total_experiments = len(models) * len(optimizations)
        
        logger.info("=" * 70)
        logger.info("FULL EXPERIMENT SUITE")
        logger.info("Models: %s", models)
        logger.info("Optimizations: %s", optimizations)
        logger.info("Total experiments: %d", total_experiments)
        logger.info("=" * 70)
        
        start_time = time.time()
        self.results = []
        
        # Run experiments with progress bar
        for model_name in tqdm(models, desc="Models"):
            for optimization in tqdm(optimizations, desc=f"{model_name} optimizations", leave=False):
                logger.info("\nExperiment %d/%d", 
                          len(self.results) + 1, total_experiments)
                
                result = self.run_single_experiment(model_name, optimization)
                self.results.append(result)
                
                # Save intermediate results
                if save_results and len(self.results) % 3 == 0:
                    self._save_results()
                    logger.info("Intermediate results saved")
        
        total_duration = time.time() - start_time
        
        # Final save
        if save_results:
            df = self._save_results()
        else:
            df = pd.DataFrame(self.results)
        
        # Print summary
        self._print_summary(df, total_duration)
        
        return df
    
    def _save_results(self) -> pd.DataFrame:
        """Save results to CSV file."""
        df = pd.DataFrame(self.results)
        
        csv_path = self.output_dir / "experiment_results.csv"
        df.to_csv(csv_path, index=False)
        
        logger.info("Results saved to: %s", csv_path)
        return df
    
    def _print_summary(self, df: pd.DataFrame, total_duration: float):
        """Print experiment summary."""
        logger.info("\n" + "=" * 70)
        logger.info("EXPERIMENT SUITE COMPLETE")
        logger.info("=" * 70)
        
        logger.info("\nTotal experiments: %d", len(df))
        logger.info("Total duration: %.1f seconds (%.1f minutes)", 
                   total_duration, total_duration / 60)
        
        # Average metrics by model
        logger.info("\nAverage Performance by Model:")
        logger.info("-" * 70)
        
        for model in self.MODELS:
            model_df = df[df['model'] == model]
            if len(model_df) > 0:
                avg_fps = model_df['fps'].mean() if 'fps' in model_df.columns else 0
                avg_lat = model_df['latency_ms'].mean() if 'latency_ms' in model_df.columns else 0
                avg_map = model_df['map_score'].mean() if 'map_score' in model_df.columns else 0
                
                logger.info("%-12s: FPS=%6.2f | Latency=%7.2f ms | mAP=%.3f",
                          model.upper(), avg_fps, avg_lat, avg_map)
        
        # Best performing configuration
        if 'fps' in df.columns and len(df) > 0:
            best_idx = df['fps'].idxmax()
            best = df.loc[best_idx]
            logger.info("\nBest FPS: %s + %s (%.2f FPS)",
                       best['model'], best['optimization'], best['fps'])
        
        if 'map_score' in df.columns and len(df) > 0:
            best_map_idx = df['map_score'].idxmax()
            best_map = df.loc[best_map_idx]
            logger.info("Best mAP:  %s + %s (%.3f)",
                       best_map['model'], best_map['optimization'], best_map['map_score'])
        
        logger.info("\nResults saved to: %s", self.output_dir / "experiment_results.csv")
        logger.info("=" * 70)
    
    def load_previous_results(self, csv_path: str) -> pd.DataFrame:
        """
        Load results from previous experiment.
        
        Args:
            csv_path: Path to CSV file
        
        Returns:
            DataFrame with results
        """
        path = Path(csv_path)
        if not path.exists():
            logger.error("Results file not found: %s", csv_path)
            raise FileNotFoundError(f"Results not found: {csv_path}")
        
        df = pd.read_csv(csv_path)
        logger.info("Loaded %d experiments from: %s", len(df), csv_path)
        return df


def run_experiments(config: Optional[Dict] = None, save_results: bool = True) -> pd.DataFrame:
    """
    Convenience function to run experiments.
    
    Args:
        config: Configuration dictionary
        save_results: Whether to save results
    
    Returns:
        DataFrame with results
    """
    runner = ExperimentRunner(config=config)
    return runner.run_full_experiment_suite(save_results=save_results)


if __name__ == "__main__":
    # Test experiment runner
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s %(message)s'
    )
    
    print("=" * 70)
    print("Edge AI Object Detection Experiment")
    print("Raspberry Pi 4 - Lightweight CNN Evaluation")
    print("=" * 70)
    
    # Run with placeholder models (simulated results)
    runner = ExperimentRunner()
    df = runner.run_full_experiment_suite(save_results=True)
    
    print("\nExperiment Results:")
    print(df.to_string(index=False))
    
    print("\n" + "=" * 70)
    print("All experiments complete!")
    print("=" * 70)
