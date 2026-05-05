"""
Model Optimizer Module for Edge AI Object Detection

Implements quantization, pruning, and combined optimization techniques
for TensorFlow Lite and Keras models.

Author: Edge AI Research Team
Date: 2024
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class ModelOptimizer:
    """
    Apply various optimization techniques to neural network models.
    
    Supports post-training quantization, structured pruning, and
    combined optimization strategies for edge deployment.
    """
    
    def __init__(self, output_dir: str = "models/optimized"):
        """
        Initialize model optimizer.
        
        Args:
            output_dir: Directory to save optimized models
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("ModelOptimizer initialized with output directory: %s", output_dir)
    
    def apply_quantization(self, model_path: str, output_path: Optional[str] = None,
                          precision: str = "int8",
                          representative_data: Optional[np.ndarray] = None) -> str:
        """
        Apply post-training quantization to convert model to INT8/Float16.
        
        Args:
            model_path: Path to original model (.h5 or .tflite)
            output_path: Path to save quantized model (auto-generated if None)
            precision: Quantization precision ('int8' or 'float16')
            representative_data: Sample data for calibration (optional)
        
        Returns:
            Path to quantized model file
        """
        logger.info("Applying %s quantization to: %s", precision, model_path)
        
        if not Path(model_path).exists():
            logger.error("Model file not found: %s", model_path)
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Generate output path if not provided
        if output_path is None:
            stem = Path(model_path).stem
            output_path = str(self.output_dir / f"{stem}_quantized_{precision}.tflite")
        
        try:
            if model_path.endswith('.tflite'):
                # Already TFLite, apply quantization
                output_path = self._quantize_tflite(
                    model_path, output_path, precision, representative_data
                )
            elif model_path.endswith('.h5'):
                # Keras model, convert and quantize
                output_path = self._quantize_keras(
                    model_path, output_path, precision, representative_data
                )
            else:
                raise ValueError(f"Unsupported model format: {Path(model_path).suffix}")
            
            # Compare sizes
            original_size = Path(model_path).stat().st_size / (1024 ** 2)
            quantized_size = Path(output_path).stat().st_size / (1024 ** 2)
            reduction = (1 - quantized_size / original_size) * 100
            
            logger.info("Quantization complete:")
            logger.info("  Original: %.2f MB", original_size)
            logger.info("  Quantized: %.2f MB", quantized_size)
            logger.info("  Reduction: %.1f%%", reduction)
            
            return output_path
            
        except Exception as e:
            logger.error("Quantization failed: %s", str(e))
            raise
    
    def _quantize_tflite(self, model_path: str, output_path: str,
                        precision: str, rep_data: Optional[np.ndarray]) -> str:
        """Quantize an existing TFLite model."""
        try:
            import tflite_runtime.interpreter as tflite
        except ImportError:
            import tensorflow.lite as tflite
        
        # Load original model
        with open(model_path, 'rb') as f:
            model_content = f.read()
        
        # Configure converter
        converter = tflite.TFLiteConverter.from_saved_model(model_path.replace('.tflite', '_saved_model'))
        
        if precision == "int8":
            converter.optimizations = [tflite.Optimize.DEFAULT]
            
            if rep_data is not None:
                def representative_dataset():
                    for i in range(min(100, len(rep_data))):
                        yield [rep_data[i:i+1].astype(np.float32)]
                
                converter.representative_dataset = representative_dataset
                converter.target_spec.supported_ops = [tflite.OpsSet.TFLITE_BUILTINS_INT8]
                converter.inference_input_type = tf.int8
                converter.inference_output_type = tf.int8
            
            logger.info("Applied INT8 quantization")
        
        elif precision == "float16":
            converter.optimizations = [tflite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
            logger.info("Applied Float16 quantization")
        
        else:
            raise ValueError(f"Unsupported precision: {precision}")
        
        # Convert and save
        quantized_model = converter.convert()
        
        with open(output_path, 'wb') as f:
            f.write(quantized_model)
        
        logger.info("Quantized model saved to: %s", output_path)
        return output_path
    
    def _quantize_keras(self, model_path: str, output_path: str,
                       precision: str, rep_data: Optional[np.ndarray]) -> str:
        """Convert Keras model to quantized TFLite."""
        try:
            import tensorflow as tf
            import tflite_runtime.interpreter as tflite
        except ImportError:
            import tensorflow as tf
            tflite = tf.lite
        
        # Load Keras model
        model = tf.keras.models.load_model(model_path)
        logger.info("Loaded Keras model from: %s", model_path)
        
        # Create converter
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        if precision == "int8":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            if rep_data is not None:
                def representative_dataset():
                    for i in range(min(100, len(rep_data))):
                        yield [rep_data[i:i+1].astype(np.float32)]
                
                converter.representative_dataset = representative_dataset
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            
            logger.info("Applied INT8 quantization to Keras model")
        
        elif precision == "float16":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
            logger.info("Applied Float16 quantization to Keras model")
        
        # Convert
        tflite_model = converter.convert()
        
        # Save
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        logger.info("Quantized TFLite model saved to: %s", output_path)
        return output_path
    
    def apply_pruning(self, model: Any, sparsity: float = 0.5,
                     block_size: Tuple[int, int] = (1, 4),
                     output_path: Optional[str] = None) -> Any:
        """
        Apply structured pruning to reduce model size.
        
        Args:
            model: Keras model to prune
            sparsity: Target sparsity level (0.0 to 1.0, where 0.5 = 50% pruning)
            block_size: Block size for structured pruning
            output_path: Path to save pruned model (optional)
        
        Returns:
            Pruned model
        """
        logger.info("Applying structured pruning with %.1f%% sparsity", sparsity * 100)
        
        try:
            import tensorflow_model_optimization as tfmot
            import tensorflow as tf
        except ImportError:
            logger.error("tensorflow-model-optimization not installed")
            logger.info("Install with: pip install tensorflow-model-optimization")
            raise
        
        try:
            # Apply pruning to entire model
            prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
            
            # Calculate end step (assume 1 epoch for fine-tuning)
            batch_size = 32
            epochs = 1
            num_samples = 1000
            end_step = np.ceil(num_samples / batch_size).astype(np.int32) * epochs
            
            # Pruning parameters
            pruning_params = {
                'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(
                    target_sparsity=sparsity,
                    begin_step=0,
                    end_step=end_step,
                    frequency=100
                ),
                'block_size': block_size
            }
            
            # Apply pruning
            pruned_model = prune_low_magnitude(model, **pruning_params)
            
            # Compile pruned model
            pruned_model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            logger.info("Pruning applied successfully")
            
            # Save if output path provided
            if output_path:
                pruned_model.save(output_path)
                logger.info("Pruned model saved to: %s", output_path)
            
            return pruned_model
            
        except Exception as e:
            logger.error("Pruning failed: %s", str(e))
            raise
    
    def apply_combined(self, model_path: str, output_path: Optional[str] = None,
                      sparsity: float = 0.5, precision: str = "int8") -> str:
        """
        Apply combined optimization: pruning followed by quantization.
        
        Args:
            model_path: Path to original model
            output_path: Path to save final model
            sparsity: Pruning sparsity level
            precision: Quantization precision
        
        Returns:
            Path to optimized model
        """
        logger.info("Applying combined optimization (pruning + quantization)")
        
        try:
            import tensorflow as tf
        except ImportError:
            logger.error("TensorFlow not installed")
            raise
        
        # Step 1: Load model
        logger.info("Step 1: Loading model from %s", model_path)
        if model_path.endswith('.h5'):
            model = tf.keras.models.load_model(model_path)
        else:
            raise ValueError("Combined optimization requires .h5 Keras model")
        
        # Step 2: Apply pruning
        logger.info("Step 2: Applying pruning (%.1f%% sparsity)", sparsity * 100)
        pruned_model_path = str(self.output_dir / "temp_pruned.h5")
        pruned_model = self.apply_pruning(
            model, 
            sparsity=sparsity,
            output_path=pruned_model_path
        )
        
        # Step 3: Apply quantization to pruned model
        logger.info("Step 3: Applying quantization (%s)", precision)
        if output_path is None:
            stem = Path(model_path).stem
            output_path = str(self.output_dir / f"{stem}_combined_{precision}.tflite")
        
        quantized_path = self._quantize_keras(
            pruned_model_path, 
            output_path, 
            precision
        )
        
        # Clean up temporary file
        if Path(pruned_model_path).exists():
            Path(pruned_model_path).unlink()
            logger.debug("Removed temporary pruned model")
        
        # Report results
        original_size = Path(model_path).stat().st_size / (1024 ** 2)
        final_size = Path(quantized_path).stat().st_size / (1024 ** 2)
        reduction = (1 - final_size / original_size) * 100
        
        logger.info("Combined optimization complete:")
        logger.info("  Original: %.2f MB", original_size)
        logger.info("  Optimized: %.2f MB", final_size)
        logger.info("  Total reduction: %.1f%%", reduction)
        
        return quantized_path
    
    def compare_model_sizes(self, model_paths: list) -> Dict[str, float]:
        """
        Compare sizes of multiple model files.
        
        Args:
            model_paths: List of model file paths
        
        Returns:
            Dictionary mapping model names to sizes in MB
        """
        sizes = {}
        
        for path in model_paths:
            name = Path(path).name
            if Path(path).exists():
                size_mb = Path(path).stat().st_size / (1024 ** 2)
                sizes[name] = size_mb
                logger.debug("Model size - %s: %.2f MB", name, size_mb)
            else:
                logger.warning("Model not found: %s", path)
                sizes[name] = 0.0
        
        return sizes
    
    def get_optimization_info(self, optimization_type: str) -> Dict:
        """
        Get information about an optimization technique.
        
        Args:
            optimization_type: Type of optimization
        
        Returns:
            Dictionary with optimization details
        """
        optimizations = {
            'baseline': {
                'description': 'No optimization applied',
                'expected_speedup': '1.0x',
                'expected_size_reduction': '0%'
            },
            'quantization': {
                'description': 'Post-training INT8/Float16 quantization',
                'expected_speedup': '2-4x',
                'expected_size_reduction': '50-75%'
            },
            'pruning': {
                'description': 'Structured pruning with 50% sparsity',
                'expected_speedup': '1.5-2x',
                'expected_size_reduction': '40-60%'
            },
            'combined': {
                'description': 'Pruning followed by quantization',
                'expected_speedup': '3-6x',
                'expected_size_reduction': '70-90%'
            }
        }
        
        if optimization_type not in optimizations:
            raise ValueError(f"Unknown optimization: {optimization_type}")
        
        return optimizations[optimization_type]


def create_optimizer(config: Optional[Dict] = None) -> ModelOptimizer:
    """
    Factory function to create ModelOptimizer from config.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Configured ModelOptimizer instance
    """
    if config is None:
        return ModelOptimizer()
    
    output_dir = config.get('output_dir', 'models/optimized')
    return ModelOptimizer(output_dir=output_dir)


if __name__ == "__main__":
    # Test the model optimizer
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(message)s')
    
    optimizer = ModelOptimizer()
    
    print("=" * 50)
    print("Model Optimizer Test")
    print("=" * 50)
    
    # Display optimization info
    print("\nOptimization Techniques:")
    for opt_type in ['baseline', 'quantization', 'pruning', 'combined']:
        info = optimizer.get_optimization_info(opt_type)
        print(f"\n{opt_type.upper()}:")
        print(f"  Description: {info['description']}")
        print(f"  Expected Speedup: {info['expected_speedup']}")
        print(f"  Expected Size Reduction: {info['expected_size_reduction']}")
    
    print("\n" + "=" * 50)
    print("Note: Actual optimization requires trained model files")
    print("=" * 50)
