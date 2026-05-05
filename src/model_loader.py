"""
Model Loader Module for Edge AI Object Detection

Provides functions to load MobileNetV2, SqueezeNet, and YOLOv4-tiny models
in various formats (TensorFlow Lite, OpenCV DNN, PyTorch).

Author: Edge AI Research Team
Date: 2024
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Load and manage different object detection models.
    
    Supports MobileNetV2, SqueezeNet, and YOLOv4-tiny with
    automatic format detection and fallback mechanisms.
    """
    
    # COCO dataset class labels (80 classes)
    COCO_CLASSES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
        'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
        'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
        'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
        'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
        'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    
    def __init__(self, models_dir: str = "models"):
        """
        Initialize model loader.
        
        Args:
            models_dir: Directory containing model files
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.loaded_models: Dict[str, Any] = {}
        
        logger.info("ModelLoader initialized with directory: %s", models_dir)
    
    def load_mobilenet(self, model_path: Optional[str] = None, 
                       use_tflite: bool = True) -> Any:
        """
        Load MobileNetV2 model for object detection.
        
        Args:
            model_path: Path to model file (auto-detected if None)
            use_tflite: Use TensorFlow Lite format (recommended for edge)
        
        Returns:
            Loaded model object (TFLite interpreter or Keras model)
        
        Raises:
            FileNotFoundError: If model file not found
            ImportError: If required libraries not installed
        """
        model_key = "mobilenet"
        
        # Return cached model if available
        if model_key in self.loaded_models:
            logger.info("Using cached MobileNetV2 model")
            return self.loaded_models[model_key]
        
        # Determine model path
        if model_path is None:
            model_path = str(self.models_dir / "mobilenet_v2.tflite")
        
        logger.info("Loading MobileNetV2 from: %s", model_path)
        
        try:
            if use_tflite and Path(model_path).exists():
                model = self._load_tflite_model(model_path)
                logger.info("MobileNetV2 TFLite model loaded successfully")
            else:
                # Fallback to creating a placeholder model
                logger.warning("TFLite model not found, creating placeholder")
                model = self._create_mobilenet_placeholder()
            
            self.loaded_models[model_key] = model
            return model
            
        except Exception as e:
            logger.error("Failed to load MobileNetV2: %s", str(e))
            raise
    
    def load_squeezenet(self, model_path: Optional[str] = None,
                        use_tflite: bool = True) -> Any:
        """
        Load SqueezeNet model for object detection.
        
        Args:
            model_path: Path to model file (auto-detected if None)
            use_tflite: Use TensorFlow Lite format
        
        Returns:
            Loaded model object
        
        Raises:
            FileNotFoundError: If model file not found
        """
        model_key = "squeezenet"
        
        if model_key in self.loaded_models:
            logger.info("Using cached SqueezeNet model")
            return self.loaded_models[model_key]
        
        if model_path is None:
            model_path = str(self.models_dir / "squeezenet.tflite")
        
        logger.info("Loading SqueezeNet from: %s", model_path)
        
        try:
            if use_tflite and Path(model_path).exists():
                model = self._load_tflite_model(model_path)
                logger.info("SqueezeNet TFLite model loaded successfully")
            else:
                logger.warning("TFLite model not found, creating placeholder")
                model = self._create_squeezenet_placeholder()
            
            self.loaded_models[model_key] = model
            return model
            
        except Exception as e:
            logger.error("Failed to load SqueezeNet: %s", str(e))
            raise
    
    def load_yolo_lightweight(self, 
                               weights_path: Optional[str] = None,
                               config_path: Optional[str] = None,
                               input_size: Tuple[int, int] = (416, 416)) -> Any:
        """
        Load YOLOv4-tiny model using OpenCV DNN module.
        
        Args:
            weights_path: Path to .weights file
            config_path: Path to .cfg file
            input_size: Input resolution (width, height)
        
        Returns:
            OpenCV DNN model object
        """
        model_key = "yolo"
        
        if model_key in self.loaded_models:
            logger.info("Using cached YOLO model")
            return self.loaded_models[model_key]
        
        # Default paths
        if weights_path is None:
            weights_path = str(self.models_dir / "yolov4-tiny.weights")
        if config_path is None:
            config_path = str(self.models_dir / "yolov4-tiny.cfg")
        
        logger.info("Loading YOLOv4-tiny from: %s, %s", weights_path, config_path)
        
        try:
            import cv2
            
            # Check if files exist
            if not Path(weights_path).exists() or not Path(config_path).exists():
                logger.warning("YOLO weights/config not found, creating placeholder")
                model = self._create_yolo_placeholder(input_size)
                self.loaded_models[model_key] = model
                return model
            
            # Load YOLO with OpenCV DNN
            model = cv2.dnn.readNetFromDarknet(config_path, weights_path)
            
            # Set backend based on available hardware
            try:
                model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                logger.info("YOLO using OpenCV DNN CPU backend")
            except Exception as e:
                logger.warning("Failed to set OpenCV DNN preferences: %s", str(e))
            
            logger.info("YOLOv4-tiny loaded successfully")
            self.loaded_models[model_key] = model
            return model
            
        except ImportError:
            logger.error("OpenCV not installed. Install with: pip install opencv-python")
            raise
        except Exception as e:
            logger.warning("Failed to load YOLO, falling back to placeholder: %s", str(e))
            model = self._create_yolo_placeholder(input_size)
            self.loaded_models[model_key] = model
            return model
    
    def _load_tflite_model(self, model_path: str) -> Any:
        """
        Load a TensorFlow Lite model.
        
        Args:
            model_path: Path to .tflite file
        
        Returns:
            TFLite interpreter
        """
        try:
            import tflite_runtime.interpreter as tflite
            logger.info("Using tflite_runtime")
        except ImportError:
            try:
                import tensorflow.lite as tflite
                logger.info("Using tensorflow.lite")
            except ImportError:
                logger.error("Neither tflite_runtime nor tensorflow.lite available")
                raise ImportError("TensorFlow Lite not installed")
        
        # Load model
        interpreter = tflite.Interpreter(model_path=str(model_path))
        interpreter.allocate_tensors()
        
        logger.info("TFLite model loaded: %s", model_path)
        return interpreter
    
    def _create_mobilenet_placeholder(self) -> Dict:
        """
        Create a placeholder MobileNet model for testing.
        
        Returns:
            Dictionary simulating model interface
        """
        logger.warning("Using MobileNetV2 placeholder - results will be simulated")
        return {
            'type': 'mobilenet_v2',
            'input_shape': (1, 224, 224, 3),
            'output_shape': (1, 1000),
            'is_placeholder': True
        }
    
    def _create_squeezenet_placeholder(self) -> Dict:
        """Create a placeholder SqueezeNet model for testing."""
        logger.warning("Using SqueezeNet placeholder - results will be simulated")
        return {
            'type': 'squeezenet',
            'input_shape': (1, 227, 227, 3),
            'output_shape': (1, 1000),
            'is_placeholder': True
        }
    
    def _create_yolo_placeholder(self, input_size: Tuple[int, int]) -> Dict:
        """Create a placeholder YOLO model for testing."""
        logger.warning("Using YOLO placeholder - results will be simulated")
        return {
            'type': 'yolov4_tiny',
            'input_size': input_size,
            'num_classes': 80,
            'is_placeholder': True
        }
    
    def get_model_info(self, model_name: str) -> Dict:
        """
        Get information about a model.
        
        Args:
            model_name: Name of model (mobilenet, squeezenet, yolo)
        
        Returns:
            Dictionary with model metadata
        """
        model_paths = {
            'mobilenet': {
                'tflite': str(self.models_dir / "mobilenet_v2.tflite"),
                'h5': str(self.models_dir / "mobilenet_v2.h5"),
                'input_size': (224, 224),
                'description': 'MobileNetV2 - Depthwise Separable CNN'
            },
            'squeezenet': {
                'tflite': str(self.models_dir / "squeezenet.tflite"),
                'h5': str(self.models_dir / "squeezenet.h5"),
                'input_size': (227, 227),
                'description': 'SqueezeNet - Fire Module Architecture'
            },
            'yolo': {
                'weights': str(self.models_dir / "yolov4-tiny.weights"),
                'config': str(self.models_dir / "yolov4-tiny.cfg"),
                'input_size': (416, 416),
                'description': 'YOLOv4-tiny - Lightweight Real-time Detection'
            }
        }
        
        if model_name not in model_paths:
            raise ValueError(f"Unknown model: {model_name}. Choose from {list(model_paths.keys())}")
        
        info = model_paths[model_name]
        info['loaded'] = model_name in self.loaded_models
        return info
    
    def clear_cache(self):
        """Clear all loaded models from memory."""
        self.loaded_models.clear()
        logger.info("Model cache cleared")
    
    def get_model_size(self, model_path: str) -> float:
        """
        Get model file size in MB.
        
        Args:
            model_path: Path to model file
        
        Returns:
            File size in megabytes
        """
        path = Path(model_path)
        if not path.exists():
            logger.warning("Model file not found: %s", model_path)
            return 0.0
        
        size_bytes = path.stat().st_size
        size_mb = size_bytes / (1024 ** 2)
        logger.debug("Model size: %.2f MB (%s)", size_mb, model_path)
        return size_mb


def create_model_loader(config: Optional[Dict] = None) -> ModelLoader:
    """
    Factory function to create ModelLoader from config.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Configured ModelLoader instance
    """
    if config is None:
        return ModelLoader()
    
    models_dir = config.get('models_dir', 'models')
    return ModelLoader(models_dir=models_dir)


if __name__ == "__main__":
    # Test the model loader
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(message)s')
    
    loader = ModelLoader()
    
    print("=" * 50)
    print("Model Loader Test")
    print("=" * 50)
    
    # Test loading models (will use placeholders if files don't exist)
    try:
        mobilenet = loader.load_mobilenet()
        print(f"✓ MobileNetV2 loaded: {type(mobilenet)}")
    except Exception as e:
        print(f"✗ MobileNetV2 failed: {e}")
    
    try:
        squeezenet = loader.load_squeezenet()
        print(f"✓ SqueezeNet loaded: {type(squeezenet)}")
    except Exception as e:
        print(f"✗ SqueezeNet failed: {e}")
    
    try:
        yolo = loader.load_yolo_lightweight()
        print(f"✓ YOLOv4-tiny loaded: {type(yolo)}")
    except Exception as e:
        print(f"✗ YOLOv4-tiny failed: {e}")
    
    # Display model info
    print("\nModel Information:")
    for model_name in ['mobilenet', 'squeezenet', 'yolo']:
        info = loader.get_model_info(model_name)
        print(f"\n{model_name.upper()}:")
        print(f"  Description: {info['description']}")
        print(f"  Input Size: {info['input_size']}")
        print(f"  Loaded: {info['loaded']}")
    
    print("\n" + "=" * 50)
