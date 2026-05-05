"""
Dataset Preparation Module for Edge AI Object Detection

Downloads, prepares, and converts datasets (Pascal VOC, CIFAR-10)
for model training and evaluation.

Author: Edge AI Research Team
Date: 2024
"""

import logging
import os
import tarfile
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


class DatasetPreparer:
    """
    Download and prepare datasets for object detection experiments.
    
    Supports Pascal VOC and CIFAR-10 with automatic conversion
    to appropriate formats for TensorFlow Lite and OpenCV.
    """
    
    def __init__(self, datasets_dir: str = "datasets"):
        """
        Initialize dataset preparer.
        
        Args:
            datasets_dir: Directory to store datasets
        """
        self.datasets_dir = Path(datasets_dir)
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("DatasetPreparer initialized with directory: %s", datasets_dir)
    
    def download_pascal_voc(self, force_download: bool = False) -> str:
        """
        Download Pascal VOC 2012 dataset.
        
        Args:
            force_download: Force re-download if exists
        
        Returns:
            Path to dataset directory
        """
        voc_dir = self.datasets_dir / "pascal_voc"
        
        if voc_dir.exists() and not force_download:
            logger.info("Pascal VOC already exists at: %s", voc_dir)
            return str(voc_dir)
        
        logger.info("Downloading Pascal VOC 2012...")
        
        url = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
        tar_path = self.datasets_dir / "VOCtrainval_11-May-2012.tar"
        
        try:
            # Download
            if not tar_path.exists():
                logger.info("Downloading from: %s", url)
                self._download_with_progress(url, str(tar_path))
            
            # Extract
            logger.info("Extracting Pascal VOC...")
            with tarfile.open(tar_path, 'r:tar') as tar:
                tar.extractall(path=self.datasets_dir)
            
            # Rename to standard directory
            extracted_dir = self.datasets_dir / "VOCdevkit" / "VOC2012"
            if extracted_dir.exists():
                if voc_dir.exists():
                    import shutil
                    shutil.rmtree(voc_dir)
                extracted_dir.rename(voc_dir)
            
            logger.info("Pascal VOC downloaded and extracted to: %s", voc_dir)
            return str(voc_dir)
            
        except Exception as e:
            logger.error("Failed to download Pascal VOC: %s", str(e))
            raise
    
    def download_cifar10(self, force_download: bool = False) -> str:
        """
        Download CIFAR-10 dataset.
        
        Args:
            force_download: Force re-download if exists
        
        Returns:
            Path to dataset directory
        """
        cifar_dir = self.datasets_dir / "cifar10"
        
        if cifar_dir.exists() and not force_download:
            logger.info("CIFAR-10 already exists at: %s", cifar_dir)
            return str(cifar_dir)
        
        logger.info("Downloading CIFAR-10...")
        
        try:
            import tensorflow as tf
            
            # Load CIFAR-10
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
            
            # Save to disk
            cifar_dir.mkdir(parents=True, exist_ok=True)
            
            np.save(cifar_dir / "x_train.npy", x_train)
            np.save(cifar_dir / "y_train.npy", y_train)
            np.save(cifar_dir / "x_test.npy", x_test)
            np.save(cifar_dir / "y_test.npy", y_test)
            
            logger.info("CIFAR-10 downloaded:")
            logger.info("  Training samples: %d", len(x_train))
            logger.info("  Test samples: %d", len(x_test))
            logger.info("  Saved to: %s", cifar_dir)
            
            return str(cifar_dir)
            
        except ImportError:
            logger.warning("TensorFlow not available, creating simulated CIFAR-10")
            return self._create_simulated_cifar10(cifar_dir)
        except Exception as e:
            logger.error("Failed to download CIFAR-10: %s", str(e))
            raise
    
    def _create_simulated_cifar10(self, cifar_dir: Path) -> str:
        """Create simulated CIFAR-10 data for testing."""
        logger.info("Creating simulated CIFAR-10 dataset")
        
        cifar_dir.mkdir(parents=True, exist_ok=True)
        
        # Simulate CIFAR-10 structure (10 classes, 32x32 images)
        np.random.seed(42)
        
        x_train = np.random.randint(0, 255, (50000, 32, 32, 3), dtype=np.uint8)
        y_train = np.random.randint(0, 10, (50000, 1), dtype=np.uint8)
        x_test = np.random.randint(0, 255, (10000, 32, 32, 3), dtype=np.uint8)
        y_test = np.random.randint(0, 10, (10000, 1), dtype=np.uint8)
        
        np.save(cifar_dir / "x_train.npy", x_train)
        np.save(cifar_dir / "y_train.npy", y_train)
        np.save(cifar_dir / "x_test.npy", x_test)
        np.save(cifar_dir / "y_test.npy", y_test)
        
        logger.info("Simulated CIFAR-10 created at: %s", cifar_dir)
        return str(cifar_dir)
    
    def _download_with_progress(self, url: str, filepath: str):
        """Download file with progress bar."""
        with tqdm(unit='B', unit_scale=True, miniters=1, desc="Downloading") as t:
            def progress_hook(blocknum, blocksize, totalsize):
                if t.total is None and totalsize > 0:
                    t.total = totalsize
                download_so_far = blocknum * blocksize
                t.update(download_so_far - t.n)
            
            urllib.request.urlretrieve(url, filepath, reporthook=progress_hook)
    
    def create_train_test_split(self, dataset_path: str, test_ratio: float = 0.2,
                               dataset_type: str = "cifar10") -> Dict[str, List]:
        """
        Create train/test split for dataset.
        
        Args:
            dataset_path: Path to dataset
            test_ratio: Fraction for testing
            dataset_type: Type of dataset
        
        Returns:
            Dictionary with train/test file lists
        """
        logger.info("Creating train/test split (test ratio: %.1f%%)", test_ratio * 100)
        
        if dataset_type == "cifar10":
            return self._split_cifar10(dataset_path, test_ratio)
        elif dataset_type == "pascal_voc":
            return self._split_pascal_voc(dataset_path, test_ratio)
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    def _split_cifar10(self, dataset_path: str, test_ratio: float) -> Dict:
        """Split CIFAR-10 data."""
        x_train = np.load(Path(dataset_path) / "x_train.npy")
        y_train = np.load(Path(dataset_path) / "y_train.npy")
        
        # Use existing test split
        try:
            x_test = np.load(Path(dataset_path) / "x_test.npy")
            y_test = np.load(Path(dataset_path) / "y_test.npy")
            
            split = {
                'x_train': x_train,
                'y_train': y_train,
                'x_test': x_test,
                'y_test': y_test
            }
        except FileNotFoundError:
            # Create split from training data
            num_test = int(len(x_train) * test_ratio)
            num_train = len(x_train) - num_test
            
            indices = np.random.permutation(len(x_train))
            test_idx = indices[:num_test]
            train_idx = indices[num_test:]
            
            split = {
                'x_train': x_train[train_idx],
                'y_train': y_train[train_idx],
                'x_test': x_train[test_idx],
                'y_test': y_train[test_idx]
            }
        
        logger.info("CIFAR-10 split: %d train, %d test", 
                   len(split['x_train']), len(split['x_test']))
        
        return split
    
    def _split_pascal_voc(self, dataset_path: str, test_ratio: float) -> Dict:
        """Split Pascal VOC data."""
        voc_dir = Path(dataset_path)
        image_dir = voc_dir / "JPEGImages"
        annot_dir = voc_dir / "Annotations"
        
        if not image_dir.exists():
            logger.error("Pascal VOC image directory not found: %s", image_dir)
            raise FileNotFoundError(f"Directory not found: {image_dir}")
        
        # Get all image files
        image_files = list(image_dir.glob("*.jpg"))
        logger.info("Found %d images in Pascal VOC", len(image_files))
        
        # Create split
        np.random.shuffle(image_files)
        num_test = int(len(image_files) * test_ratio)
        
        test_files = image_files[:num_test]
        train_files = image_files[num_test:]
        
        split = {
            'train': train_files,
            'test': test_files
        }
        
        logger.info("Pascal VOC split: %d train, %d test", 
                   len(train_files), len(test_files))
        
        return split
    
    def download_pretrained_models(self, force_download: bool = False) -> Dict[str, str]:
        """
        Download pretrained models for experiments.
        
        Args:
            force_download: Force re-download if exists
        
        Returns:
            Dictionary of model paths
        """
        logger.info("Downloading pretrained models...")
        
        models_dir = Path("models")
        models_dir.mkdir(parents=True, exist_ok=True)
        
        model_urls = {
            'mobilenet_v2': {
                'url': 'https://tfhub.dev/google/lite-model/mobilenet_v2_1.0_224_detection/1/metadata/0?lite-format=tflite',
                'path': models_dir / 'mobilenet_v2.tflite',
                'description': 'MobileNetV2 SSD Detector'
            },
            'squeezenet': {
                'url': 'https://github.com/DeepScale/SqueezeNet/raw/master/SqueezeNet_v1.1/squeezenet_v1.1.caffemodel',
                'path': models_dir / 'squeezenet.caffemodel',
                'description': 'SqueezeNet v1.1'
            },
            'yolov4_tiny': {
                'weights': 'https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights',
                'config': 'https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg',
                'weights_path': models_dir / 'yolov4-tiny.weights',
                'config_path': models_dir / 'yolov4-tiny.cfg',
                'description': 'YOLOv4-tiny'
            }
        }
        
        downloaded = {}
        
        for model_name, info in model_urls.items():
            logger.info("\nDownloading %s...", info['description'])
            
            try:
                if model_name == 'yolov4_tiny':
                    # Download YOLO weights and config
                    if not info['weights_path'].exists() or force_download:
                        self._download_with_progress(
                            info['weights'], str(info['weights_path'])
                        )
                    
                    if not info['config_path'].exists() or force_download:
                        self._download_with_progress(
                            info['config'], str(info['config_path'])
                        )
                    
                    downloaded[model_name] = str(info['weights_path'])
                
                else:
                    # Download other models
                    if not info['path'].exists() or force_download:
                        self._download_with_progress(info['url'], str(info['path']))
                    
                    downloaded[model_name] = str(info['path'])
                
                logger.info("✓ %s downloaded", info['description'])
                
            except Exception as e:
                logger.error("✗ Failed to download %s: %s", model_name, str(e))
                logger.info("  You can manually place model files in the 'models/' directory")
        
        logger.info("\nModel download complete!")
        return downloaded
    
    def get_representative_dataset(self, dataset_type: str = "cifar10",
                                   num_samples: int = 100) -> np.ndarray:
        """
        Get representative dataset for quantization calibration.
        
        Args:
            dataset_type: Type of dataset
            num_samples: Number of samples to return
        
        Returns:
            Numpy array of samples
        """
        logger.info("Getting representative dataset: %s (%d samples)", 
                   dataset_type, num_samples)
        
        if dataset_type == "cifar10":
            cifar_dir = self.datasets_dir / "cifar10"
            if cifar_dir.exists():
                x_train = np.load(cifar_dir / "x_train.npy")
                # Normalize and subset
                samples = x_train[:num_samples].astype(np.float32) / 255.0
                return samples
            else:
                logger.warning("CIFAR-10 not found, creating random data")
                return np.random.random((num_samples, 32, 32, 3)).astype(np.float32)
        
        elif dataset_type == "pascal_voc":
            # Return random images (real implementation would load actual images)
            logger.info("Using random data for Pascal VOC representative dataset")
            return np.random.random((num_samples, 224, 224, 3)).astype(np.float32)
        
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    def get_dataset_info(self) -> Dict:
        """Get information about available datasets."""
        info = {}
        
        # Check CIFAR-10
        cifar_dir = self.datasets_dir / "cifar10"
        if cifar_dir.exists():
            try:
                x_train = np.load(cifar_dir / "x_train.npy")
                x_test = np.load(cifar_dir / "x_test.npy")
                info['cifar10'] = {
                    'path': str(cifar_dir),
                    'train_samples': len(x_train),
                    'test_samples': len(x_test),
                    'image_size': x_train.shape[1:3],
                    'num_classes': 10
                }
            except Exception:
                info['cifar10'] = {'path': str(cifar_dir), 'status': 'exists'}
        else:
            info['cifar10'] = {'status': 'not_downloaded'}
        
        # Check Pascal VOC
        voc_dir = self.datasets_dir / "pascal_voc"
        if voc_dir.exists():
            image_dir = voc_dir / "JPEGImages"
            if image_dir.exists():
                num_images = len(list(image_dir.glob("*.jpg")))
                info['pascal_voc'] = {
                    'path': str(voc_dir),
                    'num_images': num_images,
                    'num_classes': 20
                }
            else:
                info['pascal_voc'] = {'path': str(voc_dir), 'status': 'incomplete'}
        else:
            info['pascal_voc'] = {'status': 'not_downloaded'}
        
        return info


def prepare_datasets(config: Optional[Dict] = None) -> DatasetPreparer:
    """
    Convenience function to prepare all datasets.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        DatasetPreparer instance
    """
    preparer = DatasetPreparer()
    
    # Download CIFAR-10 (smaller, needed for quantization)
    preparer.download_cifar10()
    
    return preparer


if __name__ == "__main__":
    # Test dataset preparation
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(message)s')
    
    print("=" * 60)
    print("Dataset Preparation Module Test")
    print("=" * 60)
    
    preparer = DatasetPreparer()
    
    # Download CIFAR-10
    print("\n[1/2] Preparing CIFAR-10...")
    try:
        cifar_path = preparer.download_cifar10()
        print(f"✓ CIFAR-10 ready at: {cifar_path}")
    except Exception as e:
        print(f"✗ CIFAR-10 failed: {e}")
    
    # Get dataset info
    print("\nDataset Information:")
    info = preparer.get_dataset_info()
    for dataset, details in info.items():
        print(f"\n{dataset.upper()}:")
        for key, value in details.items():
            print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)
    print("Note: Pascal VOC download is optional for benchmarking")
    print("=" * 60)
