"""
Camera Capture Module for Edge AI Object Detection

Handles Pi Camera and USB webcam initialization, frame capture,
and image preprocessing for object detection inference.

Author: Edge AI Research Team
Date: 2024
"""

import logging
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CameraCapture:
    """
    Camera handler for real-time object detection.
    
    Supports both Raspberry Pi Camera Module and USB webcams
    with automatic fallback and error handling.
    """
    
    def __init__(self, camera_type: str = "usb", resolution: Tuple[int, int] = (640, 480),
                 fps: int = 30, save_dir: str = "output/images"):
        """
        Initialize camera capture.
        
        Args:
            camera_type: Type of camera ('picamera' or 'usb')
            resolution: Frame resolution (width, height)
            fps: Target frames per second
            save_dir: Directory to save captured images
        """
        self.camera_type = camera_type
        self.resolution = resolution
        self.target_fps = fps
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.camera = None
        self.is_running = False
        self.frame_count = 0
        
        logger.info("CameraCapture initialized: %s, %dx%d @ %dfps",
                   camera_type, resolution[0], resolution[1], fps)
    
    def initialize(self) -> bool:
        """
        Initialize camera device.
        
        Returns:
            True if initialization successful, False otherwise
        """
        logger.info("Initializing %s camera...", self.camera_type)
        
        try:
            if self.camera_type == "picamera":
                return self._initialize_picamera()
            else:
                return self._initialize_usb_camera()
                
        except Exception as e:
            logger.error("Camera initialization failed: %s", str(e))
            logger.info("Attempting fallback to USB camera...")
            
            # Fallback to USB camera
            if self.camera_type != "usb":
                self.camera_type = "usb"
                return self._initialize_usb_camera()
            
            return False
    
    def _initialize_picamera(self) -> bool:
        """Initialize Raspberry Pi Camera Module."""
        try:
            # Try picamera2 (newer Raspberry Pi OS)
            try:
                from picamera2 import Picamera2
                self.camera = Picamera2()
                
                config = self.camera.create_preview_configuration(
                    main={"size": self.resolution, "format": "RGB888"}
                )
                self.camera.configure(config)
                self.camera.start()
                
                logger.info("Pi Camera v2 initialized via picamera2")
                self.is_running = True
                return True
                
            except ImportError:
                # Fallback to older picamera
                import picamera
                self.camera = picamera.PiCamera()
                self.camera.resolution = self.resolution
                self.camera.framerate = self.target_fps
                
                # Warm up
                time.sleep(2)
                
                logger.info("Pi Camera initialized via picamera (legacy)")
                self.is_running = True
                return True
                
        except Exception as e:
            logger.error("Pi Camera initialization failed: %s", str(e))
            return False
    
    def _initialize_usb_camera(self) -> bool:
        """Initialize USB webcam using OpenCV."""
        try:
            # Try common device indices
            for device_idx in [0, 1, 2]:
                logger.debug("Trying USB camera at index %d...", device_idx)
                cap = cv2.VideoCapture(device_idx)
                
                if cap.isOpened():
                    # Set resolution and FPS
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
                    cap.set(cv2.CAP_PROP_FPS, self.target_fps)
                    
                    self.camera = cap
                    self.is_running = True
                    
                    logger.info("USB camera initialized at index %d", device_idx)
                    return True
            
            logger.error("No USB camera found on indices 0-2")
            return False
            
        except Exception as e:
            logger.error("USB camera initialization failed: %s", str(e))
            return False
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """
        Capture a single frame from camera.
        
        Returns:
            Frame as numpy array (RGB format), or None if failed
        """
        if not self.is_running or self.camera is None:
            logger.error("Camera not initialized")
            return None
        
        try:
            if self.camera_type == "picamera":
                return self._capture_picamera_frame()
            else:
                return self._capture_usb_frame()
                
        except Exception as e:
            logger.error("Frame capture failed: %s", str(e))
            return None
    
    def _capture_picamera_frame(self) -> Optional[np.ndarray]:
        """Capture frame from Pi Camera."""
        try:
            # picamera2
            if hasattr(self.camera, 'capture_array'):
                frame = self.camera.capture_array()
                return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Older picamera
            import io
            import picamera
            
            stream = io.BytesIO()
            self.camera.capture(stream, format='jpeg', use_video_port=True)
            nparr = np.frombuffer(stream.getvalue(), np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            self.frame_count += 1
            return frame
            
        except Exception as e:
            logger.error("Pi Camera frame capture failed: %s", str(e))
            return None
    
    def _capture_usb_frame(self) -> Optional[np.ndarray]:
        """Capture frame from USB webcam."""
        try:
            ret, frame = self.camera.read()
            
            if not ret:
                logger.warning("Failed to read frame from USB camera")
                return None
            
            self.frame_count += 1
            return frame
            
        except Exception as e:
            logger.error("USB camera frame capture failed: %s", str(e))
            return None
    
    def preprocess_frame(self, frame: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """
        Preprocess frame for model inference.
        
        Args:
            frame: Input frame (BGR format)
            target_size: Target size (width, height)
        
        Returns:
            Preprocessed frame
        """
        try:
            # Resize
            resized = cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)
            
            # Convert BGR to RGB
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            
            # Normalize to [0, 1]
            normalized = rgb.astype(np.float32) / 255.0
            
            return normalized
            
        except Exception as e:
            logger.error("Frame preprocessing failed: %s", str(e))
            raise
    
    def save_frame(self, frame: np.ndarray, prefix: str = "frame") -> str:
        """
        Save frame to disk with timestamp.
        
        Args:
            frame: Frame to save (BGR format)
            prefix: Filename prefix
        
        Returns:
            Path to saved file
        """
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"{prefix}_{timestamp}_{self.frame_count}.jpg"
            filepath = self.save_dir / filename
            
            cv2.imwrite(str(filepath), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            logger.debug("Frame saved: %s", filepath)
            return str(filepath)
            
        except Exception as e:
            logger.error("Failed to save frame: %s", str(e))
            raise
    
    def capture_and_save(self, target_size: Optional[Tuple[int, int]] = None) -> Optional[str]:
        """
        Capture frame and save to disk.
        
        Args:
            target_size: Optional resize target
        
        Returns:
            Path to saved image, or None if failed
        """
        frame = self.capture_frame()
        
        if frame is None:
            return None
        
        # Resize if needed
        if target_size:
            frame = cv2.resize(frame, target_size)
        
        return self.save_frame(frame)
    
    def release(self):
        """Release camera resources."""
        if self.camera is not None:
            try:
                if self.camera_type == "picamera":
                    if hasattr(self.camera, 'stop'):
                        self.camera.stop()
                    elif hasattr(self.camera, 'close'):
                        self.camera.close()
                else:
                    self.camera.release()
                
                logger.info("Camera released")
                
            except Exception as e:
                logger.error("Error releasing camera: %s", str(e))
            finally:
                self.camera = None
                self.is_running = False
    
    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
    
    def __del__(self):
        """Destructor."""
        self.release()


def create_camera(config: dict) -> CameraCapture:
    """
    Factory function to create CameraCapture from config.
    
    Args:
        config: Configuration dictionary with 'camera_type', 'resolution', etc.
    
    Returns:
        Configured CameraCapture instance
    """
    camera_type = config.get('camera_type', 'usb')
    resolution = tuple(config.get('resolution', [640, 480]))
    fps = config.get('fps', 30)
    save_dir = config.get('save_dir', 'output/images')
    
    return CameraCapture(
        camera_type=camera_type,
        resolution=resolution,
        fps=fps,
        save_dir=save_dir
    )


if __name__ == "__main__":
    # Test camera capture
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(message)s')
    
    print("=" * 50)
    print("Camera Capture Test")
    print("=" * 50)
    
    # Try USB camera first
    camera = CameraCapture(camera_type="usb", resolution=(640, 480))
    
    if camera.initialize():
        print("✓ Camera initialized")
        
        # Capture 5 frames
        for i in range(5):
            frame = camera.capture_frame()
            if frame is not None:
                print(f"✓ Frame {i+1} captured: {frame.shape}")
                time.sleep(0.5)
            else:
                print(f"✗ Frame {i+1} failed")
        
        camera.release()
        print("✓ Camera released")
    else:
        print("✗ Camera initialization failed")
        print("Note: This is expected if no camera is connected")
    
    print("=" * 50)
