"""
Inference with Bounding Boxes for Edge AI Object Detection

Runs object detection on images and draws bounding boxes with labels
and confidence scores. Supports TFLite, OpenCV DNN, and placeholder models.

Author: Edge AI Research Team
Date: 2024
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from model_loader import ModelLoader

logger = logging.getLogger(__name__)


class InferenceWithBoxes:
    """
    Object detection with bounding box visualization.
    
    Supports multiple model types and provides methods to run inference,
    parse detections, and draw annotated results.
    """
    
    # Default colors for different classes (BGR format)
    COLORS = np.random.uniform(0, 255, size=(80, 3))
    
    def __init__(self, model: Any, model_type: str = "tflite",
                 confidence_threshold: float = 0.5, nms_threshold: float = 0.4,
                 input_size: Tuple[int, int] = (224, 224)):
        """
        Initialize inference engine.
        
        Args:
            model: Loaded model
            model_type: Type of model ('tflite', 'opencv', 'placeholder')
            confidence_threshold: Minimum confidence to keep detection
            nms_threshold: Non-maximum suppression threshold
            input_size: Model input size (width, height)
        """
        self.model = model
        self.model_type = model_type
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size
        
        self.class_labels = ModelLoader.COCO_CLASSES
        
        logger.info("InferenceWithBoxes initialized: %s, threshold=%.2f",
                   model_type, confidence_threshold)
    
    def run_inference(self, image: np.ndarray) -> Tuple[List[Dict], float]:
        """
        Run object detection on image.
        
        Args:
            image: Input image (BGR format)
        
        Returns:
            Tuple of (detections, inference_time_ms)
            detections: List of dicts with 'class', 'confidence', 'bbox' keys
        """
        start_time = time.perf_counter()
        
        try:
            if self.model_type == "placeholder":
                detections = self._simulate_detection(image)
            elif self.model_type == "tflite":
                detections = self._run_tflite_inference(image)
            elif self.model_type == "opencv":
                detections = self._run_opencv_inference(image)
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
            
            inference_time_ms = (time.perf_counter() - start_time) * 1000
            
            logger.debug("Detected %d objects in %.1f ms", len(detections), inference_time_ms)
            return detections, inference_time_ms
            
        except Exception as e:
            logger.error("Inference failed: %s", str(e))
            raise
    
    def _run_tflite_inference(self, image: np.ndarray) -> List[Dict]:
        """Run inference with TFLite model."""
        # Preprocess
        input_image = cv2.resize(image, self.input_size)
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        input_image = input_image.astype(np.float32) / 255.0
        input_image = np.expand_dims(input_image, axis=0)
        
        # Get input/output details
        input_details = self.model.get_input_details()
        output_details = self.model.get_output_details()
        
        # Set input
        self.model.set_tensor(input_details[0]['index'], input_image)
        
        # Run inference
        self.model.invoke()
        
        # Get outputs
        # Output format depends on model - assuming SSD MobileNet format
        boxes = self.model.get_tensor(output_details[0]['index'])  # [1, N, 4]
        classes = self.model.get_tensor(output_details[1]['index'])  # [1, N]
        scores = self.model.get_tensor(output_details[2]['index'])  # [1, N]
        num_detections = int(self.model.get_tensor(output_details[3]['index']))  # [1]
        
        # Parse detections
        detections = []
        h_orig, w_orig = image.shape[:2]
        
        for i in range(int(num_detections[0])):
            confidence = scores[0][i]
            
            if confidence < self.confidence_threshold:
                continue
            
            # Get bounding box (normalized coordinates)
            ymin, xmin, ymax, xmax = boxes[0][i]
            
            # Convert to pixel coordinates
            x_min = int(xmin * w_orig)
            y_min = int(ymin * h_orig)
            x_max = int(xmax * w_orig)
            y_max = int(ymax * h_orig)
            
            class_id = int(classes[0][i])
            class_name = self.class_labels[class_id] if class_id < len(self.class_labels) else f"class_{class_id}"
            
            detections.append({
                'class': class_name,
                'class_id': class_id,
                'confidence': float(confidence),
                'bbox': (x_min, y_min, x_max, y_max)
            })
        
        return detections
    
    def _run_opencv_inference(self, image: np.ndarray) -> List[Dict]:
        """Run inference with OpenCV DNN model (YOLO)."""
        # Preprocess for YOLO
        blob = cv2.dnn.blobFromImage(
            image, 
            scalefactor=1/255.0, 
            size=self.input_size,
            swapRB=True, 
            crop=False
        )
        
        # Run inference
        self.model.setInput(blob)
        
        # Get output layer names
        layer_names = self.model.getLayerNames()
        output_layers = [layer_names[i - 1] for i in self.model.getUnconnectedOutLayers()]
        
        # Forward pass
        outputs = self.model.forward(output_layers)
        
        # Parse detections
        detections = []
        h_orig, w_orig = image.shape[:2]
        
        for output in outputs:
            for detection in output:
                # Get class scores
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence < self.confidence_threshold:
                    continue
                
                # Get bounding box
                center_x = int(detection[0] * w_orig)
                center_y = int(detection[1] * h_orig)
                width = int(detection[2] * w_orig)
                height = int(detection[3] * h_orig)
                
                x_min = center_x - width // 2
                y_min = center_y - height // 2
                x_max = center_x + width // 2
                y_max = center_y + height // 2
                
                class_name = self.class_labels[class_id] if class_id < len(self.class_labels) else f"class_{class_id}"
                
                detections.append({
                    'class': class_name,
                    'class_id': class_id,
                    'confidence': float(confidence),
                    'bbox': (x_min, y_min, x_max, y_max)
                })
        
        # Apply NMS
        detections = self._apply_nms(detections)
        
        return detections
    
    def _simulate_detection(self, image: np.ndarray) -> List[Dict]:
        """Simulate detection for placeholder models."""
        h, w = image.shape[:2]
        
        # Generate some random detections for testing
        np.random.seed(int(time.time()) % 1000)
        num_detections = np.random.randint(1, 4)
        
        detections = []
        for _ in range(num_detections):
            class_id = np.random.randint(0, min(10, len(self.class_labels)))
            confidence = np.random.uniform(0.6, 0.95)
            
            # Random bbox
            x_min = np.random.randint(0, w // 2)
            y_min = np.random.randint(0, h // 2)
            x_max = np.random.randint(x_min + 50, w)
            y_max = np.random.randint(y_min + 50, h)
            
            detections.append({
                'class': self.class_labels[class_id],
                'class_id': class_id,
                'confidence': float(confidence),
                'bbox': (x_min, y_min, x_max, y_max)
            })
        
        logger.info("Simulated %d detections", num_detections)
        return detections
    
    def _apply_nms(self, detections: List[Dict]) -> List[Dict]:
        """
        Apply non-maximum suppression to remove duplicate detections.
        
        Args:
            detections: List of detection dicts
        
        Returns:
            Filtered detections
        """
        if len(detections) == 0:
            return []
        
        # Extract boxes and confidences
        boxes = np.array([d['bbox'] for d in detections], dtype=np.float32)
        confidences = np.array([d['confidence'] for d in detections])
        
        # Apply NMS
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(),
            confidences.tolist(),
            self.confidence_threshold,
            self.nms_threshold
        )
        
        # Filter detections
        if len(indices) > 0:
            indices = indices.flatten() if len(indices.shape) > 1 else indices
            filtered = [detections[i] for i in indices]
        else:
            filtered = []
        
        return filtered
    
    def draw_detections(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Draw bounding boxes and labels on image.
        
        Args:
            image: Input image (BGR)
            detections: List of detection dicts
        
        Returns:
            Annotated image
        """
        annotated = image.copy()
        
        for det in detections:
            x_min, y_min, x_max, y_max = det['bbox']
            class_name = det['class']
            confidence = det['confidence']
            
            # Get color for this class
            color = self.COLORS[det.get('class_id', 0) % len(self.COLORS)].tolist()
            
            # Draw bounding box
            cv2.rectangle(annotated, (x_min, y_min), (x_max, y_max), color, 2)
            
            # Draw label background
            label = f"{class_name}: {confidence:.2f}"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            
            label_y_min = y_min - 10 if y_min - 10 > 10 else y_min + 10
            label_y_max = label_y_min + label_h + 5
            
            cv2.rectangle(
                annotated,
                (x_min, label_y_min - label_h - 5),
                (x_min + label_w, label_y_min),
                color,
                -1
            )
            
            # Draw label text
            text_y = label_y_min - 5 if y_min - 10 > 10 else y_min + label_h + 5
            cv2.putText(
                annotated,
                label,
                (x_min, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )
        
        return annotated
    
    def process_image(self, image_path: str, output_path: Optional[str] = None) -> Dict:
        """
        Process a single image: detect, draw, and save.
        
        Args:
            image_path: Path to input image
            output_path: Path to save output (auto-generated if None)
        
        Returns:
            Dictionary with results
        """
        logger.info("Processing image: %s", image_path)
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            logger.error("Failed to load image: %s", image_path)
            raise FileNotFoundError(f"Cannot load image: {image_path}")
        
        # Run inference
        detections, inference_time = self.run_inference(image)
        
        # Draw detections
        annotated = self.draw_detections(image, detections)
        
        # Save output
        if output_path is None:
            output_path = str(Path("output/images") / f"detected_{Path(image_path).name}")
        
        cv2.imwrite(output_path, annotated, [cv2.IMWRITE_JPEG_QUALITY, 95])
        logger.info("Saved annotated image to: %s", output_path)
        
        results = {
            'input_path': image_path,
            'output_path': output_path,
            'num_detections': len(detections),
            'detections': detections,
            'inference_time_ms': inference_time,
            'image_size': image.shape[:2]
        }
        
        return results
    
    def process_video_stream(self, duration_seconds: float = 10.0,
                            save_output: bool = False) -> Dict:
        """
        Process video stream from camera.
        
        Args:
            duration_seconds: How long to run
            save_output: Whether to save frames
        
        Returns:
            Dictionary with streaming results
        """
        from camera_capture import CameraCapture
        
        logger.info("Starting video stream for %.1f seconds...", duration_seconds)
        
        camera = CameraCapture(resolution=self.input_size)
        if not camera.initialize():
            logger.error("Failed to initialize camera")
            return {'error': 'Camera initialization failed'}
        
        frame_count = 0
        total_inference_time = 0.0
        start_time = time.time()
        
        try:
            while time.time() - start_time < duration_seconds:
                # Capture frame
                frame = camera.capture_frame()
                if frame is None:
                    continue
                
                # Run inference
                detections, inf_time = self.run_inference(frame)
                
                # Draw detections
                annotated = self.draw_detections(frame, detections)
                
                # Display
                cv2.imshow("Object Detection", annotated)
                
                # Save if requested
                if save_output and frame_count % 30 == 0:
                    camera.save_frame(annotated, prefix="detection")
                
                frame_count += 1
                total_inference_time += inf_time
                
                # Exit on 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("Stream stopped by user")
                    break
            
        except KeyboardInterrupt:
            logger.info("Stream interrupted")
        finally:
            camera.release()
            cv2.destroyAllWindows()
        
        elapsed = time.time() - start_time
        avg_fps = frame_count / elapsed if elapsed > 0 else 0
        avg_inf_time = total_inference_time / frame_count if frame_count > 0 else 0
        
        results = {
            'frames_processed': frame_count,
            'duration_seconds': elapsed,
            'avg_fps': avg_fps,
            'avg_inference_time_ms': avg_inf_time
        }
        
        logger.info("Video stream complete: %d frames, %.1f fps", frame_count, avg_fps)
        return results


def create_inference(model: Any, config: Dict) -> InferenceWithBoxes:
    """
    Factory function to create InferenceWithBoxes from config.
    
    Args:
        model: Loaded model
        config: Configuration dictionary
    
    Returns:
        Configured InferenceWithBoxes instance
    """
    model_type = config.get('model_type', 'tflite')
    confidence_threshold = config.get('confidence_threshold', 0.5)
    nms_threshold = config.get('nms_threshold', 0.4)
    input_size = tuple(config.get('input_size', [224, 224]))
    
    return InferenceWithBoxes(
        model=model,
        model_type=model_type,
        confidence_threshold=confidence_threshold,
        nms_threshold=nms_threshold,
        input_size=input_size
    )


if __name__ == "__main__":
    # Test inference with placeholder
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(message)s')
    
    print("=" * 50)
    print("Inference with Bounding Boxes Test")
    print("=" * 50)
    
    # Create placeholder model
    placeholder_model = {
        'type': 'mobilenet_v2',
        'is_placeholder': True
    }
    
    inference = InferenceWithBoxes(
        model=placeholder_model,
        model_type="placeholder",
        confidence_threshold=0.5,
        input_size=(224, 224)
    )
    
    # Create test image
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Run inference
    detections, inf_time = inference.run_inference(test_image)
    
    print(f"\nInference time: {inf_time:.1f} ms")
    print(f"Detections: {len(detections)}")
    
    for i, det in enumerate(detections):
        print(f"  {i+1}. {det['class']}: {det['confidence']:.2f} at {det['bbox']}")
    
    # Draw and save
    annotated = inference.draw_detections(test_image, detections)
    output_path = "output/images/test_detection.jpg"
    cv2.imwrite(output_path, annotated)
    print(f"\nSaved annotated image to: {output_path}")
    
    print("=" * 50)
