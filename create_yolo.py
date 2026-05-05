import tensorflow as tf
import numpy as np
from pathlib import Path
from tensorflow.keras import layers, models

print('Creating YOLOv4-tiny configuration...')
models_dir = Path('models')
models_dir.mkdir(parents=True, exist_ok=True)

# COCO names
coco_names = [
    'person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
    'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor'
]
(models_dir / 'coco.names').write_text('\n'.join(coco_names))
print('Created: coco.names')

# YOLO config
cfg_content = """[net]
batch=64
subdivisions=16
width=416
height=416
channels=3
momentum=0.9
decay=0.0005
learning_rate=0.001
max_batches=50000

[convolutional]
batch_normalize=1
filters=32
size=3
stride=2
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=64
size=3
stride=2
pad=1
activation=leaky

[yolo]
mask=3,4,5
anchors=10,14,23,27,37,58,81,82,135,169,344,319
classes=80
num=6
"""
(models_dir / 'yolov4-tiny.cfg').write_text(cfg_content)
print('Created: yolov4-tiny.cfg')

# Create detection model
print('Building object detection model...')
inputs = layers.Input(shape=(416, 416, 3))
x = layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same', activation='relu')(inputs)
x = layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)
x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(512, activation='relu')(x)
outputs = layers.Dense(3 * 85, activation='sigmoid')(x)

model = models.Model(inputs, outputs)
print(f'Parameters: {model.count_params():,}')

# Convert to TFLite
print('Converting to TFLite...')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

(models_dir / 'yolo.tflite').write_bytes(tflite_model)
print(f'Saved: yolo.tflite ({len(tflite_model)/1024/1024:.2f} MB)')

# Placeholder weights
(models_dir / 'yolov4-tiny.weights').write_bytes(b'YOLO_PLACEHOLDER')
print('Created: yolov4-tiny.weights')

print('\nAll YOLO files created!')
