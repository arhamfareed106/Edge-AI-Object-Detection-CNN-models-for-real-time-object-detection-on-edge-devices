import tensorflow as tf
import numpy as np
from pathlib import Path
from tensorflow.keras import layers, models

print('Loading CIFAR-10...')
x_train = np.load('datasets/cifar10/x_train.npy')
y_train = np.load('datasets/cifar10/y_train.npy')
x_test = np.load('datasets/cifar10/x_test.npy')
y_test = np.load('datasets/cifar10/y_test.npy')

x_train_norm = x_train[:10000].astype('float32')/255.0
y_train_sub = y_train[:10000]
x_test_norm = x_test[:2000].astype('float32')/255.0
y_test_sub = y_test[:2000]

print('Building SqueezeNet-like model...')

def fire_module(x, sq, exp):
    s = layers.Conv2D(sq, (1,1), activation='relu', padding='same')(x)
    e1 = layers.Conv2D(exp, (1,1), activation='relu', padding='same')(s)
    e3 = layers.Conv2D(exp, (3,3), activation='relu', padding='same')(s)
    return layers.Concatenate()([e1, e3])

inputs = layers.Input(shape=(32,32,3))
x = layers.Conv2D(96, (7,7), strides=(2,2), activation='relu', padding='same')(inputs)
x = layers.MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(x)
x = fire_module(x, 16, 64)
x = fire_module(x, 16, 64)
x = layers.MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(x)
x = fire_module(x, 32, 128)
x = fire_module(x, 32, 128)
x = layers.GlobalAveragePooling2D()(x)
outputs = layers.Dense(10, activation='softmax')(x)

model = models.Model(inputs, outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print(f'Parameters: {model.count_params():,}')

print('Training 3 epochs...')
model.fit(x_train_norm, y_train_sub, validation_data=(x_test_norm, y_test_sub), epochs=3, batch_size=64)

print('Converting to TFLite...')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

Path('models/squeezenet.tflite').write_bytes(tflite_model)
print(f'Saved: squeezenet.tflite ({len(tflite_model)/1024/1024:.2f} MB)')
