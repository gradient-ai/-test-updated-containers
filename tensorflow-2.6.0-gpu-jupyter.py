# Test new container on Gradient with Python: tensorflow/tensorflow:2.6.0-gpu-jupyter
#
# Last updated: Oct 21st 2021

# --- Check versions of packages of interest ---

# Run same commands in notebook and .py
# Not running prefixed by "!" as that calls CLI version

import platform
print(platform.python_version())

import subprocess
subprocess.run('pip3 install gradient', shell=True, check=True, stdout=subprocess.PIPE, universal_newlines=True)

import gradient
gradient.version.version

subprocess.run('pip --version', shell=True, check=True, stdout=subprocess.PIPE, universal_newlines=True)

import numpy as np
print(np.__version__)

import tensorflow as tf
print(tf.__version__)

# --- Check GPU ---

subprocess.run('nvidia-smi', shell=True, check=True, stdout=subprocess.PIPE, universal_newlines=True)

tf.config.list_physical_devices()

from tensorflow.python.client import device_lib # Or this, for TF
print(device_lib.list_local_devices())

# --- Check model picks up GPU ---

# Show GPU is being used by running a basic model, e.g., MNIST [1,2]
# The .debugging line adds output showing if the CPU or GPU is being used for each step
# It should default to the GPU when it's available (except for non-GPU operations, but that doesn't include model training)
# This assumes 1 GPU: testing multi-GPU setups is not covered yet
# .evaluate acts as a sanity check that the model ran properly: should be ~98% accuracy

tf.debugging.set_log_device_placement(True)

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5) # Gives long output
model.evaluate(x_test,  y_test, verbose=2)

# [1] https://www.tensorflow.org/guide/gpu
# [2] Subset of lines from https://www.tensorflow.org/tutorials/quickstart/beginner
