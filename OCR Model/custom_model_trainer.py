# -*- coding: utf-8 -*-
"""
Created on Mon Sep  8 13:06:24 2025

@author: barna
"""

import os
import cv2 as cv
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping

data_dir = r"C:\Users\barna\Documents\ziply-solver\OCR Model\train"
save_loc = r"C:\Users\barna\Documents\ziply-solver\OCR Model"
img_size = 28

X = []
y = []

for digit_folder in os.listdir(data_dir):
    digit_path = os.path.join(data_dir, digit_folder)
    if not os.path.isdir(digit_path):
        continue
    label = int(digit_folder)
    
    for img_file in os.listdir(digit_path):
        img_path = os.path.join(digit_path, img_file)
        img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv.resize(img, (img_size, img_size))
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=-1)
        X.append(img)
        y.append(label-1)
        
X = np.array(X)
y = np.array(y)
num_classes = 18
# Convert y to categorical via one-hot encoding
y = to_categorical(y, num_classes=num_classes)

# Split data into two groups: training and test
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.1, random_state = 42)

# Define the model
model = Sequential([
    Conv2D(32, (3, 3), activation = 'relu', input_shape = (28,28,1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation = 'relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation = 'relu'),
    Dense(num_classes, activation = 'softmax')
    ])

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
#model.summary()

# Stop training once validation accuracy reaches 100%
early_stop = EarlyStopping(
    monitor='val_accuracy',   # metric to monitor
    patience=0,               # how many epochs to wait after last improvement
    verbose=1,
    mode='max',
    baseline=1.0              # stop if val_accuracy >= 1.0
)

history = model.fit(
    X_train, y_train,
    epochs = 5,
    batch_size = 16,
    validation_data = (X_val, y_val)
    )

model.save(f"{save_loc}/mnist_custom_digits01.keras")