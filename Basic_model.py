# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 14:47:33 2021

@author: ss38
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
import joblib
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from keras import regularizers
directory = r"C:\Users\ss38\Rafa AI stuff\210104_trimmed\freq_29.00_ampl_phase"

classes = ['1brass','1brass_on_ferrite','1co_on_ferrite','1copper','1ferrite','1ferrite_on_co','1steel','2co_on_ferrite','no_plates']
batch_size = 32
img_height = 180
img_width = 180

train_ds = tf.keras.preprocessing.image_dataset_from_directory(directory,validation_split = 0.2,subset = "training",seed = 123, image_size = (180,180))
validation_ds = tf.keras.preprocessing.image_dataset_from_directory(directory,validation_split = 0.2,subset = "validation",seed = 123, image_size = (180,180))
class_names = train_ds.class_names


AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = validation_ds.cache().prefetch(buffer_size=AUTOTUNE) 
normalization_layer = layers.Rescaling(1./255)
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))

num_classes = len(classes)

model = Sequential([
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()
epochs=10

history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)


# filename = '0degrees_model.sav'
# joblib.dump(model,filename)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)
tf.keras.models.save_model(model, 'basic_model_29.00')

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

