# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 16:55:23 2021

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
directory = r'C:\Users\ss38\Rafa AI stuff\210104_trimmed\freq_29.00_ampl_phase'

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
  layers.Dense(3, activation='softmax')

])
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics='acc')
epochs = 10
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)


# TEST

Y = y_test
Y_AI = model.predict(x_test)
test_loss, test_acc = model.evaluate(x_test, Y)
print('LOSS = '+ str(test_loss) + '      ACC = ' + str(test_acc))
fig, ax1 = plt.subplots(3)
for i in range(3):
    ax1[i].plot(range(len(Y)), Y[:, i], 'ok', range(len(Y)), Y_AI[:, i], 'xr')
plt.show()
Y_AI_classes = np.argmax(Y_AI,axis = 1)
Y_true = np.argmax(Y,axis=1)


samples = ['1brass',

 '1copper',
 '1co_on_ferrite',
 '1ferrite',
 '1ferrite_on_co',
 
 'no_plates']

colors = ['r', 'b', 'g', 'y', 'm', 'c', 'orange', 'pink', 'grey']
symbols = ['s', 'o', '*', 'D', '^', 'x', '>', '<', 'v']
names = ['brass',  'copper', 'copper on ferrite', 'ferrite', 'ferrite on copper',  'no samples']

####################################### Organising data and predicting


data_train = []
data_val = []
data_test = []

r_train = 0
r_val = 0
r_test = 1

#Applying model to unknown data

data_train, data_val, data_test = split_data(samples, frequencies, all_amplitudes, all_phases,  r_train, r_val)

x_test, y_test = split_xy(data_test)

y_test = np.argmax(y_test, axis=1)

y_AI = model.predict(x_test)

# gathering the data for final plot
jump = False
sigmas = []
mus = []
errsigma = []
errmu = []
sigma = []
mu = []
Xs = []
Ys = []
for i in range(len(y_test)-1):
    jump = y_test[i] != y_test[i+1] # jump if there is new type of sample
    last = i == len(y_test) - 2
    if jump or last:
        sigma.append(y_AI[i][1])
        mu.append(y_AI[i][0])
        Xs.append(mu)
        Ys.append(sigma)
        sigmas.append(np.average(np.asarray(sigma)))
        mus.append(np.average(np.asarray(mu)))
        errsigma.append(np.std(np.asarray(sigma)))
        errmu.append(np.std(np.asarray(mu)))

        sigma = []
        mu = []

    else:
        sigma.append(y_AI[i][1])
        mu.append(y_AI[i][0])


plt.figure()
plt.plot([1, 0, 0, 1], [0, 1, 0, 0], '--k')
for ii in range(len(mus)):
    plt.errorbar(mus[ii], sigmas[ii], errsigma[ii], errmu[ii],
                 ecolor=colors[ii], marker=symbols[ii], mfc=colors[ii], mec='k', label=names[ii], ms=12, fmt='o')

ox = np.arange(0, 1, 0.001)
oy = 1 - ox
plt.legend(loc=1, prop={'size': 9})
plt.ylim((-.1, 1.1))
plt.xlim((-.1, 1.1))
plt.show()


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

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)
tf.keras.models.save_model(model, 'basic_model')

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
