# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 09:33:12 2021

@author: ss38
"""
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pickle
from tensorflow import keras
import pandas as pd
import seaborn as sns

test_set = r'C:\Users\ss38\Rafa AI stuff\211108\freq_25.02_ampl_phase_0degrees'
testData = tf.keras.utils.image_dataset_from_directory(test_set,image_size = (180,180),labels='inferred',label_mode='categorical',seed=324893,batch_size=32)
basic_model = tf.keras.models.load_model('basic_model')
classes = testData.class_names
#results = basic_model.evaluate(testData,batch_size=32)
predictions = np.array([])
labels =  np.array([])
for x, y in testData:
  predictions = np.concatenate([predictions,np.argmax(basic_model.predict(x),axis=-1)])
  labels = np.concatenate([labels, np.argmax(y.numpy(), axis=-1)])

con_mat = tf.math.confusion_matrix(labels=labels, predictions=predictions).numpy()
con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
con_mat_df = pd.DataFrame(con_mat_norm,
                      index = classes, 
                      columns = classes)

figure = plt.figure(figsize=(8, 8))
sns.heatmap(con_mat_df, annot=True,cmap=plt.cm.Blues)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
a = 0
for i in range(len(predictions)):
    if predictions[i] == labels[i]:
        a+=1
accuracy = a/len(predictions)
print(accuracy)
    