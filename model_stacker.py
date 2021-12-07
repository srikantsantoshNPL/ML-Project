# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 10:54:27 2021

@author: ss38
"""
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import seaborn as sns

pred_all = pd.DataFrame(data = [])
m1_350 = tf.keras.models.load_model('basic_model_3.5')
test_set = r"C:\Users\ss38\Rafa AI stuff\210104_trimmed\freq_7.00_ampl_phase"
test_set2 = r"C:\Users\ss38\Rafa AI stuff\210104_trimmed\freq_29.00_ampl_phase"
m2_700 = tf.keras.models.load_model('basic_model_7.00')
testData = tf.keras.utils.image_dataset_from_directory(test_set,image_size = (180,180),labels='inferred',label_mode='categorical',seed=324893,batch_size=32)
testData2 = tf.keras.utils.image_dataset_from_directory(test_set2,image_size = (180,180),labels='inferred',label_mode='categorical',seed=324893,batch_size=32)



predictions = np.array([])
labels =  np.array([])
for x,y in testData:
  predictions = np.concatenate([predictions,np.argmax(m1_350.predict(x),axis=-1)])
  labels = np.concatenate([labels, np.argmax(y.numpy(), axis=-1)])


predictions2 = np.array([])
labels2 =  np.array([])
for x,y in testData2:
  predictions2 = np.concatenate([predictions2,np.argmax(m1_350.predict(x),axis=-1)])
  labels2 = np.concatenate([labels2, np.argmax(y.numpy(), axis=-1)])


test_set3 = r"C:\Users\ss38\Rafa AI stuff\210104_trimmed\freq_3.50_ampl_phase"
m3_290 = tf.keras.models.load_model('basic_model_7.00')
testData3 = tf.keras.utils.image_dataset_from_directory(test_set,image_size = (180,180),labels='inferred',label_mode='categorical',seed=324893,batch_size=32)
predictions3 = np.array([])
labels3 =  np.array([])
for x,y in testData3:
  predictions3 = np.concatenate([predictions,np.argmax(m1_350.predict(x),axis=-1)])
  labels3 = np.concatenate([labels, np.argmax(y.numpy(), axis=-1)])


predictions3  = predictions3[:-32]

   # join all the predictions together
pred_all['pred_1'] = predictions
pred_all['pred_2'] = predictions2
pred_all['pred_3'] = predictions3
mode_predictions = np.where(pred_all.nunique(1).eq(pred_all.shape[1]),np.nan,pred_all.mode(axis=1).iloc[:,0])
pred_all['Mode'] = mode_predictions
classes = testData.class_names

con_mat = tf.math.confusion_matrix(labels=labels, predictions=mode_predictions).numpy()
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





