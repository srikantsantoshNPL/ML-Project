# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 09:46:21 2021

@author: ss38
"""
import numpy as np
from Normalisation import file_label_feature_frequency,normalise_data,output_directory_sorter,material_folder_finder,image_saver_from_array,file_in_material_sorter
import time

directory_input = (input("Enter Sorted files directory: ") or r'C:\Users\ss38\Rafa AI stuff\211108')
directory_output = (input("Enter Output files directory: ") or r'C:\Users\ss38\Rafa AI stuff\211108\Processed_Images2')
freq = (input("Enter Frequency of Scan in 4sf: ") or '90.21')

def master_function(directory_input,directory_output,freq):
    feature = 'phase'
    freq = str(freq)
    feature2 = 'ampl'
    folder_list = material_folder_finder(directory_input)
    print(folder_list)
    angle_range = np.arange(-35,40,5)
    for i in range(len(angle_range)): 
        data_file_label = file_label_feature_frequency(feature,folder_list,angle_range[i])
        print(data_file_label)
        data_norm_split = normalise_data(feature,data_file_label) 
        data_file_label2 = file_label_feature_frequency(feature2,folder_list,angle_range[i])
        data_norm_split2 = normalise_data(feature2,data_file_label2)
        direc_output = output_directory_sorter(directory_output, angle_range[i])
        print(direc_output)
        image_saver_from_array(direc_output,data_file_label,data_norm_split,data_norm_split2,feature2)
    time.sleep(1)
    for i in range(len(angle_range)):       
        file_in_material_sorter(direc_output)
    return 

master_function(directory_input, directory_output, freq)