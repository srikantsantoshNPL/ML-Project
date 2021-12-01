# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 16:37:49 2021

@author: ss38
"""
"""
This programme takes the RF scans of Amplitude and Phase and orders them correctly for use in 
the ML module. The Values are normalised wrt each other and then converted into RBG channels,
they are then merged to form a single image. Phase and Amplitude take up two different RGB channels and are
merged. The Phase is adjusted so that all values between 0-180 are the same as 360-180 since there is a 
phase shift at 180. But the files need to be formatted correctly from the angles.py programme, with 
a setup of Part,Material,Angle,file
Part1->1brass->0


"""
import shutil
from PIL import Image
import numpy as np
import os
from sklearn import preprocessing
import time
directory_input = r'C:\Users\ss38\Rafa AI stuff\211108'
keywords = ['1brass','1brs_on_ferrite','1co_on_ferrite','1copper','1ferrite','1frt_on_copper','1steel','2co_on_ferrite','no_plates']
directory_output = r'C:\Users\ss38\Rafa AI stuff\211108\Processed_Images2'

def material_folder_finder(directory_input,keywords = ['1brass','1brs_on_ferrite','1co_on_ferrite','1copper','1ferrite','1frt_on_copper','1steel','2co_on_ferrite','no_plates']):
    """  Finds all the material folders, the keywords define what their names are in the files.
Make sure all the materials are stored in a format, Part then material i.e.
Part1-> 1brass

        """ 
    folder_list = []
    list_subfolders_with_paths = [f.path for f in os.scandir(directory_input) if f.is_dir()]
    for i in range(len(keywords)):
        for item in list_subfolders_with_paths:
            list_sub_subfolders_with_paths = [f.path for f in os.scandir(item) if f.is_dir()]
            for subfolder in list_sub_subfolders_with_paths: 
                if keywords[i] == os.path.basename(subfolder):
                        folder_list.append(subfolder)
    return folder_list
def file_label_feature_frequency(feature,folder_list,set_angle):
    """ Returns all the filenames that corresponds to a specific angle
    
    """
    data_file_label = []
    for i in range(len(folder_list)):
        angle_folder_list = [f.path for f in os.scandir(folder_list[i]) if f.is_dir()]
        for angle in angle_folder_list:
            if os.path.split(angle)[1] == str(set_angle):
                files = os.listdir(angle)
                sorted_files = [filename for filename in files if feature in filename]
                for file in sorted_files:
                    filename = os.path.join(folder_list[i],angle,file)
                    data_file_label.append(filename)
    return data_file_label

def normalise_data(feature,data_file_label):
    """Normalises all the Amplitude or Phase values of all the materials that correspond to a specific angle
    with respect to each other, then multiply each normalised value by 255 to get it ready for an RGB channel

    """
    data = [[] for i in range(len(data_file_label))]
    for i in range(len(data_file_label)):
        data[i] = np.loadtxt(data_file_label[i])
    a = np.concatenate(data)
    if feature == 'phase':
        a = np.where(a>180,abs(a-360),a)
    data_norm = 255*preprocessing.normalize(a)
    data_norm_split = np.array_split(data_norm,len(data_file_label))
    return data_norm_split

def array_producer(feature,data_norm_split):
    """Takes the normalised data and puts it into an array ready to be screened as an image
    The array is produced such that it is can be easily merged with its corresponding Phase or Amplitude array

    """
    height,width = data_norm_split.shape
    array = np.zeros([height,width,3],dtype = np.uint8)
    for j in range(height):
        for k in range(width):
            if feature == 'ampl':
                m=2
                array[j][k][m] = (data_norm_split[j][k])
            elif feature == 'phase':
                m=1
                array[j][k][m] = (data_norm_split[j][k])
    return array
    
def image_saver_from_array(directory_output,data_file_label,data_norm_split,data_norm_split2 = None,feature2 = None):
    """ Produces the image from the generated array with one RGB channel representing Phase and the other Amplitude, this 
    then renames and saves the image in a chosen directory
    
    """
    for i in range(len(data_file_label)):
        array = array_producer(feature,data_norm_split[i]) 
        if data_norm_split2 is not None:
            array2 = array_producer(feature2,data_norm_split2[i])
            array = np.add(array,array2)
        img = Image.fromarray(array)
        basewidth = 500
        wpercent = (basewidth / float(img.size[0]))
        hsize = int((float(img.size[1]) * float(wpercent)))
        img = img.resize((basewidth, hsize), Image.ANTIALIAS)
        filename = os.path.basename(data_file_label[i])
        material_label = os.path.basename(os.path.dirname(os.path.dirname(data_file_label[i])))
        file = filename.split('.d',1)[0]+ material_label +'.png'  
        if data_norm_split2 is not None:
            file = os.path.splitext(filename)[0]+ material_label +feature+feature2+'.png'
        file_location1 = os.path.join(directory_output, file) 
        img.save(file_location1)
    return

def file_in_material_sorter(directory, keywords = ['1brass','1brs_on_ferrite','1co_on_ferrite','1copper','1ferrite','1frt_on_copper','1steel','2co_on_ferrite','no_plates']):
    """
Sorts the output images in the output directory into its individual material folder which it creates within the programme

    """
    files = os.listdir(directory)
    sorted_files = [filename for filename in files if '.p' in filename]
    for item in keywords:
        for filename in sorted_files:
            if item in filename:
                if not os.path.exists(directory+item):
                    new_path = os.path.join(directory,item)
                    os.makedirs(os.path.join(directory,item),exist_ok=True)
                if os.path.exists(new_path):
                    a = os.path.join(directory,filename)
                    shutil.move(a,new_path)
    return


def output_directory_sorter(directory_output,angle,freq):
    """ Input a defined output directory for the images and creates within it a folder for each individual angle recorded
    """
    a = []
    angle_list = np.arange(-35,40,5)
    for i in range(len(angle_list)):
        folder_name = 'freq_'+str(freq)+'_ampl_phase_'+str(angle_list[i])+'degrees'
        path = os.path.join(directory_output,folder_name)
        a.append(path)
        os.makedirs(path,exist_ok=True)

    b = []
    c= []
    if angle ==0 or angle ==5 :
        for folder in a:
            if len(os.path.basename(folder))==30: 
                b.append(folder)
                for subfolder in b:
                    if str(angle) in folder[-15:]:
                        thi = folder
                        
                        
    elif angle != 0 or angle != 5:
        for folder in a:
            c.append(os.path.basename(folder))
            for subfolder in c:
                if str(angle) in folder[-15:]:
                    thi = folder 
    return thi


feature = 'phase'
freq = '25.04'
feature2 = 'ampl'
folder_list = material_folder_finder(directory_input)
# angle_range = np.arange(-35,40,5)
# for i in range(len(angle_range)):   
    # print(angle_range[i])
    # data_file_label = file_label_feature_frequency(feature,folder_list,angle_range[i])
    # data_norm_split = normalise_data(feature,data_file_label)
    # data_file_label2 = file_label_feature_frequency(feature2,folder_list,angle_range[i])
    # data_norm_split2 = normalise_data(feature2,data_file_label2)
    # direc_output = output_directory_sorter(directory_output, angle_range[i])
    #image_saver_from_array(direc_output,data_file_label,data_norm_split,data_norm_split2,feature2)
    #file_in_material_sorter(direc_output)
    


data_file_label = file_label_feature_frequency(feature2,folder_list,-25)
data_norm_split = normalise_data(feature,data_file_label)
#data_file_label2 = file_label_feature_frequency(feature2,folder_list,5)
#data_norm_split2 = normalise_data(feature2,data_file_label2)
#image_saver_from_array(directory_output,data_file_label,data_norm_split,data_norm_split2,feature2)

#image_saver_from_array(directory_output,data_file_label,data_norm_split)
#test_set = r'C:\Users\ss38\Rafa AI stuff\211108\freq_25.02_ampl_phase'

#file_in_material_sorter(directory_output)



    

