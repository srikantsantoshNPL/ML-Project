# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 14:40:54 2021

@author: ss38
"""
from sklearn import preprocessing
import numpy as np
from PIL import Image
import os

def PictureProducer(filename,material,directory_output,feature):
    """
    Function that takes in a filename and the material which you want an image produced for
    returns the image in the directory specified. Be careful about the dimensions with which you want the image.
    Try and group it with folders with the same material name.
    """
    new_folder = os.path.split(filename)
    data = np.loadtxt(filename)
    a,b = data.shape
    if feature == 'ampl':
        data_norm = 255*preprocessing.normalize(data)
    elif feature == 'phase':
        data = np.where(data>180,abs(data-360),data)
        data_norm = 255*preprocessing.normalize(data)
    width = b
    height = a
    array = np.zeros([height,width,3],dtype = np.uint8)
    for i in range(len(data)):   
        for j in range(len(data)):
            if feature == 'ampl2':
                array[i][j][2] = (data_norm[i][j])
            elif feature == 'phase':
                array[i][j][1] = (data_norm[i][j])
    img = Image.fromarray(array)
    basewidth = 500
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    dir_name = directory_output
    img = img.resize((basewidth, hsize), Image.ANTIALIAS)
    file = new_folder[1].split('.d',1)[0]+ '.png'   
    file_location1 = os.path.join(dir_name, file) 
    img.save(file_location1)
    return 

def folder_allocater(keywords,directory):
    """ Function which takes in the directory, and the materials that you are looking for must be in string format and in the format of how they are saved
    as. Returns all the folders with the same material name in its own individual list, embedded
    within a larger list of all the folders.   
    """
    b = [[] for i in range(len(keywords))]
    list_subfolders_with_paths = [f.path for f in os.scandir(directory) if f.is_dir()]
    for i in range(len(keywords)):
        for item in list_subfolders_with_paths:
            list_sub_subfolders_with_paths = [f.path for f in os.scandir(item) if f.is_dir()]
            for subfolder in list_sub_subfolders_with_paths: 
                if keywords[i] == os.path.basename(subfolder):
                        b[i].append(subfolder)
    return b


def Picture_generator_Directory_saver(keywords,directory_input,directory_output,feature):
    """
 Function puts it all together, takes in directory finds all the files, sorts them and produces images
and saves to the directory you want. Feature can be 'ampl' but we shall incorporate phase and then both eventually
to lock in better images. For keywords, add only the materials of the results and make sure the exact same string in 
the save folder is used for it.  
    
    """


    b = folder_allocater(keywords,directory_input)
    for i in range(len(b)):
         for j in range(len(b[i])):
             files = os.listdir(b[i][j])
             sorted_files = [filename for filename in files if feature in filename]
             for file in sorted_files:
                 filename = os.path.join(b[i][j],file)
                 PictureProducer(filename,os.path.basename(b[i][j]),directory_output,feature)
    return