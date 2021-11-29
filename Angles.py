# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 16:26:09 2021

@author: ss38
"""
import os,fnmatch
import glob
import numpy as np
import re
import shutil

path = r"C:\Users\ss38\Rafa AI stuff\211108"
filelist = []
def freq_finder(file):
    b= []
    b = re.findall(r"\d+",os.path.basename(file))
    c = b[1]+"."+ b[2]
    return float(c)

def angle_number_finder(file):
    b= []
    b = re.findall(r"\d+",os.path.basename(file))
    return b[0]

def angle_converter(file):
    angle_list = np.arange(-35,40,5)
    c = int(angle_number_finder(file))
    return angle_list[c]

    
# for i in range(1,5):
#     rename_path = os.path.join(path,'Part'+ str(i))
#     files = os.listdir(rename_path)
#     for file in files:
#         rename_file = os.path.join(rename_path,os.path.splitext(file)[0]+'_part'+str(i)+'.dat')
#         os.rename(os.path.join(rename_path,file),rename_file)
                                   
def output_folder_generator(directory,freq):
    a = []
    angle_list = np.arange(-35,40,5)
    for i in range(len(angle_list)):
        folder_name = 'freq_'+str(freq)+'_ampl_phase_'+str(angle_list[i])+'degrees'
        path = os.path.join(directory,folder_name)
        a.append(path)
        os.makedirs(path,exist_ok=True)
    return a
    


for root,dirs,files in os.walk(path):
    for file in files:
        if 'scan' in file:
            filelist.append(os.path.join(root,file))




# a = {'angle':[],'freq':[]}
# for i in range(len(filelist)):
#     a['angle'].append(angle_converter(filelist[i]))
#     a['freq'].append(freq_finder(filelist[i]))
#     freq = str(a['freq'][i])
#     angle = str(a['angle'][i])
#     new_path = os.path.join(path,freq)
   
#     if not os.path.exists(path+freq):
#         new_path = os.path.join(path,freq)
#         os.makedirs(os.path.join(path,freq),exist_ok=True)
#     if os.path.exists(new_path):
#         shutil.move(filelist[i],new_path)
        
#     if not os.path.exists(new_path+angle):
#         second_new_path = os.path.join(new_path,angle)
#         os.makedirs(second_new_path,exist_ok=True)
#     if os.path.exists(second_new_path):
#         shutil.move(filelist[i],second_new_path)

