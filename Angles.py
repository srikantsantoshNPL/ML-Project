# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 16:26:09 2021

@author: ss38
This programme finds all the files for processing and puts it into the correct formatting for process
"""
import os,fnmatch
import glob
import numpy as np
import re
import shutil


def freq_finder(file):
    """Finds the frequency of the file
    """
    b= []
    b = re.findall(r"\d+",os.path.basename(file))
    c = b[0]+"."+ b[1]
    return float(c)

def angle_number_finder(file):
    """Finds the angle of the plate for the file scan but is in the wrong format

    """
    b= []
    b = re.findall(r"\d+",os.path.basename(file))
    return b[2]

def angle_converter(file):
    """Ouputs the real angle of the plate found from the angle number finder function

    """
    angle_list = np.arange(-35,40,5)
    c = int(angle_number_finder(file))
    return angle_list[c]

def file_rename_part(no_of_parts,path): 
    """Sometimes input data is in individual parts within a material,this renames the filenames so these individual parts can be removed
    and all the files can be placed together, otherwise the files may have the same name as the others and it can be hard for programmes to 
    distringuish them

    """
    for i in range(1,no_of_parts+1):
        rename_path = os.path.join(path,'Part'+ str(i))
        files = os.listdir(rename_path)
        for file in files:
            rename_file = os.path.join(rename_path,os.path.splitext(file)[0]+'_part'+str(i)+'.dat')
            os.rename(os.path.join(rename_path,file),rename_file)
    return
def output_folder_generator(directory,freq):
    a = []
    angle_list = np.arange(-35,40,5)
    for i in range(len(angle_list)):
        folder_name = 'freq_'+str(freq)+'_ampl_phase_'+str(angle_list[i])+'degrees'
        path = os.path.join(directory,folder_name)
        a.append(path)
        os.makedirs(path,exist_ok=True)
    return a
    

def file_finder(path):
    """Finds all the files in the path even ones within subfolders

    """
    filelist = []
    for root,dirs,files in os.walk(path):
        for file in files:
            if 'scan' in file:
                filelist.append(os.path.join(root,file))
    return filelist
# path2 = r"C:\Users\ss38\Rafa AI stuff\211130"
# files = file_finder(path2)
# freq_list = [3.94,7.03,30.15]   
# keywords = ['1copper','1ferrite']
# # for freq in freq_list:
# #     for file in files:
# #         if str(freq) in os.path.basename(file):
# #             new_path = os.path.join(path2,str(freq))
# #             if not os.path.exists(new_path):
# #                 os.makedirs(new_path,exist_ok = True)
# #             if os.path.exists(new_path):
# #                 shutil.move(file,new_path)
# for freq in freq_list:
#     files = file_finder(os.path.join(path2,str(freq)))
#     new_path = os.path.join(path2,str(freq))
#     for keyword in keywords:
#         for file in files: 
#             if keyword in os.path.basename(file):
#                 new_path2 =os.path.join(new_path,keyword)
#                 if not os.path.exists(new_path2):
#                     os.makedirs(new_path2,exist_ok=True)
#                 if os.path.exists(new_path2):
#                     shutil.move(file,new_path2)


       
def sort_by_freq_then_angle(filelist,feature):
    """ Sorts all the files either wihtin its frequency or angle 
    """
    a = {'angle':[],'freq':[]}
    for i in range(len(filelist)):
        path = os.path.split(filelist[i])[0]
        a['angle'].append(angle_converter(filelist[i]))
        a['freq'].append(freq_finder(filelist[i]))
        freq = str(a['freq'][i])
        angle = str(a['angle'][i])
        new_path = os.path.join(path,freq)
        if feature == 'freq':
            if not os.path.exists(path+freq):
                new_path = os.path.join(path,freq)
                os.makedirs(os.path.join(path,freq),exist_ok=True)
            if os.path.exists(new_path):
                shutil.move(filelist[i],new_path)
        if feature == 'angle':         
            if not os.path.exists(path+angle):
                second_new_path = os.path.join(path,angle)
                os.makedirs(second_new_path,exist_ok=True)
            if os.path.exists(second_new_path):
                shutil.move(filelist[i],second_new_path)
    return


path = r"C:\Users\ss38\Rafa AI stuff\211130\7.03\Part1"
filelist2 = file_finder(path)

# path_1 = r"C:\Users\ss38\Rafa AI stuff\211130\30.15\Part1\1ferrite"
# for file in filelist:
#     if os.path.basename(file).startswith('Part4'):
#         filename = os.path.basename(file).split('Part4',1)[1]
#         os.rename(file,os.path.join(path_1,filename.replace('.dat','_part4.dat')))
sort_by_freq_then_angle(filelist2,'angle')