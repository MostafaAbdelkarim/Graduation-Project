# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 20:23:41 2020

@author: Mostafa-PC
"""
  
import os

from PIL import Image

#def crop_image(img, crop_area, new_filename):
#    cropped_image = img.crop(crop_area)
#    cropped_image.save(new_filename)

def resize():
    f = r'D:/aptos2019-blindness-detection/Dr.Amr pre-processing/Histo-Spec'
    for file in os.listdir(f):
        f_img = f+"/"+file
        img = Image.open(f_img)
        img = img.resize((225,225))
        img.save(f_img)


#crop_areas = [(430, 50, 1800, 1430)]
#directory = r'E:/Amr S. Ghoneim/MESSIDOR/Base13'
#
#for i, crop_area in enumerate(crop_areas):
#    
#    for file in os.listdir(directory):
#        f_img = directory+"/"+file
#        img = Image.open(f_img)
#        filename = os.path.splitext(f_img)[0]
#        ext = os.path.splitext(f_img)[1]
#        new_filename = filename + ext
#
#        crop_image(img, crop_area, new_filename)
print("Starting resizing!")
resize()
print("Done resizing")
