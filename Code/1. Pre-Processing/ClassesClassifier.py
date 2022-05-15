# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 02:26:59 2020

@author: Mostafa-PC
"""

import pandas as pd
from shutil import copyfile

trainCSVPath = "D:/aptos2019-blindness-detection/data/train.csv"
trainLabels = pd.read_csv(trainCSVPath, encoding='latin-1')
data_4 = trainLabels[trainLabels['diagnosis']==4]

for i in range(len(data_4)):
    stringpath = "D:/aptos2019-blindness-detection/data/train_images/"+data_4['id_code'].iloc[i]+".png"
    stringpath2 = "D:/aptos2019-blindness-detection/NewClassified/1/"+data_4['id_code'].iloc[i]+".png"
    copyfile(stringpath, stringpath2)

#print(data_4)
 