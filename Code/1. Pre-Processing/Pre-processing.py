# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 21:53:16 2020

@author: Mostafa-PC
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

#list files in the directory
os.listdir('D:/aptos2019-blindness-detection/data/')
#read in training dataset
#df_train = pd.read_csv('train.csv', index_col = 0)
df_train = pd.read_csv('D:/aptos2019-blindness-detection/data/train.csv')
df_train.head()
df_train['diagnosis'].unique()
#add label column into training dataset
def add_label(df):
    if df['diagnosis'] == 0:
        val = "No DR"
    elif df['diagnosis'] == 1:
        val = "Mild"
    elif df['diagnosis'] == 2:
        val = "Moderate"
    elif df['diagnosis'] == 3:
        val = "Severe"
    elif df['diagnosis'] == 4:
        val = "Poliferative DR"
    return val
df_train['diagnosis_names'] = df_train.apply(add_label, axis=1)
df_train.head()
#show training images
fig = plt.figure(figsize=(25, 16))
# display 10 images from each class
for class_id in sorted(df_train['diagnosis'].unique()):
    for i, (idx, row) in enumerate(df_train.loc[df_train['diagnosis'] == class_id].sample(5).iterrows()):
        ax = fig.add_subplot(5, 10, class_id * 10 + i + 1, xticks=[], yticks=[])
        im = Image.open(f"D:/aptos2019-blindness-detection/data/train_images/{row['id_code']}.png")
        plt.imshow(im)
        ax.set_title(f'Label: {class_id}')
print("Shape of dataset:",df_train.shape)
PATH = "D:/aptos2019-blindness-detection/data/train_images"
image_size_list=[]
images_files = os.listdir(PATH)
for image in images_files :
    image_size_list.append(Image.open(os.path.join(PATH, image)).size)

images_size = np.array(image_size_list)
images_area =  images_size[:,0] * images_size[:,1]

DF = pd.DataFrame(images_size,columns=['Width','Height'])
DF.head()

SEED =3
IMG_SIZE = 225
NUM_CLASSES =5
fig = plt.figure(figsize=(25, 16))

for class_id in sorted(df_train.diagnosis.unique()):
    for i, (idx, row) in enumerate(df_train.loc[df_train['diagnosis'] == class_id].sample(5, random_state=SEED).iterrows()):
        ax = fig.add_subplot(5, 5, class_id * 5 + i + 1, xticks=[], yticks=[])
        path=f"D:/aptos2019-blindness-detection/data/train_images/{row['id_code']}.png"
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #image=cv2.addWeighted ( image, 0 , cv2.GaussianBlur( image , (0 ,0 ) , 10) ,-4 ,128)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

        plt.imshow(image, cmap='gray')
        ax.set_title('Label: %d-%d-%s' % (class_id, idx, row['id_code']) )


fig = plt.figure(figsize=(25, 16))
for class_id in sorted(df_train.diagnosis.unique()):
    for i, (idx, row) in enumerate(df_train.loc[df_train['diagnosis'] == class_id].sample(5, random_state=SEED).iterrows()):
        ax = fig.add_subplot(5, 5, class_id * 5 + i + 1, xticks=[], yticks=[])
        path=f"D:/aptos2019-blindness-detection/data/train_images/{row['id_code']}.png"
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        image=cv2.addWeighted (image,4, cv2.GaussianBlur(image,(0,0),IMG_SIZE/10),-4,128) # the trick is to add this line

        plt.imshow(image, cmap='gray')
        ax.set_title('Label: %d-%d-%s' % (class_id, idx, row['id_code']) )
        
fig = plt.figure(figsize=(25, 16))
for class_id in sorted(df_train.diagnosis.unique()):
    for i, (idx, row) in enumerate(df_train.loc[df_train['diagnosis'] == class_id].sample(5, random_state=SEED).iterrows()):
        ax = fig.add_subplot(5, 5, class_id * 5 + i + 1, xticks=[], yticks=[])
        path=f"D:/aptos2019-blindness-detection/data/train_images/{row['id_code']}.png"
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        image=cv2.addWeighted (image,4, cv2.GaussianBlur( image , (0,0) , IMG_SIZE/10) ,-4 ,128) # the trick is to add this line
        
        plt.imshow(image, cmap='gray')
        ax.set_title('Label: %d-%d-%s' % (class_id, idx, row['id_code']) )
        
for image_id in tqdm(df_train['id_code']):
     path=f"D:/aptos2019-blindness-detection/data/train_images/{image_id}.png"
     image = cv2.imread(path)
     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
     image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
     image=cv2.addWeighted (image,4, cv2.GaussianBlur( image , (0,0) , IMG_SIZE/10) ,-4 ,128) # the trick is to add this line
     cv2.imwrite("D:/aptos2019-blindness-detection/data/processed_color_images/{}.png".format(image_id), image)
        