# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 20:05:47 2020

@author: Mostafa-PC
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as pyplot
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dropout, GaussianNoise, GaussianDropout
from tensorflow.keras.layers import (Activation, Dropout, Flatten, Dense, GlobalMaxPooling2D,
                                     MaxPooling2D,SeparableConv2D,BatchNormalization, Input, 
                                     Conv2D, GlobalAveragePooling2D)
from tensorflow.keras import metrics
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.utils import Sequence
from sklearn.metrics import classification_report
from sklearn.utils import class_weight
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.resnet50 import ResNet50


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

labelMap = {0:'No DR',1:'Mild',2:'Moderate',3:'Severe',4:'Proliferative DR'}

# read in the preprocessed images
N = df_train.shape[0]
X = np.empty((N, 225, 225, 3), dtype=np.uint8)

for i, image_id in enumerate(tqdm(df_train['id_code'])):
    X[i, :, :, :] = cv2.imread(f'D:/aptos2019-blindness-detection/data/processed_color_images/{image_id}.png')

# Model parameters
IMG_SIZE = 225
SEED = 53
BATCH_SIZE = 8
EPOCHS = 5
WARMUP_EPOCHS = 2
LEARNING_RATE = 1e-4
WARMUP_LEARNING_RATE = 1e-3
N_CLASSES = df_train['diagnosis'].nunique()
ES_PATIENCE = 5
RLROP_PATIENCE = 3
DECAY_DROP = 0.5

y = to_categorical(df_train['diagnosis'], num_classes=5)
#y = df_train['diagnosis']
print(X.shape)
print(y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=SEED)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# Random Rotations
plt.figure(figsize=(20,10))
datagen = ImageDataGenerator(rotation_range=180,
                            horizontal_flip=True,
                            vertical_flip=True)

# fit parameters from data
datagen.fit(X_train)

# Configure batch size and retrieve one batch of images
for X_batch, y_batch in datagen.flow(X_train, y_train, batch_size=9):
    # Show 9 images
    for i in range(0, 9):
        pyplot.subplot(330 + 1 + i)
        pyplot.imshow((X_batch[i].reshape(225, 225, 3).astype('uint8')))
    # show the plot
    pyplot.show()
    break

# Add Image augmentation to our generator
train_datagen = ImageDataGenerator(rotation_range=180,
                                   horizontal_flip=True,
                                   vertical_flip=True,
                                   rescale =1./128,
                                  validation_split=0.20)
#train_datagen.fit(X_train)

train_generator=train_datagen.flow(X_train,y_train, batch_size =BATCH_SIZE,subset='training',seed=SEED)
valid_generator= train_datagen.flow(X_train, y_train,batch_size=BATCH_SIZE,subset='validation',seed=SEED)

def create_model(input_shape, n_out):
    input_tensor = Input(shape=input_shape)
    base_model = ResNet50(include_top=False,
                   weights='imagenet',
                   input_tensor=input_tensor)
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    #dense 
    final_output = Dense(n_out, activation="softmax", name='final_output')(x)
    model = Model(input_tensor, final_output)
    
    return model

model = create_model(
    input_shape=(IMG_SIZE,IMG_SIZE,3), 
    n_out=N_CLASSES)

for layer in model.layers:
    layer.trainable = False

for i in range(-5, 0):
    model.layers[i].trainable = True
    
class_weights = class_weight.compute_class_weight('balanced', np.unique(df_train['diagnosis'].astype('int').values), df_train['diagnosis'].astype('int').values)

metric_list = ["accuracy"]
optimizer = Adam(lr=WARMUP_LEARNING_RATE)
model.compile(optimizer=optimizer, loss='categorical_crossentropy',  metrics=metric_list)
model.summary()

STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size
STEP_SIZE_VALID = valid_generator.n//valid_generator.batch_size

history_warmup = model.fit_generator(generator=train_generator,
                                     steps_per_epoch=STEP_SIZE_TRAIN,
                                     validation_data=valid_generator,
                                     validation_steps=STEP_SIZE_VALID,
                                     epochs=WARMUP_EPOCHS,
                                     class_weight=class_weights,
                                     verbose=1).history
                                
for layer in model.layers:
    layer.trainable = True

es = EarlyStopping(monitor='val_loss', mode='min', patience=ES_PATIENCE, restore_best_weights=True, verbose=1)
rlrop = ReduceLROnPlateau(monitor='val_loss', mode='min', patience=RLROP_PATIENCE, factor=DECAY_DROP, min_lr=1e-6, verbose=1)
callback_list = [es, rlrop]
optimizer = Adam(lr=LEARNING_RATE)
model.compile(optimizer=optimizer, loss='categorical_crossentropy',  metrics=metric_list)
model.summary()

history_finetunning = model.fit_generator(generator=train_generator,
                                          steps_per_epoch=STEP_SIZE_TRAIN,
                                          validation_data=valid_generator,
                                          validation_steps=STEP_SIZE_VALID,
                                          epochs=EPOCHS,
                                          callbacks=callback_list,
                                          class_weight=class_weights,
                                          verbose=1).history
                                          
history = {'loss': history_warmup['loss'] + history_finetunning['loss'], 
           'val_loss': history_warmup['val_loss'] + history_finetunning['val_loss'], 
           'acc': history_warmup['accuracy'] + history_finetunning['accuracy'], 
           'val_acc': history_warmup['val_accuracy'] + history_finetunning['val_accuracy']}

sns.set_style("whitegrid")
fig, (ax1, ax2) = plt.subplots(2, 1, sharex='col', figsize=(10, 7))

ax1.plot(history['loss'], label='Train loss')
ax1.plot(history['val_loss'], label='Validation loss')
ax1.legend(loc='best')
ax1.set_title('Loss')

ax2.plot(history['acc'], label='Train accuracy')
ax2.plot(history['val_acc'], label='Validation accuracy')
ax2.legend(loc='best')
ax2.set_title('Accuracy')

plt.xlabel('Epochs')
sns.despine()
plt.show()

test_generator= train_datagen.flow(X_test, y_test,batch_size=BATCH_SIZE,seed=SEED)

STEP_SIZE_TEST = test_generator.n//test_generator.batch_size
model.evaluate(test_generator,steps=STEP_SIZE_TEST)

# Create empty arays to keep the predictions and labels
lastFullTrainPred = np.empty((0, N_CLASSES))
lastFullTrainLabels = np.empty((0, N_CLASSES))
lastFulltestPred = np.empty((0, N_CLASSES))
lastFulltestLabels = np.empty((0, N_CLASSES))

# Add train predictions and labels
for i in range(STEP_SIZE_TRAIN+1):
    im, lbl = next(train_generator)
    scores = model.predict(im, batch_size=train_generator.batch_size)
    lastFullTrainPred = np.append(lastFullTrainPred, scores, axis=0)
    lastFullTrainLabels = np.append(lastFullTrainLabels, lbl, axis=0)

# Add validation predictions and labels
for i in range(STEP_SIZE_TEST+1):
    im, lbl = next(test_generator)
    scores = model.predict(im, batch_size=test_generator.batch_size)
    lastFulltestPred = np.append(lastFulltestPred, scores, axis=0)
    lastFulltestLabels = np.append(lastFulltestLabels, lbl, axis=0)
    
    
lastFullComPred = np.concatenate((lastFullTrainPred, lastFulltestPred))
lastFullComLabels = np.concatenate((lastFullTrainLabels, lastFulltestLabels))
complete_labels = [np.argmax(label) for label in lastFullComLabels]

train_preds = [np.argmax(pred) for pred in lastFullTrainPred]
train_labels = [np.argmax(label) for label in lastFullTrainLabels]
test_preds = [np.argmax(pred) for pred in lastFulltestPred]
test_labels = [np.argmax(label) for label in lastFulltestLabels]

from sklearn.metrics import confusion_matrix
fig, (ax1, ax2) = plt.subplots(1, 2, sharex='col', figsize=(25, 8))
labels = ['0 - No DR', '1 - Mild', '2 - Moderate', '3 - Severe', '4 - Proliferative DR']
train_cnf_matrix = confusion_matrix(train_labels, train_preds)
test_cnf_matrix = confusion_matrix(test_labels, test_preds)

train_cnf_matrix_norm = train_cnf_matrix.astype('float') / train_cnf_matrix.sum(axis=1)[:, np.newaxis]
test_cnf_matrix_norm = test_cnf_matrix.astype('float') / test_cnf_matrix.sum(axis=1)[:, np.newaxis]

train_df_cm = pd.DataFrame(train_cnf_matrix_norm, index=labels, columns=labels)
test_df_cm = pd.DataFrame(test_cnf_matrix_norm, index=labels, columns=labels)

sns.heatmap(train_df_cm, annot=True, fmt='.2f', cmap="Blues", ax=ax1).set_title('Train')
sns.heatmap(test_df_cm, annot=True, fmt='.2f', cmap=sns.cubehelix_palette(8), ax=ax2).set_title('Test')
plt.show()

#model_yaml = model.to_yaml()
#with open("D:/aptos2019-blindness-detection/models/Resnet50.yaml", "w") as yaml_file:
#    yaml_file.write(model_yaml)
#
#model.save("D:/aptos2019-blindness-detection/models/Resnet50.h5")
#print("Saved model to disk")

print(classification_report(test_preds, test_labels, target_names=labels))