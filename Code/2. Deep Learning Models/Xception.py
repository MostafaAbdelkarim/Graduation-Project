# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 19:32:03 2020

@author: Mostafa-PC
"""

from tensorflow.python.client import device_lib
device_lib.list_local_devices()

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, partial, rand, space_eval
from sklearn.metrics import log_loss
import math
import os
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from tensorflow.keras.utils import to_categorical
import tensorflow.keras
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras.models import load_model
# import keras.callbacks as kcall
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Activation, Dropout, Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.applications.xception import Xception, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.regularizers import l2, l1
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import tensorflow.keras.callbacks as kcallbacks
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, concatenate
from tensorflow.keras import optimizers, metrics, models
from tensorflow.keras.layers import Input, Flatten, Dense
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import tensorflow as tf

batch_size = 16
img_height, img_width = 299, 299
input_shape = (img_height, img_width, 3)
epochs = 100
tf.compat.v1.disable_eager_execution()
train_dir = 'D:/aptos2019-blindness-detection/NewClassified/new/train'
test_dir = 'D:/aptos2019-blindness-detection/NewClassified/new/test'

def preprocess_input(x):
    # 'RGB'->'BGR'
    x = x[:, :, ::-1]
    # Zero-center by imagenet mean pixel
    x[:, :, 0] -= 103.939
    x[:, :, 1] -= 116.779
    x[:, :, 2] -= 123.68
    return x

random_seed = np.random.seed(1142)

train_datagen = ImageDataGenerator(
    rescale=(1. / 255),
##     featurewise_center=True,
##     featurewise_std_normalization=True,
 #   (preprocessing_function = preprocess_input),
    validation_split= 0.2)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    seed = random_seed,
    shuffle = False,
    subset = 'training',
    classes = ['0','4'],
    class_mode='binary')


validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    seed = random_seed,
    shuffle = False,
    subset = 'validation',
    classes = ['0','4'],
    class_mode='binary')

test_datagen = ImageDataGenerator(rescale=1. / 255, preprocessing_function = preprocess_input)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    seed = random_seed,
    shuffle = False,
    class_mode='binary')

nb_train_samples = len(train_generator.filenames)
nb_validation_samples = len(validation_generator.filenames)
nb_test_samples = len(test_generator.filenames)

predict_size_train = int(math.ceil(nb_train_samples / batch_size))
predict_size_validation = int(math.ceil(nb_validation_samples / batch_size))
predict_size_test = int(math.ceil(nb_test_samples / batch_size))

num_classes = len(train_generator.class_indices)

print("nb_train_samples:", nb_train_samples)
print("nb_validation_samples:", nb_validation_samples)
print("nb_test_samples:", nb_test_samples)

print("\npredict_size_train:", predict_size_train)
print("predict_size_validation:", predict_size_validation)
print("predict_size_test:", predict_size_test)

print("\n num_classes:", num_classes)

os.mkdir("extracted_features")
extracted_features_dir = "extracted_features/"
model_name = "Xception"

#vgg19_weights ="../input/full-keras-pretrained-no-top/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5"
#inception_weights ="../input/full-keras-pretrained-no-top//inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"
#vgg16_weights ="../input/full-keras-pretrained-no-top/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
#denseNet201_weights ="../input/full-keras-pretrained-no-top/densenet201_weights_tf_dim_ordering_tf_kernels_notop.h5"
#denseNet121_weights ="../input/full-keras-pretrained-no-top/densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5"
#resenet50_weights ="../input/full-keras-pretrained-no-top/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"
#inception_resnet_v2_weights ="../input/full-keras-pretrained-no-top/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5"
#nasnet_weights ="../input/full-keras-pretrained-no-top/nasnet_large_no_top.h5"
#nasnet_mobile_weights ="../input/full-keras-pretrained-no-top/nasnet_mobile_no_top.h5"
#mobilenet_weights ="../input/full-keras-pretrained-no-top/mobilenet_1_0_224_tf_no_top.h5"
#xception_weights = "../input/full-keras-pretrained-no-top/xception_weights_tf_dim_ordering_tf_kernels_notop.h5"
xception_weights1 = 'C:/Users/Mostafa-PC/Desktop/Python Apps/Weights/xception_weights_tf_dim_ordering_tf_kernels_notop.h5'


#from tensorflow.keras.applications.vgg19 import VGG19
#from tensorflow.keras.applications.inception_v3 import InceptionV3
#from tensorflow.keras.applications.vgg16 import VGG16
#from tensorflow.keras.applications.xception import Xception, preprocess_input
#from tensorflow.keras.applications import DenseNet201
#from tensorflow.keras.applications import DenseNet121
#from tensorflow.keras.applications import ResNet50
#from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
#from tensorflow.keras.applications import NASNetLarge, NASNetMobile
#from tensorflow.keras.applications import MobileNet


model = Xception(weights="imagenet", include_top=False, pooling = 'avg', input_tensor=Input(shape=input_shape))

c1 = model.layers[16].output 
c1 = GlobalAveragePooling2D()(c1)       

c2 = model.layers[26].output
c2 = GlobalAveragePooling2D()(c2)       

c3 = model.layers[36].output
c3 = GlobalAveragePooling2D()(c3)       

c4 = model.layers[126].output
c4 = GlobalAveragePooling2D()(c4) 

con = concatenate([ c2, c3, c4])

bottleneck_final_model = Model(inputs=model.input, outputs=con)
max_q_size = 1
pickle_safe= False
bottleneck_features_train = bottleneck_final_model.predict_generator(train_generator, predict_size_train)
np.save(extracted_features_dir+'bottleneck_features_train_'+model_name+'.npy', bottleneck_features_train)

bottleneck_features_test = bottleneck_final_model.predict_generator(test_generator, predict_size_test)
np.save(extracted_features_dir+'bottleneck_features_test_'+model_name+'.npy', bottleneck_features_test)

bottleneck_features_validation = bottleneck_final_model.predict_generator(validation_generator, predict_size_validation)
np.save(extracted_features_dir+'bottleneck_features_validation_'+model_name+'.npy', bottleneck_features_validation)

train_data = np.load(extracted_features_dir+'bottleneck_features_train_'+model_name+'.npy')
validation_data = np.load(extracted_features_dir+'bottleneck_features_validation_'+model_name+'.npy')
test_data = np.load(extracted_features_dir+'bottleneck_features_test_'+model_name+'.npy')

train_labels = train_generator.classes
train_labels = to_categorical(train_labels, num_classes=num_classes)

validation_labels = validation_generator.classes
validation_labels = to_categorical(validation_labels, num_classes=num_classes)

test_labels = test_generator.classes
test_labels = to_categorical(test_labels, num_classes=num_classes)

#print(model.summary())
#print(train_data.shape)

adam_opt=Adam(lr=1e-05, beta_1=0.3, beta_2=0.6)

model = Sequential()
model.add(Dense(128, activation="relu", kernel_regularizer=l2(1e-05), bias_regularizer=l2(1e-05), activity_regularizer=l1(1e-06)))
#model.add(Dropout(0.5))
##
#model.add(Dense(1024, activation="relu", kernel_regularizer=l2(1e-05), bias_regularizer=l2(1e-05), activity_regularizer=l1(1e-06)))
#model.add(Dropout(0.5))

model.add(Dense(128, activation="relu", kernel_regularizer=l2(1e-06), bias_regularizer=l2(0.01), activity_regularizer=l1(1e-07)))
model.add(Dropout(0.5))

model.add(Dense(num_classes, activation="sigmoid"))

model.compile(optimizer=adam_opt, loss='binary_crossentropy', metrics=['accuracy'])

saveBestModel = kcallbacks.ModelCheckpoint(filepath='./Xception-best.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

history = model.fit(train_data, train_labels,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(validation_data, validation_labels),
                    verbose= 1,
                    callbacks=[saveBestModel])

score = model.evaluate(validation_data, validation_labels, verbose=0)

print('\n')
print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')
print('\n')



model = load_model('./Xception-best.h5', compile = True)
print('Model loaded successfully')
#img = cv2.imread('D:/KaggleDataset/BinaryDR-NoDR/train/0/418_right.jpeg')
#img = cv2.resize(img,(224,224))
#img = np.reshape(img,[1,224,224,3])
classes = model.predict_classes(test_data)
#print (classes)
#print(test_labels)
#test_labels2 = test_labels.reshape(-1)
#print(test_labels2)
#classes2 = test_labels2[[True, False,True, False,True, False,True, False,True, False,True, False,True, False,True, False,True, False,True, False,True, False,True, False,True, False,True, False,True, False,True, False,True, False,True, False,True, False,True, False]]
#print(classes2)
#classes3 = classes2.reshape(-1)
#print(classes3)


(eval_loss, eval_accuracy) = model.evaluate(validation_data, validation_labels, batch_size= batch_size, verbose=1)

print("Validation Accuracy: {:.4f}%".format(eval_accuracy * 100))
print("Validation Loss: {}".format(eval_loss))

filename = test_generator.filenames
truth = test_generator.classes
label = test_generator.class_indices
indexlabel = dict((value, key) for key, value in label.items())


preds = model.predict(test_data)

predictions = [i.argmax() for i in preds]
y_true = [i.argmax() for i in test_labels]
cm = confusion_matrix(y_pred=predictions, y_true=y_true)

print('Test Accuracy: {}'.format(accuracy_score(y_true=y_true, y_pred=predictions)))

plt.rcParams["axes.grid"] = False
plt.rcParams.update({'font.size': 20})

labels = []

label = test_generator.class_indices
indexlabel = dict((value, key) for key, value in label.items())

for k,v in indexlabel.items():
    labels.append(v)


import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion Matrix')

    print(cm)
#     fig = plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
#     plt.savefig('plots/3.InceptionV3-2-Private-DataSet-CM.png', bbox_inches='tight', dpi = 100) 


plt.figure(figsize=(5,5))
plot_confusion_matrix(cm, classes=labels, title=' ')


y_pred=predictions
y_pred_probabilities=y_pred

# y_pred = np.argmax(y_pred,axis = 1) 
y_actual = y_true

classnames=[]
for classname in test_generator.class_indices:
    classnames.append(classname)

confusion_mtx = confusion_matrix(y_actual, y_pred) 
print(confusion_mtx)
target_names = classnames
print(classification_report(y_actual, y_pred, target_names=target_names))

tensorflow.keras.backend.clear_session()


#from keras.models import load_model
#from keras.preprocessing import image
#import matplotlib.pyplot as plt
#import numpy as np
#import os
#
## load model
#model = load_model("./Xception-best.h5")
## image path
#img_path = 'D:/KaggleDataset/BinaryDR-NoDR/train/0/418_right.jpeg'    # dog
##img_path = '/media/data/dogscats/test1/19.jpg'      # cat
## load a single image
#img = image.load_img(img_path, target_size=(224, 224))
#img_tensor = image.img_to_array(img)                    # (height, width, channels)
#img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
#img_tensor /= 255.    
#new_image = img_tensor
## check prediction
#preds = model.predict(new_image)

