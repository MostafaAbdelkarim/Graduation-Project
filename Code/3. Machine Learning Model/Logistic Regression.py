# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 20:03:08 2020

@author: Mostafa-PC
"""
#Import necessary libraries
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix as CM
from random import randint
from keras.utils import plot_model
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.manifold import TSNE
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly
import plotly.graph_objs as go
from sklearn.linear_model import LogisticRegression

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

SEED = 53
IMG_SIZE = 225
NUM_CLASSES =5
labelMap = {0:'No DR',1:'Mild',2:'Moderate',3:'Severe',4:'Proliferative DR'}

# read in the preprocessed images
N = df_train.shape[0]
X = np.empty((N, 225, 225, 3), dtype=np.uint8)

for i, image_id in enumerate(tqdm(df_train['id_code'])):
    X[i, :, :, :] = cv2.imread(f'D:/aptos2019-blindness-detection/data/processed_color_images/{image_id}.png')

# normalize
X1 = X / 225

# reshape
X1 = X1.reshape(X1.shape[0], -1)

trainX, valX, trainy, valy = train_test_split(X1, df_train['diagnosis'], test_size=0.5, random_state=1220)

X_decomposed = TSNE(n_components=2).fit_transform(trainX)

fig, ax = plt.subplots(1, 1, figsize=(12, 8))

for i in range(5):
    idx = trainy == i
    ax.scatter(X_decomposed[idx, 0], X_decomposed[idx, 1], label=labelMap[i])
ax.set_title("t-SNE")
ax.legend()

y = df_train['diagnosis']
print(X.shape)
print(y.shape)

# reshape
X_norm = X.reshape(X.shape[0], -1)

X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.3, random_state=SEED)

scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

pca = PCA().fit(X_train_scaled)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');

pca = PCA(n_components=750).fit(X_train_scaled)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.fit_transform(X_test_scaled)

train_idx = 2563
pca = PCA(n_components=3)
emb = pca.fit_transform(X_train_scaled)
te_emb = emb[train_idx:]
emb = emb[:train_idx]

y = y_train
init_notebook_mode(connected=True)

data = [go.Scatter3d(x=emb[np.array(y) == t, 0],y=emb[np.array(y) == t, 1],z=emb[np.array(y) == t, 2],mode='markers',marker=dict(size=3,opacity=0.75),name=labelMap[t])for t in list(set(y))]

plotly.offline.init_notebook_mode()

data.append(go.Scatter3d(x=te_emb[:, 0],y=te_emb[:, 1],z=te_emb[:, 2],mode='markers',marker=dict(color='#c0c0c0',size=2,opacity=0.75),name='test data'))

layout = go.Layout(margin=dict(l=0,r=0,b=0,t=0),)

fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='simple-3d-scatter')

from imblearn.over_sampling import SMOTE
# using SMOTE for unbalanced data
X_train_pca_s, y_train_s= SMOTE().fit_resample(X_train_pca, y_train)

import collections
collections.Counter(y_train_s)

Mod_1 = LogisticRegression(random_state=SEED,
                           multi_class='ovr').fit(X_train_pca_s, 
                                                  y_train_s.ravel())
Mod_1.score(X_train_pca_s, y_train_s.ravel())

y_pred1 = Mod_1.predict(X_test_pca)
print(classification_report(y_test, y_pred1))


class_names=[0,1,2,3,4] # name  of classes
cnf_matrix=confusion_matrix(y_test,  y_pred1,class_names)

fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))


# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")

ax.set_xticklabels(class_names)
ax.set_yticklabels(class_names)

plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

y_pred2 = Mod_1.predict(X_train_pca_s)
print(classification_report(y_train_s, y_pred2))


class_names=[0,1,2,3,4] # name  of classes
cnf_matrix=confusion_matrix(y_train_s,y_pred2,class_names)

fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))


# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")

ax.set_xticklabels(class_names)
ax.set_yticklabels(class_names)

plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()