#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 20:51:05 2019

@author: hosni
"""
from sklearn import svm
import pandas as pd
from pydataset import data
from sklearn import decomposition
import numpy as np
from sklearn import svm
from  sklearn import multiclass
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from PIL import Image 
import os

Datam=[]

dirs=os.listdir("/home/hosni/Downloads/faces/")
for ligne in dirs:
    if('.pgm' not in ligne):
        im=Image.open("/home/hosni/Downloads/faces/"+ligne)
        data=list(im.getdata())
        trans=np.transpose(data)
        Datam.append(trans)

y=[]
j=0
for i in range(0,15):
    for i1 in range(0,11):
        y.append(j)
    j=j+1
    
pca=decomposition.PCA(n_components=165)
x_pca=pca.fit_transform(Datam)

x_train,x_test,y_train,y_test=train_test_split(Datam,y,test_size=0.33)
clf=svm.SVC(kernel='linear',C=1,decision_function_shape='OVR')
clf.fit(x_train,y_train)
predicted=clf.predict(x_test)
acc=accuracy_score(y_test,predicted)
print("accuracy=",acc)