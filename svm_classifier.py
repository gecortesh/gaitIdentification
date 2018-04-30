#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 11:18:10 2018

@author: gabych
"""

import numpy as np
from sklearn import svm
import vicon_reader
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import column_or_1d
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn import cross_validation
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# subjects and experiment
experiment = "Level"
subjects = ['CA40B', 'LA40B', 'FN20D', 'ML20B', 'TM20F', 'LS20N', 'KA40S', 'AA40H', 'KI20S', 'JE20D', 'MN20G', 'GB20K', 'ME40G']

# input = [n_samples, n_features]  1 sample = gait cycle ROM, feature = knee angle, ankle angle ...
x = np.array([])
x = np.concatenate([vicon_reader.feature_vector(experiment, subjects[s]) for s in range(len(subjects))],  axis=0) 
y = np.concatenate([vicon_reader.y_vector(experiment, subjects[l], l) for l in range (len(subjects))], axis=0)

# shuffle
indices = np.arange(x.shape[0])
np.random.shuffle(indices)
x = x[indices]
y = y[indices]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# OVO
clf = svm.SVC(decision_function_shape='ovo')
#clf.fit(X_train,y_train)
#clf.score(X_test, y_test)
  
# 5-fold cross validation
#scores = cross_val_score(clf, x, y, cv=5)
#print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# loo cross validation
loocv = LeaveOneOut()
scores = cross_val_score(clf, x, y, cv=loocv)
np.save("scores",scores)
#print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean()*100, scores.std() * 100))
