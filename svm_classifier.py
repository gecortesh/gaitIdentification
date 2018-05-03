#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 11:18:10 2018

@author: gabych
"""

import numpy as np
from sklearn import svm
import vicon_reader
from sklearn.model_selection import train_test_split, cross_val_score, LeaveOneOut, KFold
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# subjects and experiment
experiment = "Level"
subjects = ['CA40B', 'LA40B', 'FN20D', 'ML20B', 'TM20F', 'LS20N', 'KA40S', 'AA40H', 'KI20S', 'JE20D', 'MN20G', 'GB20K', 'ME40G']
weights = [50,60,75,68,68,85,59,56,76,82,86,85,58]
heights = [169.4, 163, 177.5, 186, 181.5,195,161.8,174,180.8,173.5,188,194,169]

# input = [n_samples, n_features]  1 sample = gait cycle ROM, feature = knee angle, ankle angle ...
x = np.array([])
#x = np.concatenate([vicon_reader.feature_vector(experiment, subjects[s],weights[s], heights[s]) for s in range(len(subjects))],  axis=0) 
x = np.concatenate([vicon_reader.feature_vector_angles(experiment, subjects[s]) for s in range(len(subjects))],  axis=0) 
y = np.concatenate([vicon_reader.y_vector(experiment, subjects[l], l) for l in range (len(subjects))], axis=0)

# shuffle
indices = np.arange(x.shape[0])
np.random.shuffle(indices)
x = x[indices]
y = y[indices]

# train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# OVO  rbf obtain the best accuracy
clf = svm.SVC(kernel='rbf', decision_function_shape='ovo')
clf = clf.fit(x_train,y_train)
clf.score(x_test, y_test)
  
# 5-fold cross validation
kfold = KFold(n_splits=5)
scores = cross_val_score(clf, x, y, cv=kfold)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))


# loo cross validation
#loocv = LeaveOneOut()
#scores = cross_val_score(clf, x, y, cv=loocv)
#np.save("scores",scores)
#print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean()*100, scores.std() * 100))

# Random forest feture selection
names = ['r_knee_rom', 'l_knee_rom', 'r_hip_rom', 'l_hip_rom', 'trunk_rom', 'r_ankle_rom', 'l_ankle_rom', 'weight', 'height']
rf = RandomForestClassifier(n_estimators=1000)
rf.fit(x_train, y_train)
print ("Features sorted by their score:")
print (sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), names), reverse=True))

# Select from model
selector = SelectFromModel(rf)
selector.fit(x_train, y_train)
features = selector.get_support([indices])
print ("Features sorted by their score:")
print ([names[f] for f in features])

x_reduced = x[:,features]
kfold = KFold(n_splits=5)
scores = cross_val_score(clf, x_reduced, y, cv=kfold)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))