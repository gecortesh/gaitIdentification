#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 11:18:10 2018

@author: gabych
"""

import numpy as np
from sklearn import svm
import vicon_reader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import column_or_1d
from sklearn.model_selection import cross_val_score
from sklearn import cross_validation

# subjects and experiment
experiment = "Level"
subject_1 = "Kai"
subject_2 = "Jaime"

# input = [n_samples, n_features]  1 sample = gait cycle ROM, feature = knee angle, ankle angle ...
r_knee_rom1, l_knee_rom1, r_hip_rom1, l_hip_rom1, r_ankle_rom1, l_ankle_rom1, trunk_rom1 = vicon_reader.feature_vector(experiment, subject_1)
r_knee_rom2, l_knee_rom2, r_hip_rom2, l_hip_rom2, r_ankle_rom2, l_ankle_rom2, trunk_rom2 = vicon_reader.feature_vector(experiment, subject_2)
x1 = np.hstack([r_knee_rom1, l_knee_rom1, r_hip_rom1, l_hip_rom1, r_ankle_rom1, l_ankle_rom1, trunk_rom1])
x2 =  np.hstack([r_knee_rom2, l_knee_rom2, r_hip_rom2, l_hip_rom2, r_ankle_rom2, l_ankle_rom2, trunk_rom2])
x = np.vstack([x1 , x2])
y = np.vstack([np.zeros((len(x1),1)), np.ones((len(x2),1))])

# shuffle
indices = np.arange(x.shape[0])
np.random.shuffle(indices)
x = x[indices]
y = y[indices]

# normal svm
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)
y_test = column_or_1d(y_test, warn=True)
y_train = column_or_1d(y_train, warn=True)
clf = svm.SVC(kernel='linear', C=1)
clf.fit(X_train,y_train)
clf.score(X_test, y_test)

# 5-fold cross validation
y = column_or_1d(y, warn=True)
scores = cross_val_score(clf, x, y, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# loo cross validation
loo = cross_validation.LeaveOneOut(5)
clf2 = svm.SVC(kernel='linear', C=1)
for train_index, test_index in loo:
    score = clf2.fit(x[train_index], y[train_index]).score(x[test_index], y[test_index])
    print('Sample %d score: %f' % (test_index[0], score))