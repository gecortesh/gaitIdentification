#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  2 17:16:22 2018

@author: gabych
"""

import numpy as np
from sklearn import svm
import vicon_reader
from sklearn.model_selection import train_test_split

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
size = x.shape[1]

# initial popultation 
def population(size, n_indviduals):
    population = np.zeros((n_indviduals,size))
    for i in range(0,len(n_indviduals)):
        individual =  np.random.randint(2, size=size)
        population[i]=individual
    return population
    
# fitness function is accuracy obtained in classification
def fitness_function(x, y):
    # train-test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    # OVO  rbf obtain the best accuracy
    clf = svm.SVC(kernel='rbf', decision_function_shape='ovo')
    clf = clf.fit(x_train,y_train)
    return clf.score(x_test, y_test)

# selection is tournament selection
def tournament_selection(population, fitnesses):
    
    
    
