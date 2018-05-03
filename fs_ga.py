#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  2 17:16:22 2018

@author: gabych
"""

import numpy as np
from sklearn import svm
import vicon_reader
from sklearn.model_selection import cross_val_score, KFold, train_test_split

clf, x, y = svm_clf()
pop_size = 100
generations = 1
pop = population(pop_size, x.shape[1])

for g in range(0,generations):
    fitnesses = population_fitness(pop, x, y, clf)
    elite = elitism(fitnesses)
    pop[0] = pop[elite]
    y[0]= y[elite]



# initial popultation 
def population(n_indviduals, size):
    population = np.zeros((n_indviduals,size))
    for i in range(0,n_indviduals):
        individual =  np.random.randint(2, size=size)
        population[i]=individual
    return population
    
# fitness function is accuracy obtained in classification
def fitness_function(clf, x_reduced, y):
    kfold = KFold(n_splits=5)
    scores = cross_val_score(clf, x_reduced, y, cv=kfold)
    return scores.mean()

def population_fitness(pop, x, y, clf):
    fitnesses = np.zeros((len(pop),1))
    for p in range(0,len(pop)):
        x_index = np.nonzero(pop[p])
        x_i = x[:, x_index[0]]
        fitness = fitness_function(clf, x_i, y)
        fitnesses[p] = fitness
    return fitnesses

# selection is tournament selection
def tournament_selection(population, fitnesses):
    ind1 = np.random.randint(0,len(population))
    ind2 = np.random.randint(0,len(population))
    if fitnesses[ind1] >= fitnesses[ind2]:
        winner = population[ind1]
    else:
        winner = population[ind2]
    return winner
    
# crossover method is random point 
def crossover(population):
    parent1 = np.random.randint(0,len(population))
    parent2 = np.random.randint(0,len(population))
    child = np.zeros(np.shape(parent1))
    crossover_point = np.rnp.random.randint(0, parent1.shape[1])
    child[:crossover_point] = parent1[:crossover_point]
    child[crossover_point:] = parent2[crossover_point:]
    return child

# elitism is to save the indivual with the best performance to the next generation
def elitism(fitnesses):
    elite = np.argmax(fitnesses)
    return elite
    
def svm_clf():
    # subjects and experiment
    experiment = "Level"
    subjects = ['CA40B', 'LA40B', 'FN20D', 'ML20B', 'TM20F', 'LS20N', 'KA40S', 'AA40H', 'KI20S', 'JE20D', 'MN20G', 'GB20K', 'ME40G']
    x = np.array([])
    #x = np.concatenate([vicon_reader.feature_vector(experiment, subjects[s],weights[s], heights[s]) for s in range(len(subjects))],  axis=0) 
    x = np.concatenate([vicon_reader.feature_vector_angles(experiment, subjects[s]) for s in range(len(subjects))],  axis=0) 
    y = np.concatenate([vicon_reader.y_vector(experiment, subjects[l], l) for l in range (len(subjects))], axis=0)
    
    # shuffle
    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)
    x = x[indices]
    y = y[indices]
    
    # classifier
    clf = svm.SVC(kernel='rbf', decision_function_shape='ovo')
    return clf, x, y