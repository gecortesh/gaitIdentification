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

# variable initit
clf, x, y = svm_clf()
pop_size = 100
generations = 100
pop = population(pop_size, x.shape[1])
crossover_rate = 0.70 # best results between 0.65-.85
mutation_rate = 0.001
mean_fitness_g = np.zeros((generations,1))
median_fitness_g = np.zeros((generations,1))
elite_g =  np.zeros((generations,x.shape[1]))

# main loop0
for g in range(0,generations):
    fitnesses = population_fitness(pop, x, y, clf)
    elite = elitism(fitnesses)
    pop[0] = pop[elite]
    y[0]= y[elite]
    elite_g[g,:] = pop[elite]
    pop = crossover_pop(pop, fitnesses, crossover_rate)
    pop = mutate_pop(pop, mutation_rate)
    mean_fitness_g[g] = np.mean(fitnesses)
    median_fitness_g[g] =  np.median(fitnesses)
    
np.save('median_fit',median_fitness_g)
np.save('mean_fit',mean_fitness_g)

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
def crossover(pop, fitnesses):
    parent1 = tournament_selection(pop, fitnesses)
    parent2 = tournament_selection(pop, fitnesses)
    child = np.zeros(np.shape(parent1))
    crossover_point = np.random.randint(0, len(parent1))
    child[:crossover_point] = parent1[:crossover_point]
    child[crossover_point:] = parent2[crossover_point:]
    return child

# apply crossover in all population
def crossover_pop(pop, fitnesses, crossover_rate):
    for i in range(1,len(pop)):
        if np.random.random() <= crossover_rate:
            pop[i] = crossover(pop,fitnesses)
    return pop

# single point mutation
def mutation(individual):
    point = np.random.randint(0, len(individual))
    if individual[point] == 1:
        individual[point] = 0
    else:
        individual[point] = 1
    return individual

# apply mutation over all population
def mutate_pop(pop, mutation_rate):
    for i in range(1,len(pop)):
        if np.random.random() <= mutation_rate:
            pop[i] = mutation(pop[i])
    return pop

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