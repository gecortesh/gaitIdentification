#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 15:14:58 2018

@author: gabych
"""
import numpy as np
import fastdtw

outarr = np.load('outarr.npy')

def distance_metric(ts1, ts2):
    return fastdtw.fastdtw(ts1, ts2)[0]

p = len(outarr[0])
distances_s = {}
left = np.arange(0,len(outarr))
for subject in range(0, len(outarr)):
    for subject2 in left:
        if subject != subject2:
            distances = []
            for w in range(0, np.shape(outarr)[1]):
                distance = distance_metric(np.reshape(outarr[subject][w],1), np.reshape(outarr[subject2][w],1))
                distances.append(distance)
            distances_s[str(subject)+'-'+str(subject2)] = {}
            distances_s[str(subject)+'-'+str(subject2)]['Distances'] = distances
    left = np.delete(left, np.where(left==subject))
    
np.save('distances',distances_s)
