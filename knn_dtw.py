#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 11 13:36:22 2018

@author: gabych
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial.distance import euclidean
import fastdtw
import vicon_reader_lap
import matplotlib.pyplot as plt


def distances(joint, subjects, gaits):
    distances_s = {}
    left = np.arange(0,len(subjects))
    for subject in range(0, len(subjects)):
        for subject2 in range(0, len(left)):
            if subject != subject2:
                r = np.min((len(gaits[subject]),len(gaits[subject2])))
                distances = []
                for w in range(0, r-1):
                    distance, path = fastdtw.fastdtw(joint[subject][gaits[subject][w]:gaits[subject][w+1]],joint[subject2][gaits[1][subject2]:gaits[subject2][w+1]])
                    distances.append(distance)
                distances_s[str(subject)+'-'+str(subject2)] = distances
        left = np.delete(left, subject)
    return distances_s
    
    
def distance_metric(ts1, ts2):
    return fastdtw.fastdtw(ts1, ts2)[0]


def treshold(dictionary):
    means = []
    stds =[]
    for key, val in dictionary.items():
        means.append(np.mean(val))
        stds.append(np.std(val))
    mean = np.mean(means)
    std = np.std(stds)
    return [mean-std, mean+std]

experiment = "Level"
subjects = ['CA40B', 'LA40B', 'FN20D', 'ML20B', 'TM20F', 'LS20N', 'KA40S', 'AA40H', 'KI20S', 'JE20D', 'MN20G', 'GB20K', 'ME40G']
x = []
vgrfs = []
fzls = []
fzrs = []
gaitsl =[]
gaitsr =[]
r_knee_s = [] 
l_knee_s = [] 
r_hip_s = []
l_hip_s = []
r_ankle_s = []
l_ankle_s = []
trunk_s = []


for s in range(0,len(subjects)):
    vgrf, points =  vicon_reader_lap.read_file(experiment, subjects[s])
    vgrfs.append(vgrf)
    a_r_knee_angle, a_l_knee_angle, r_hip_angle, l_hip_angle, r_ankle_angle, l_ankle_angle, trunk_angle = vicon_reader_lap.kinematics(points)
    gait_cycle_l, gait_cycle_r, Fz_l, Fz_r = vicon_reader_lap.gait_cycles(vgrf, False, False)
    gaitsl.append(gait_cycle_l)
    gaitsr.append(gait_cycle_r)
    fzls.append(Fz_l)
    fzrs.append(Fz_r)
    r_knee_s.append(a_r_knee_angle)
    l_knee_s.append(a_l_knee_angle)
    r_hip_s.append(r_hip_angle)
    l_hip_s.append(l_hip_angle)
    r_ankle_s.append(r_ankle_angle)
    l_ankle_s.append(l_ankle_angle)
    trunk_s.append(trunk_angle)

th_rknee = treshold(distances(r_knee_s,subjects,gaitsr))
th_lknee = treshold(distances(l_knee_s,subjects,gaitsl))
th_rhip = treshold(distances(r_hip_s,subjects,gaitsr))
th_lhip = treshold(distances(l_hip_s,subjects,gaitsl))
th_rank = treshold(distances(r_ankle_s,subjects,gaitsr))
th_lank = treshold(distances(l_ankle_s,subjects,gaitsl))
th_trunk = treshold(distances(trunk_s,subjects,gaitsr))

np.save('th_rknee', th_rknee)
np.save('th_lknee',th_lknee)
np.save('th_rhip', th_rhip)
np.save('th_lhip', th_lhip)
np.save('th_rank',th_rank)
np.save('th_lank', th_lank)
np.save('th_trunk', th_trunk)