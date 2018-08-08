#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 29 11:18:50 2018

@author: gabych
"""
import numpy as np
from scipy.spatial.distance import euclidean
import fastdtw
import c3d_test
import btk
import matplotlib.animation as animation
import matplotlib.pyplot as plt

def read_files(subject):
    reader_c3d = c3d_test.Reader(open(subject,'rb'))
    
    if hasattr(reader_c3d,'subjects_prefixes'):
        prefix = reader_c3d.subjects_used - 1      
        label_prefix = reader_c3d.subjects_prefixes[prefix].strip()
        labels = reader_c3d.point_labels
        labels =  [x.replace(label_prefix," ") for x in labels]
        labels = list(map(str.strip, labels))
    else:
        labels = reader_c3d.point_labels
        labels = list(map(str.strip, labels))
    points = [] 
    for i, point, analog in reader_c3d.read_frames():
        points.append(point)
    points_array = np.array(points)   
     
    reader_btk = btk.btkAcquisitionFileReader() # build a btk reader object
    reader_btk.SetFilename(subject) # set a filename to the reader
    reader_btk.Update()
    acq = reader_btk.GetOutput() # acq is the btk aquisition object
    events = {}
    n_events = acq.GetEventNumber() # return number of events in file
    events_frames = []
    for e in range(n_events):
        event = acq.GetEvent(e) # extract the first event of the aquisition
        events[e]={}
        events[e]['Frame'] = event.GetFrame() # return the frame as an integer
        events[e]['Label'] = event.GetLabel() # return a string representing the Label
        events[e]['Context'] =  event.GetContext() # return a string representing the Context
        events[e]['Time'] = event.GetTime()
        events_frames.append([e,  event.GetFrame()])
    return points_array, events, labels

def markers_coordinates_balgrist(points, labels):
    # saving markers coordinates per element for joint angle calculation
    L_TRC = points[:,labels.index('LASI'),0:3] # left hip
    R_TRC = points[:,labels.index('RASI'),0:3] # right hip
    C_TRC = (points[:,labels.index('RASI'),0:3] + points[:,labels.index('LASI'),0:3])/2 # center of hip (check with )
    COM = points[:,labels.index('SACR'),0:3] # center of mass
    R_KNE = points[:,labels.index('RKNE'),0:3] # right knee
    L_KNE = points[:,labels.index('LKNE'),0:3] # left knee
    if 'RANK' in labels:
        R_ANK = points[:,labels.index('RANK'),0:3] # right ankle
    else:
        R_ANK = points[:,labels.index('RANM'),0:3] # right ankle
    L_ANK = points[:,labels.index('LANK'),0:3] # left ankle
    if 'RT3' in labels:
        R_UIM = points[:,labels.index('RT3'),0:3] # right upper IMU
    else:
        R_UIM = points[:,labels.index('RTHI'),0:3] # right upper IMU
    if 'LT3' in labels:
        L_UIM = points[:,labels.index('LT3'),0:3] # left upper IMU
    else:
        L_UIM = points[:,labels.index('LTHI'),0:3] # left upper IMU
    if 'RS3' in labels:
        R_LIM = points[:,labels.index('RS3'),0:3] # right lower IMU
    else:
        R_LIM = points[:,labels.index('RTIB'),0:3] # right lower IMU
    if 'LS3' in labels:
        L_LIM = points[:,labels.index('LS3'),0:3] # left lower IMU
    else:
        L_LIM = points[:,labels.index('LTIB'),0:3] # left lower IMU
    if 'RMT5' in labels:
        R_MT5 = points[:,labels.index('RMT5'),0:3] # right foot
    else:
        R_MT5 = points[:,labels.index('RTOE'),0:3] # right foot
    if 'LMT5' in labels:
        L_MT5 = points[:,labels.index('LMT5'),0:3] # left foot   
    else:
        L_MT5 = points[:,labels.index('LTOE'),0:3] # left foot   
    return COM, C_TRC, R_TRC, L_TRC, R_UIM, L_UIM, R_LIM, L_LIM, R_ANK, L_ANK, R_MT5, L_MT5, R_KNE, L_KNE

def kinematics(points, labels, angles):
    # angle calculation in sagital plane (l-left, r-right)
    COM, C_TRC, R_TRC, L_TRC, R_UIM, L_UIM, R_LIM, L_LIM, R_ANK, L_ANK, R_MT5, L_MT5, R_KNE, L_KNE = markers_coordinates_balgrist(points, labels)
    
    if angles:
        trunk_angle = np.rad2deg(np.arctan2((COM[:,2]-C_TRC[:,2]),(COM[:,1]-C_TRC[:,1])))
        r_knee_angle = points[:,labels.index('RKneeAngles'),0]
        l_knee_angle = points[:,labels.index('LKneeAngles'),0]
        r_hip_angle = points[:,labels.index('RHipAngles'),0]
        l_hip_angle = points[:,labels.index('LHipAngles'),0]
        r_ankle_angle = points[:,labels.index('RAnkleAngles'),0]
        l_ankle_angle = points[:,labels.index('LAnkleAngles'),0]       
    else:
        l_shank_angle = np.rad2deg(np.arctan2((L_KNE[:,2]-L_ANK[:,2]), (L_KNE[:,1]-L_ANK[:,1]))) #+ 360) % 360 # % (2 * np.pi)
        r_shank_angle = np.rad2deg(np.arctan2((R_KNE[:,2]-R_ANK[:,2]), (R_KNE[:,1]-R_ANK[:,1])))
        #r_shank_angle_v = np.rad2deg(np.arctan2((R_ANK[:,2]-R_KNE[:,2]),(R_ANK[:,1]-R_KNE[:,1])))
        #l_shank_angle_v = np.rad2deg(np.arctan2((L_ANK[:,2]-L_KNE[:,2]),(L_ANK[:,1]-L_KNE[:,1]))) #+ 360) % 360
        l_thigh_angle = np.rad2deg(np.arctan2((L_UIM[:,2]-L_KNE[:,2]), (L_UIM[:,1]-L_KNE[:,1]))) #+ 360) % 360
        r_thigh_angle = np.rad2deg(np.arctan2((R_UIM[:,2]-R_KNE[:,2]), (R_UIM[:,1]-R_KNE[:,1])))
        #l_thigh_angle_v = np.rad2deg(np.arctan2((L_KNE[:,2]-L_TRC[:,2]), (L_KNE[:,1]-L_TRC[:,1])))# + 360) % 360
        #r_thigh_angle_v = np.rad2deg(np.arctan2((R_KNE[:,2]-R_TRC[:,2]), (R_KNE[:,1]-R_TRC[:,1])))# + 360) % 360
        trunk_angle = np.rad2deg(np.arctan2((COM[:,2]-C_TRC[:,2]),(COM[:,1]-C_TRC[:,1]))) #+ 360) % 360
        r_foot_angle = np.rad2deg(np.arctan2((R_ANK[:,2]-R_MT5[:,2]),(R_ANK[:,1]-R_MT5[:,1]))) #+ 360) % 360
        l_foot_angle = np.rad2deg(np.arctan2((L_ANK[:,2]-L_MT5[:,2]),(L_ANK[:,1]-L_MT5[:,1]))) #+ 360) % 360
        r_knee_angle = (r_thigh_angle - r_shank_angle)
        l_knee_angle = (l_thigh_angle - l_shank_angle)
        #b_r_knee_angle = (r_shank_angle + (180-r_thigh_angle))
        #b_l_knee_angle = (l_shank_angle + (180-l_thigh_angle))
        r_hip_angle = (r_thigh_angle - trunk_angle)
        l_hip_angle = (l_thigh_angle - trunk_angle)
        r_ankle_angle = (r_foot_angle - r_shank_angle)-90
        l_ankle_angle = (l_foot_angle - l_shank_angle)-90   
        
    r_knee = standarization(r_knee_angle).reshape(len(r_knee_angle),1)
    l_knee = standarization(l_knee_angle).reshape(len(l_knee_angle),1)
    r_hip = standarization(r_hip_angle).reshape(len(r_hip_angle),1)
    l_hip = standarization(l_hip_angle).reshape(len(l_hip_angle),1)
    r_ankle = standarization(r_ankle_angle).reshape(len(r_ankle_angle),1)
    l_ankle = standarization(l_ankle_angle).reshape(len(l_ankle_angle),1)
    trunk = standarization(trunk_angle).reshape(len(trunk_angle),1)
    
    r_knee = normalization(r_knee)
    l_knee = normalization(l_knee)
    r_hip = normalization(r_hip)
    l_hip = normalization(l_hip)
    r_ankle = normalization(r_ankle)
    l_ankle = normalization(l_ankle)
    trunk = normalization(trunk)
    
    return r_knee, l_knee, r_hip, l_hip, r_ankle, l_ankle, trunk

def cycles(events, cycles):

    gait_cycles_right = np.sort([events[e]['Frame'] for e in events if events[e]['Context']=='Right'])
    right_foot_strike = np.sort([events[e]['Frame'] for e in events if events[e]['Context']=='Right' and 'Strike' in events[e]['Label']])
    right_foot_off = np.sort([events[e]['Frame'] for e in events if events[e]['Context']=='Right' and 'Off' in events[e]['Label']])
    
    gait_cycles_left = np.sort([events[e]['Frame'] for e in events if events[e]['Context']=='Left'])
    left_foot_strike = np.sort([events[e]['Frame'] for e in events if events[e]['Context']=='Left' and 'Strike' in events[e]['Label']])
    left_foot_off = np.sort([events[e]['Frame'] for e in events if events[e]['Context']=='Left' and 'Off' in events[e]['Label']])
    
    if cycles:
        return right_foot_off, right_foot_strike, left_foot_off, left_foot_strike, gait_cycles_left, gait_cycles_right
    else:
        return right_foot_off, right_foot_strike, left_foot_off, left_foot_strike

def distances(joint, subjects):
    distances_s = {}
    left = np.arange(0,len(subjects))
    for subject in range(0, len(subjects)):
        for subject2 in left:
            if subject != subject2:
                r = np.min((len(joint[subject]),len(joint[subject2])))
                distances = []
                for w in range(0, r):
                    distance = distance_metric(joint[subject][w],joint[subject2][w])
                    distances.append(distance)
                distances_s[str(subject)+'-'+str(subject2)] = {}
                distances_s[str(subject)+'-'+str(subject2)]['Distances'] = distances
                distances_s[str(subject)+'-'+str(subject2)]['NSamples'] = r
        left = np.delete(left, np.where(left==subject))
    return distances_s

def distance_metric(ts1, ts2):
    return fastdtw.fastdtw(ts1, ts2)[0]


def treshold(distances):
    means = []
    stds =[]
    samples = []
    for key, val in distances.items():
        means.append(distances[key]['NSamples']*np.mean(distances[key]['Distances']))
        stds.append((distances[key]['NSamples']-1)*np.std(distances[key]['Distances']))
        samples.append(distances[key]['NSamples'])
    pooled_mean = sum(means)/sum(samples)
    pooled_std = sum(stds)/(sum(samples)-len(samples))
    return [pooled_mean-pooled_std, pooled_mean+pooled_std]

def animation_markers(points_array):
    y = points_array[:,:,1]
    z = points_array[:,:,2]
    # Animation
    def animate(i):
        scat.set_offsets(np.c_[y[i],z[i]])
        return scat
    
    fig, ax =  plt.subplots()
    scat = ax.scatter(y[0], z[0], c = y[0])
    
    plt.ylim(0,1200)
    anim = animation.FuncAnimation(fig, animate, interval=5)
    plt.show()
    
def standarization(data):
    x = (data - np.mean(data)) / np.std(data)
    return x

def normalization(data):
    x = (data - np.min(data)) / (np.max(data)-np.min(data))
    return x

def gait_phases_perc(cycle):
    # divide each gait cycle in phases
    total = len(cycle)
    LR = cycle[:round(total*.1)]
    MST = cycle[round(total*.1): round(total*.3)]
    TST = cycle[round(total*.3): round(total*.5)]
    PSW = cycle[round(total*.5): round(total*.6)]
    ISW = cycle[round(total*.6): round(total*.73)]
    MSW = cycle[round(total*.73): round(total*.87)]
    TSW = cycle[round(total*.87):]
    phases = [LR,MST,TST,PSW,ISW,MSW, TSW]
    return phases

def average_signal(joint):
    av_signal=[]
    for s in range(0,len(joint)):
        for i in range(0,len(joint[s])):
            if len(joint[s][i]) > 255:
                av_signal.append(joint[s][i][:255])
    return np.mean(av_signal,axis=0)

def fuzzy_gait_phase_detect(joint,hs,to):
    
    return 0

def abnormality_detection(joint,hs, reference, threshold):
    ''' detect abnormalities in gait phases '''
    angle_gait = np.split(joint, hs)
    phases_reference = gait_phases_perc(reference)
    lower, upper = threshold
    abnormalities = {}
    for c in range(0,len(angle_gait)):
        phases_joint = gait_phases_perc(angle_gait[c])
        for p in range(0,len(phases_joint)):
            similarity = fastdtw.fastdtw(phases_joint[p],phases_reference[p])[0]
            if similarity > upper or similarity <  lower:
                abnormalities[c]= p
    return abnormalities
             
def gaf(cycle):
    gasf = GASF()
    im = gasf.fit_transform(cycle)
    return im
    #plt.imshow(X_gasf[0], cmap='rainbow', origin='lower')
    
def features_rom_cycle(subjects, use_angles):
    data = []
    data_labels = []
    for s in range(0,len(subjects)):
#        if "EventsAdded" in subjects[s]:
        points_array, events, labels = read_files(subjects[s])
#            has_angles = any('Angles' in string for string in labels)
        r_knee_angle, l_knee_angle, r_hip_angle, l_hip_angle, r_ankle_angle, l_ankle_angle, trunk_angle = kinematics(points_array, labels, angles = use_angles)
        right_foot_off, right_foot_strike, left_foot_off, left_foot_strike = cycles(events, cycles=False)         
        
        r_knee = np.split(r_knee_angle, right_foot_strike)
        l_knee = np.split(l_knee_angle, left_foot_strike)           
        r_hip = np.split(r_hip_angle, right_foot_strike)
        l_hip = np.split(l_hip_angle, left_foot_strike)            
        r_ankle = np.split(r_ankle_angle, right_foot_strike)
        l_ankle = np.split(l_ankle_angle, left_foot_strike)            
        trunk = np.split(trunk_angle, right_foot_strike)            
        
        rom_rknee = [np.max(r_knee[s])-np.min(r_knee[s]) for s in range(1,len(r_knee)-1)]
        rom_lknee = [np.max(l_knee[s])-np.min(l_knee[s]) for s in range(1,len(l_knee)-1)]
        rom_rhip = [np.max(r_hip[s])-np.min(r_hip[s]) for s in range(1,len(r_hip)-1)]
        rom_lhip = [np.max(l_hip[s])-np.min(l_hip[s]) for s in range(1,len(l_hip)-1)]
        rom_rank = [np.max(r_ankle[s])-np.min(r_ankle[s]) for s in range(1,len(r_ankle)-1)]
        rom_lank = [np.max(l_ankle[s])-np.min(l_ankle[s]) for s in range(1,len(l_ankle)-1)]
        rom_trunk = [np.max(trunk[s])-np.min(trunk[s]) for s in range(1,len(trunk)-1)]
        
        n_cycles = min(len(right_foot_strike), len(left_foot_strike))-1
        features = np.vstack([rom_rknee[:n_cycles], rom_lknee[:n_cycles], rom_rhip[:n_cycles], rom_lhip[:n_cycles],
                              rom_rank[:n_cycles], rom_lank[:n_cycles], rom_trunk[:n_cycles]]).T
        
        data.append(features)
        data_labels.append(np.ones(len(data[s]))*s)
    x = np.vstack((data))
    y = np.hstack((data_labels))
    return x, y

def features_rom_swst(subjects, calculate_angles):
    data = []
    data_labels = []
    for s in range(0,len(subjects)):
#        if "EventsAdded" in subjects[s]:
        points_array, events, labels = read_files(subjects[s])
#            has_angles = any('Angles' in string for string in labels)
        r_knee_angle, l_knee_angle, r_hip_angle, l_hip_angle, r_ankle_angle, l_ankle_angle, trunk_angle = kinematics(points_array, labels, angles = calculate_angles)
        right_foot_off, right_foot_strike, left_foot_off, left_foot_strike = cycles(events, cycles=False)
        right = np.sort(np.hstack([right_foot_strike, right_foot_off]))
        left = np.sort(np.hstack([left_foot_strike, left_foot_off]))
        
        if np.max(right) > len(r_knee_angle):
            right = right[:np.max(np.where(right <= len(r_knee_angle)))]
        if np.max(left) > len(l_knee_angle):
            left = left[:np.max(np.where(left <= len(l_knee_angle)))]
        
        r_knee_sw = np.split(r_knee_angle, right)[::2]
        l_knee_sw = np.split(l_knee_angle, left)[::2]           
        r_hip_sw = np.split(r_hip_angle, right)[::2]
        l_hip_sw = np.split(l_hip_angle, left)[::2]            
        r_ankle_sw = np.split(r_ankle_angle, right)[::2]
        l_ankle_sw = np.split(l_ankle_angle, left)[::2]            
        trunk_sw = np.split(trunk_angle, right)[::2]            
        
        r_knee_st = np.split(r_knee_angle, right)[1::2]
        l_knee_st = np.split(l_knee_angle, left)[1::2]           
        r_hip_st = np.split(r_hip_angle, right)[1::2]
        l_hip_st = np.split(l_hip_angle, left)[1::2]            
        r_ankle_st = np.split(r_ankle_angle, right)[1::2]
        l_ankle_st = np.split(l_ankle_angle, left)[1::2]            
        trunk_st = np.split(trunk_angle, right)[1::2]            
        
        rom_rknee_sw = [np.max(r_knee_sw[s])-np.min(r_knee_sw[s]) for s in range(1,len(r_knee_sw)-1)]
        rom_lknee_sw = [np.max(l_knee_sw[s])-np.min(l_knee_sw[s]) for s in range(1,len(l_knee_sw)-1)]
        rom_rhip_sw = [np.max(r_hip_sw[s])-np.min(r_hip_sw[s]) for s in range(1,len(r_hip_sw)-1)]
        rom_lhip_sw = [np.max(l_hip_sw[s])-np.min(l_hip_sw[s]) for s in range(1,len(l_hip_sw)-1)]
        rom_rank_sw = [np.max(r_ankle_sw[s])-np.min(r_ankle_sw[s]) for s in range(1,len(r_ankle_sw)-1)]
        rom_lank_sw = [np.max(l_ankle_sw[s])-np.min(l_ankle_sw[s]) for s in range(1,len(l_ankle_sw)-1)]
        rom_trunk_sw = [np.max(trunk_sw[s])-np.min(trunk_sw[s]) for s in range(1,len(trunk_sw)-1)]
        
        rom_rknee_st = [np.max(r_knee_st[s])-np.min(r_knee_st[s]) for s in range(1,len(r_knee_st)-1)]
        rom_lknee_st = [np.max(l_knee_st[s])-np.min(l_knee_st[s]) for s in range(1,len(l_knee_st)-1)]
        rom_rhip_st = [np.max(r_hip_st[s])-np.min(r_hip_st[s]) for s in range(1,len(r_hip_st)-1)]
        rom_lhip_st = [np.max(l_hip_st[s])-np.min(l_hip_st[s]) for s in range(1,len(l_hip_st)-1)]
        rom_rank_st = [np.max(r_ankle_st[s])-np.min(r_ankle_st[s]) for s in range(1,len(r_ankle_st)-1)]
        rom_lank_st = [np.max(l_ankle_st[s])-np.min(l_ankle_st[s]) for s in range(1,len(l_ankle_st)-1)]
        rom_trunk_st = [np.max(trunk_st[s])-np.min(trunk_st[s]) for s in range(1,len(trunk_st)-1)]
        
        n_cycles = min(len(rom_rknee_st), len(rom_rknee_sw))-1
        features = np.vstack([rom_rknee_sw[:n_cycles], rom_lknee_sw[:n_cycles], rom_rhip_sw[:n_cycles], rom_lhip_sw[:n_cycles], rom_rank_sw[:n_cycles], rom_lank_sw[:n_cycles], rom_trunk_sw[:n_cycles],
                              rom_rknee_st[:n_cycles], rom_lknee_st[:n_cycles], rom_rhip_st[:n_cycles], rom_lhip_st[:n_cycles], rom_rank_st[:n_cycles], rom_lank_st[:n_cycles], rom_trunk_st[:n_cycles]]).T
        
        data.append(features)
        data_labels.append(np.ones(len(data[s]))*s)
    x = np.vstack((data))
    y = np.hstack((data_labels))
    return x, y

def healthy_unhealthy_rom_swst(subjects, calculate_angles):
    data = []
    data_labels = []
    for s in range(0,len(subjects)):
        points_array, events, labels = read_files(subjects[s])
#            has_angles = any('Angles' in string for string in labels)
        r_knee_angle, l_knee_angle, r_hip_angle, l_hip_angle, r_ankle_angle, l_ankle_angle, trunk_angle = kinematics(points_array, labels, angles = calculate_angles)
        right_foot_off, right_foot_strike, left_foot_off, left_foot_strike = cycles(events, cycles=False)
        right = np.sort(np.hstack([right_foot_strike, right_foot_off]))
        left = np.sort(np.hstack([left_foot_strike, left_foot_off]))
        
        if np.max(right) > len(r_knee_angle):
            right = right[:np.max(np.where(right <= len(r_knee_angle)))]
        if np.max(left) > len(l_knee_angle):
            left = left[:np.max(np.where(left <= len(l_knee_angle)))]
        
        r_knee_sw = np.split(r_knee_angle, right)[::2]
        l_knee_sw = np.split(l_knee_angle, left)[::2]           
        r_hip_sw = np.split(r_hip_angle, right)[::2]
        l_hip_sw = np.split(l_hip_angle, left)[::2]            
        r_ankle_sw = np.split(r_ankle_angle, right)[::2]
        l_ankle_sw = np.split(l_ankle_angle, left)[::2]            
        trunk_sw = np.split(trunk_angle, right)[::2]            
        
        r_knee_st = np.split(r_knee_angle, right)[1::2]
        l_knee_st = np.split(l_knee_angle, left)[1::2]           
        r_hip_st = np.split(r_hip_angle, right)[1::2]
        l_hip_st = np.split(l_hip_angle, left)[1::2]            
        r_ankle_st = np.split(r_ankle_angle, right)[1::2]
        l_ankle_st = np.split(l_ankle_angle, left)[1::2]            
        trunk_st = np.split(trunk_angle, right)[1::2]            
        
        rom_rknee_sw = [np.max(r_knee_sw[i])-np.min(r_knee_sw[i]) for i in range(1,len(r_knee_sw)-1)]
        rom_lknee_sw = [np.max(l_knee_sw[i])-np.min(l_knee_sw[i]) for i in range(1,len(l_knee_sw)-1)]
        rom_rhip_sw = [np.max(r_hip_sw[i])-np.min(r_hip_sw[i]) for i in range(1,len(r_hip_sw)-1)]
        rom_lhip_sw = [np.max(l_hip_sw[i])-np.min(l_hip_sw[i]) for i in range(1,len(l_hip_sw)-1)]
        rom_rank_sw = [np.max(r_ankle_sw[i])-np.min(r_ankle_sw[i]) for i in range(1,len(r_ankle_sw)-1)]
        rom_lank_sw = [np.max(l_ankle_sw[i])-np.min(l_ankle_sw[i]) for i in range(1,len(l_ankle_sw)-1)]
        rom_trunk_sw = [np.max(trunk_sw[i])-np.min(trunk_sw[i]) for i in range(1,len(trunk_sw)-1)]
        
        rom_rknee_st = [np.max(r_knee_st[i])-np.min(r_knee_st[i]) for i in range(1,len(r_knee_st)-1)]
        rom_lknee_st = [np.max(l_knee_st[i])-np.min(l_knee_st[i]) for i in range(1,len(l_knee_st)-1)]
        rom_rhip_st = [np.max(r_hip_st[i])-np.min(r_hip_st[i]) for i in range(1,len(r_hip_st)-1)]
        rom_lhip_st = [np.max(l_hip_st[i])-np.min(l_hip_st[i]) for i in range(1,len(l_hip_st)-1)]
        rom_rank_st = [np.max(r_ankle_st[i])-np.min(r_ankle_st[i]) for i in range(1,len(r_ankle_st)-1)]
        rom_lank_st = [np.max(l_ankle_st[i])-np.min(l_ankle_st[i]) for i in range(1,len(l_ankle_st)-1)]
        rom_trunk_st = [np.max(trunk_st[i])-np.min(trunk_st[i]) for i in range(1,len(trunk_st)-1)]
        
        n_cycles = min(len(rom_rknee_st), len(rom_rknee_sw))-1
        features = np.vstack([rom_rknee_sw[:n_cycles], rom_lknee_sw[:n_cycles], rom_rhip_sw[:n_cycles], rom_lhip_sw[:n_cycles], rom_rank_sw[:n_cycles], rom_lank_sw[:n_cycles], rom_trunk_sw[:n_cycles],
                              rom_rknee_st[:n_cycles], rom_lknee_st[:n_cycles], rom_rhip_st[:n_cycles], rom_lhip_st[:n_cycles], rom_rank_st[:n_cycles], rom_lank_st[:n_cycles], rom_trunk_st[:n_cycles]]).T
        
        data.append(features)
        if "SCI" in subjects[s]:
            s_type = 1
        else:
            s_type = 0
            
        data_labels.append(np.ones(len(data[s]))*s_type)
    x = np.vstack((data))
    y = np.hstack((data_labels))
    return x, y

def rescale(arr, factor):
    n = len(arr)
    return np.interp(np.linspace(0, n, round(factor*n)), np.arange(n), np.reshape(arr,n))

def rescaled_array(joint, fixed_length=0):
    lens = []
    for s in range(0,len(joint)):
        lens.append(np.max([len(ps) for ps in joint[s]]))
        
    samples = []
    for s in range(0,len(joint)):
        samples.append(len(joint[s]))
    total_samples = sum(samples)
        
    if fixed_length >0:
        outarr=np.ones((total_samples, fixed_length))
    else:
        outarr=np.ones((total_samples, max(lens)))
    
    n = 0
    for s in range(len(joint)):
        for d in range(len(joint[s])): 
            #populate columns
            if len(joint[s][d]) >0:
                if fixed_length >0:
                    factor = fixed_length/len(joint[s][d])
                else:
                    factor = max(lens)/len(joint[s][d])
                outarr[n,:]= rescale(joint[s][d], factor) 
                n= n+1  
    return outarr

def gait_processing(subjects, kinematics_type='basic', rescaled=False, length=0, subject_type='control'):
    '''
    subject_type = control, SCI, MD, MD, parkinson
    '''
    r_knee, l_knee, r_hip, l_hip, r_ankle, l_ankle, trunk, r_thigh, l_thigh, r_shank, l_shank = [], [], [], [], [], [], [], [], [], [], []

    for s in range(0,len(subjects)):
        points_array, events, labels = read_files(subjects[s])
        
        if kinematics_type == 'all':
            r_knee_angle, l_knee_angle, r_hip_angle, l_hip_angle, r_ankle_angle, l_ankle_angle, trunk_angle, r_thigh_angle, l_thigh_angle, r_shank_angle, l_shank_angle = all_kinematics(points_array, labels, n=True)
        else:
           r_knee_angle, l_knee_angle, r_hip_angle, l_hip_angle, r_ankle_angle, l_ankle_angle, trunk_angle = kinematics(points_array, labels, angles = False)
        right_foot_off, right_foot_strike, left_foot_off, left_foot_strike = cycles(events, cycles=False) 

        if np.max(right_foot_strike) > len(r_knee_angle):
            right_foot_strike = right_foot_strike[:np.max(np.where(right_foot_strike <= len(r_knee_angle)))]
        right_foot_strike = np.unique(right_foot_strike)
        if np.max(left_foot_strike) > len(l_knee_angle):
            left_foot_strike = left_foot_strike[:np.max(np.where(left_foot_strike <= len(l_knee_angle)))]
        left_foot_strike = np.unique(left_foot_strike)         
        
        if 'UZP024' in subjects[s]:
            right = np.copy(right_foot_strike)
            right_foot_strike = np.copy(left_foot_strike)
            left_foot_strike = right
        
        rk = np.split(r_knee_angle, right_foot_strike)
        lk = np.split(l_knee_angle, left_foot_strike)           
        rh = np.split(r_hip_angle, right_foot_strike)
        lh = np.split(l_hip_angle, left_foot_strike)            
        ra = np.split(r_ankle_angle, right_foot_strike)
        la = np.split(l_ankle_angle, left_foot_strike)            
        tk = np.split(trunk_angle, right_foot_strike) 
        if kinematics_type == 'all':
            rt = np.split(r_thigh_angle, right_foot_strike)
            lt = np.split(l_thigh_angle, left_foot_strike)
            rs = np.split(r_shank_angle, right_foot_strike)
            ls = np.split(l_shank_angle, left_foot_strike)
    
        if subject_type=='control':
            r_knee.append(rk[1:len(rk)-1])
            l_knee.append(lk[1:len(lk)-1])    
            r_hip.append(rh[1:len(rh)-1])
            l_hip.append(lh[1:len(lh)-1])       
            r_ankle.append(ra[1:len(ra)-1])
            l_ankle.append(la[1:len(la)-1])           
            trunk.append(tk[1:len(tk)-1])  
            if kinematics_type == 'all':
                r_thigh.append(rt[1:len(rt)-1])
                l_thigh.append(lt[1:len(lt)-1])
                r_shank.append(rs[1:len(rs)-1])
                l_shank.append(ls[1:len(ls)-1])
            
        if subject_type != 'control':
            if subject_type in subjects[s]:    
                r_knee.append(rk[1:len(rk)-1])
                l_knee.append(lk[1:len(lk)-1])    
                r_hip.append(rh[1:len(rh)-1])
                l_hip.append(lh[1:len(lh)-1])       
                r_ankle.append(ra[1:len(ra)-1])
                l_ankle.append(la[1:len(la)-1])           
                trunk.append(tk[1:len(tk)-1])
                if kinematics_type == 'all':
                    r_thigh.append(rt[1:len(rt)-1])
                    l_thigh.append(lt[1:len(lt)-1])
                    r_shank.append(rs[1:len(rs)-1])
                    l_shank.append(ls[1:len(ls)-1])
            
    if rescaled:
        if length > 0:
            r_knee_rs = rescaled_array(r_knee, fixed_length=length)
            l_knee_rs = rescaled_array(l_knee, fixed_length=length)
            r_hip_rs = rescaled_array(r_hip, fixed_length=length)
            l_hip_rs = rescaled_array(l_hip, fixed_length=length)
            r_ankle_rs = rescaled_array(r_ankle, fixed_length=length)
            l_ankle_rs = rescaled_array(l_ankle, fixed_length=length)
            trunk_rs = rescaled_array(trunk, fixed_length=length)            
            if kinematics_type == 'all':
                r_thigh_rs = rescaled_array(r_thigh, fixed_length=length)
                l_thigh_rs = rescaled_array(l_thigh, fixed_length=length)
                r_shank_rs = rescaled_array(r_shank, fixed_length=length)
                l_shank_rs = rescaled_array(l_shank, fixed_length=length)
                return r_knee_rs, l_knee_rs, r_hip_rs, l_hip_rs, r_ankle_rs, l_ankle_rs, trunk_rs, r_thigh_rs, l_thigh_rs, r_shank_rs, l_shank_rs
            else:
                return r_knee_rs, l_knee_rs, r_hip_rs, l_hip_rs, r_ankle_rs, l_ankle_rs, trunk_rs
        else:
            r_knee_rs = rescaled_array(r_knee)
            l_knee_rs = rescaled_array(l_knee, fixed_length=np.shape(r_knee_rs)[1])
            r_hip_rs = rescaled_array(r_hip, fixed_length=np.shape(r_knee_rs)[1])
            l_hip_rs = rescaled_array(l_hip, fixed_length=np.shape(r_knee_rs)[1])
            r_ankle_rs = rescaled_array(r_ankle, fixed_length=np.shape(r_knee_rs)[1])
            l_ankle_rs = rescaled_array(l_ankle, fixed_length=np.shape(r_knee_rs)[1])
            trunk_rs = rescaled_array(trunk, fixed_length=np.shape(r_knee_rs)[1]) 
            if kinematics_type == 'all':
                r_thigh_rs = rescaled_array(r_thigh, fixed_length=np.shape(r_knee_rs)[1])
                l_thigh_rs = rescaled_array(l_thigh, fixed_length=np.shape(r_knee_rs)[1])
                r_shank_rs = rescaled_array(r_shank, fixed_length=np.shape(r_knee_rs)[1])
                l_shank_rs = rescaled_array(l_shank, fixed_length=np.shape(r_knee_rs)[1])
                return r_knee_rs, l_knee_rs, r_hip_rs, l_hip_rs, r_ankle_rs, l_ankle_rs, trunk_rs, r_thigh_rs, l_thigh_rs, r_shank_rs, l_shank_rs
            else:
                return r_knee_rs, l_knee_rs, r_hip_rs, l_hip_rs, r_ankle_rs, l_ankle_rs, trunk_rs
    else:
        if kinematics_type == 'all':
            return r_knee, l_knee, r_hip, l_hip, r_ankle, l_ankle, trunk, r_thigh, l_thigh, r_shank, l_shank
        
        else:
            return r_knee, l_knee, r_hip, l_hip, r_ankle, l_ankle, trunk

def multiclass_rom_swst(subjects):
    data = []
    data_labels = []
    for s in range(0,len(subjects)):
        points_array, events, labels = read_files(subjects[s])
#            has_angles = any('Angles' in string for string in labels)
        r_knee_angle, l_knee_angle, r_hip_angle, l_hip_angle, r_ankle_angle, l_ankle_angle, trunk_angle = kinematics(points_array, labels, angles = False)
        right_foot_off, right_foot_strike, left_foot_off, left_foot_strike = cycles(events, cycles=False)
        right = np.sort(np.hstack([right_foot_strike, right_foot_off]))
        left = np.sort(np.hstack([left_foot_strike, left_foot_off]))
        
        if 'UZP039' in subjects[s]:
            r_knee_angle = r_knee_angle[:5500]
            l_knee_angle = l_knee_angle[:5500]
            r_hip_angle = r_hip_angle[:5500]
            l_hip_angle = l_hip_angle[:5500]
            r_ankle_angle = r_ankle_angle[:5500]
            l_ankle_angle = l_ankle_angle[:5500]
            trunk_angle = trunk_angle[:5500]
        
        if np.max(right) > len(r_knee_angle):
            right = right[:np.max(np.where(right <= len(r_knee_angle)))]
        right = np.unique(right)
        if np.max(left) > len(l_knee_angle):
            left = left[:np.max(np.where(left <= len(l_knee_angle)))]
        left = np.unique(left)
        
        if 'UZP024' in subjects[s]:
            right_foot = np.copy(right)
            right = np.copy(left)
            left = right_foot
        
        r_knee_sw = np.split(r_knee_angle, right)[::2]
        l_knee_sw = np.split(l_knee_angle, left)[::2]           
        r_hip_sw = np.split(r_hip_angle, right)[::2]
        l_hip_sw = np.split(l_hip_angle, left)[::2]            
        r_ankle_sw = np.split(r_ankle_angle, right)[::2]
        l_ankle_sw = np.split(l_ankle_angle, left)[::2]            
        trunk_sw = np.split(trunk_angle, right)[::2]            
        
        r_knee_st = np.split(r_knee_angle, right)[1::2]
        l_knee_st = np.split(l_knee_angle, left)[1::2]           
        r_hip_st = np.split(r_hip_angle, right)[1::2]
        l_hip_st = np.split(l_hip_angle, left)[1::2]            
        r_ankle_st = np.split(r_ankle_angle, right)[1::2]
        l_ankle_st = np.split(l_ankle_angle, left)[1::2]            
        trunk_st = np.split(trunk_angle, right)[1::2]            
        
        rom_rknee_sw = [np.max(r_knee_sw[i])-np.min(r_knee_sw[i]) for i in range(1,len(r_knee_sw)-1)]
        rom_lknee_sw = [np.max(l_knee_sw[i])-np.min(l_knee_sw[i]) for i in range(1,len(l_knee_sw)-1)]
        rom_rhip_sw = [np.max(r_hip_sw[i])-np.min(r_hip_sw[i]) for i in range(1,len(r_hip_sw)-1)]
        rom_lhip_sw = [np.max(l_hip_sw[i])-np.min(l_hip_sw[i]) for i in range(1,len(l_hip_sw)-1)]
        rom_rank_sw = [np.max(r_ankle_sw[i])-np.min(r_ankle_sw[i]) for i in range(1,len(r_ankle_sw)-1)]
        rom_lank_sw = [np.max(l_ankle_sw[i])-np.min(l_ankle_sw[i]) for i in range(1,len(l_ankle_sw)-1)]
        rom_trunk_sw = [np.max(trunk_sw[i])-np.min(trunk_sw[i]) for i in range(1,len(trunk_sw)-1)]
        
        rom_rknee_st = [np.max(r_knee_st[i])-np.min(r_knee_st[i]) for i in range(1,len(r_knee_st)-1)]
        rom_lknee_st = [np.max(l_knee_st[i])-np.min(l_knee_st[i]) for i in range(1,len(l_knee_st)-1)]
        rom_rhip_st = [np.max(r_hip_st[i])-np.min(r_hip_st[i]) for i in range(1,len(r_hip_st)-1)]
        rom_lhip_st = [np.max(l_hip_st[i])-np.min(l_hip_st[i]) for i in range(1,len(l_hip_st)-1)]
        rom_rank_st = [np.max(r_ankle_st[i])-np.min(r_ankle_st[i]) for i in range(1,len(r_ankle_st)-1)]
        rom_lank_st = [np.max(l_ankle_st[i])-np.min(l_ankle_st[i]) for i in range(1,len(l_ankle_st)-1)]
        rom_trunk_st = [np.max(trunk_st[i])-np.min(trunk_st[i]) for i in range(1,len(trunk_st)-1)]
        
        n_cycles = min(len(rom_rknee_st), len(rom_rknee_sw))-1
        features = np.vstack([rom_rknee_sw[:n_cycles], rom_lknee_sw[:n_cycles], rom_rhip_sw[:n_cycles], rom_lhip_sw[:n_cycles], rom_rank_sw[:n_cycles], rom_lank_sw[:n_cycles], rom_trunk_sw[:n_cycles],
                              rom_rknee_st[:n_cycles], rom_lknee_st[:n_cycles], rom_rhip_st[:n_cycles], rom_lhip_st[:n_cycles], rom_rank_st[:n_cycles], rom_lank_st[:n_cycles], rom_trunk_st[:n_cycles]]).T
        
        data.append(features)
        if "SCI" in subjects[s]:
            s_type = 1
        elif "MD" in subjects[s]:
            s_type = 2
        elif "MS" in subjects[s]:
            s_type = 3
        elif "parkinson" in subjects[s]:
            s_type = 4
        else:
            s_type = 0
            
        data_labels.append(np.ones(len(data[s]))*s_type)
    x = np.vstack((data))
    y = np.hstack((data_labels))
    return x, y
            
def all_kinematics(points, labels, n):
    # angle calculation in sagital plane (l-left, r-right)
    COM, C_TRC, R_TRC, L_TRC, R_UIM, L_UIM, R_LIM, L_LIM, R_ANK, L_ANK, R_MT5, L_MT5, R_KNE, L_KNE = markers_coordinates_balgrist(points, labels)
    
    l_shank_angle = np.rad2deg(np.arctan2((L_KNE[:,2]-L_ANK[:,2]), (L_KNE[:,1]-L_ANK[:,1]))) #+ 360) % 360 # % (2 * np.pi)
    r_shank_angle = np.rad2deg(np.arctan2((R_KNE[:,2]-R_ANK[:,2]), (R_KNE[:,1]-R_ANK[:,1])))
    #r_shank_angle_v = np.rad2deg(np.arctan2((R_ANK[:,2]-R_KNE[:,2]),(R_ANK[:,1]-R_KNE[:,1])))
    #l_shank_angle_v = np.rad2deg(np.arctan2((L_ANK[:,2]-L_KNE[:,2]),(L_ANK[:,1]-L_KNE[:,1]))) #+ 360) % 360
    l_thigh_angle = np.rad2deg(np.arctan2((L_UIM[:,2]-L_KNE[:,2]), (L_UIM[:,1]-L_KNE[:,1]))) #+ 360) % 360
    r_thigh_angle = np.rad2deg(np.arctan2((R_UIM[:,2]-R_KNE[:,2]), (R_UIM[:,1]-R_KNE[:,1])))
    #l_thigh_angle_v = np.rad2deg(np.arctan2((L_KNE[:,2]-L_TRC[:,2]), (L_KNE[:,1]-L_TRC[:,1])))# + 360) % 360
    #r_thigh_angle_v = np.rad2deg(np.arctan2((R_KNE[:,2]-R_TRC[:,2]), (R_KNE[:,1]-R_TRC[:,1])))# + 360) % 360
    trunk_angle = np.rad2deg(np.arctan2((COM[:,2]-C_TRC[:,2]),(COM[:,1]-C_TRC[:,1]))) #+ 360) % 360
    r_foot_angle = np.rad2deg(np.arctan2((R_ANK[:,2]-R_MT5[:,2]),(R_ANK[:,1]-R_MT5[:,1]))) #+ 360) % 360
    l_foot_angle = np.rad2deg(np.arctan2((L_ANK[:,2]-L_MT5[:,2]),(L_ANK[:,1]-L_MT5[:,1]))) #+ 360) % 360
    r_knee_angle = (r_thigh_angle - r_shank_angle)
    l_knee_angle = (l_thigh_angle - l_shank_angle)
    #b_r_knee_angle = (r_shank_angle + (180-r_thigh_angle))
    #b_l_knee_angle = (l_shank_angle + (180-l_thigh_angle))
    r_hip_angle = (r_thigh_angle - trunk_angle)
    l_hip_angle = (l_thigh_angle - trunk_angle)
    r_ankle_angle = (r_foot_angle - r_shank_angle)-90
    l_ankle_angle = (l_foot_angle - l_shank_angle)-90 
    
    if n == True:
        
        r_knee = standarization(r_knee_angle).reshape(len(r_knee_angle),1)
        l_knee = standarization(l_knee_angle).reshape(len(l_knee_angle),1)
        r_hip = standarization(r_hip_angle).reshape(len(r_hip_angle),1)
        l_hip = standarization(l_hip_angle).reshape(len(l_hip_angle),1)
        r_ankle = standarization(r_ankle_angle).reshape(len(r_ankle_angle),1)
        l_ankle = standarization(l_ankle_angle).reshape(len(l_ankle_angle),1)
        trunk = standarization(trunk_angle).reshape(len(trunk_angle),1)
        r_thigh = standarization(r_thigh_angle).reshape(len(r_thigh_angle),1)
        l_thigh = standarization(l_thigh_angle).reshape(len(l_thigh_angle),1)
        r_shank = standarization(r_shank_angle).reshape(len(r_shank_angle),1)
        l_shank = standarization(l_shank_angle).reshape(len(l_shank_angle),1)
        
        r_knee = normalization(r_knee)
        l_knee = normalization(l_knee)
        r_hip = normalization(r_hip)
        l_hip = normalization(l_hip)
        r_ankle = normalization(r_ankle)
        l_ankle = normalization(l_ankle)
        trunk = normalization(trunk)
        r_thigh = normalization(r_thigh)
        l_thigh = normalization(l_thigh)
        r_shank = normalization(r_shank)
        l_shank = normalization(l_shank)
        
        return r_knee, l_knee, r_hip, l_hip, r_ankle, l_ankle, trunk, r_thigh, l_thigh, r_shank, l_shank
    
    else:
         return r_knee_angle, l_knee_angle, r_hip_angle, l_hip_angle, r_ankle_angle, l_ankle_angle, trunk_angle, r_thigh_angle, l_thigh_angle, r_shank_angle, l_shank_angle
    
