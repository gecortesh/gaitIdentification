#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 10:32:54 2018

@author: gabych

Abnormal gait Deviation
"""
import numpy as np
import bmh
import fuzzy_gait_improved

data_path = '/cluster/home/corteshg/gaitIdentification/joint_arrays/'

r_knee = np.load(data_path+'r_knee.npy')
l_knee = np.load(data_path+'l_knee.npy')
r_hip = np.load(data_path+'r_hip.npy')
l_hip = np.load(data_path+'l_hip.npy') 
r_ankle = np.load(data_path+'r_ankle.npy')
l_ankle = np.load(data_path+'l_ankle.npy')
trunk = np.load(data_path+'trunk.npy')

r_knee_ms = np.load(data_path+'r_knee_ms.npy')
l_knee_ms = np.load(data_path+'l_knee_ms.npy')
r_hip_ms = np.load(data_path+'r_hip_ms.npy')
l_hip_ms = np.load(data_path+'l_hip_ms.npy') 
r_ankle_ms = np.load(data_path+'r_ankle_ms.npy')
l_ankle_ms = np.load(data_path+'l_ankle_ms.npy')
trunk_ms = np.load(data_path+'trunk_ms.npy')

r_knee_md = np.load(data_path+'r_knee_md.npy')
l_knee_md = np.load(data_path+'l_knee_md.npy')
r_hip_md = np.load(data_path+'r_hip_md.npy')
l_hip_md = np.load(data_path+'l_hip_md.npy') 
r_ankle_md = np.load(data_path+'r_ankle_md.npy')
l_ankle_md = np.load(data_path+'l_ankle_md.npy')
trunk_md = np.load(data_path+'trunk_md.npy')

r_knee_sci = np.load(data_path+'r_knee_sci.npy')
l_knee_sci = np.load(data_path+'l_knee_sci.npy')
r_hip_sci = np.load(data_path+'r_hip_sci.npy')
l_hip_sci = np.load(data_path+'l_hip_sci.npy') 
r_ankle_sci = np.load(data_path+'r_ankle_sci.npy')
l_ankle_sci = np.load(data_path+'l_ankle_sci.npy')
trunk_sci = np.load(data_path+'trunk_sci.npy')

r_knee_park = np.load(data_path+'r_knee_park.npy')
l_knee_park = np.load(data_path+'l_knee_park.npy')
r_hip_park = np.load(data_path+'r_hip_park.npy')
l_hip_park = np.load(data_path+'l_hip_park.npy') 
r_ankle_park = np.load(data_path+'r_ankle_park.npy')
l_ankle_park = np.load(data_path+'l_ankle_park.npy')
trunk_park = np.load(data_path+'trunk_park.npy')


av_ctrl_r_knee = np.mean(r_knee,axis=0)
av_ctrl_l_knee = np.mean(l_knee,axis=0)
av_ctrl_r_hip = np.mean(r_hip,axis=0)
av_ctrl_l_hip = np.mean(l_hip,axis=0)
av_ctrl_r_ank = np.mean(r_ankle,axis=0)
av_ctrl_l_ank = np.mean(l_ankle,axis=0)
av_ctrl_trunk = np.mean(trunk,axis=0)

def calculate_distances_dev(reference,joint):
    distances = np.zeros(np.shape(joint))
    for i in range(len(joint)):
        for j in range(len(joint[0])):
            distances[i,j] = bmh.distance_metric(np.reshape(reference[j],1), np.reshape(joint[i][j],1))
    return distances

distances_ms_rk = calculate_distances_dev(av_ctrl_r_knee, r_knee_ms)
distances_md_rk = calculate_distances_dev(av_ctrl_r_knee, r_knee_md)  
distances_sci_rk = calculate_distances_dev(av_ctrl_r_knee, r_knee_sci)
distances_park_rk = calculate_distances_dev(av_ctrl_r_knee, r_knee_park)
distances_ctrl_rk = calculate_distances_dev(av_ctrl_r_knee, r_knee)
        
distances_ms_lk = calculate_distances_dev(av_ctrl_l_knee, l_knee_ms)
distances_md_lk = calculate_distances_dev(av_ctrl_l_knee, l_knee_md)  
distances_sci_lk = calculate_distances_dev(av_ctrl_l_knee, l_knee_sci)
distances_park_lk = calculate_distances_dev(av_ctrl_l_knee, l_knee_park)
distances_ctrl_lk = calculate_distances_dev(av_ctrl_l_knee, l_knee)

distances_ms_rh = calculate_distances_dev(av_ctrl_r_hip, r_hip_ms)
distances_md_rh = calculate_distances_dev(av_ctrl_r_hip, r_hip_md)  
distances_sci_rh = calculate_distances_dev(av_ctrl_r_hip, r_hip_sci)
distances_park_rh = calculate_distances_dev(av_ctrl_r_hip, r_hip_park)
distances_ctrl_rh = calculate_distances_dev(av_ctrl_r_hip, r_hip)

distances_ms_lh = calculate_distances_dev(av_ctrl_l_hip, l_hip_ms)
distances_md_lh = calculate_distances_dev(av_ctrl_l_hip, l_hip_md)  
distances_sci_lh = calculate_distances_dev(av_ctrl_l_hip, l_hip_sci)
distances_park_lh = calculate_distances_dev(av_ctrl_l_hip, l_hip_park)
distances_ctrl_lh = calculate_distances_dev(av_ctrl_l_hip, l_hip)

distances_ms_ra = calculate_distances_dev(av_ctrl_r_ank, r_ankle_ms)
distances_md_ra = calculate_distances_dev(av_ctrl_r_ank, r_ankle_md)  
distances_sci_ra = calculate_distances_dev(av_ctrl_r_ank, r_ankle_sci)
distances_park_ra = calculate_distances_dev(av_ctrl_r_ank, r_ankle_park)
distances_ctrl_ra = calculate_distances_dev(av_ctrl_r_ank, r_ankle)

distances_ms_la = calculate_distances_dev(av_ctrl_l_ank, l_ankle_ms)
distances_md_la = calculate_distances_dev(av_ctrl_l_ank, l_ankle_md)  
distances_sci_la = calculate_distances_dev(av_ctrl_l_ank, l_ankle_sci)
distances_park_la = calculate_distances_dev(av_ctrl_l_ank, l_ankle_park)
distances_ctrl_la = calculate_distances_dev(av_ctrl_l_ank, l_ankle)

distances_ms_tr = calculate_distances_dev(av_ctrl_trunk, trunk_ms)
distances_md_tr = calculate_distances_dev(av_ctrl_trunk, trunk_md)  
distances_sci_tr = calculate_distances_dev(av_ctrl_trunk, trunk_sci)
distances_park_tr = calculate_distances_dev(av_ctrl_trunk, trunk_park)
distances_ctrl_tr = calculate_distances_dev(av_ctrl_trunk, trunk)

    
mean_ctrl_rk_th =  np.mean(distances_ctrl_rk,axis=0)
std_ctrl_rk_th =  np.std(distances_ctrl_rk,axis=0)

mean_ctrl_lk_th =  np.mean(distances_ctrl_lk,axis=0)
std_ctrl_lk_th =  np.std(distances_ctrl_lk,axis=0)

mean_ctrl_rh_th =  np.mean(distances_ctrl_rh,axis=0)
std_ctrl_rh_th =  np.std(distances_ctrl_rh,axis=0)

mean_ctrl_lh_th =  np.mean(distances_ctrl_lh,axis=0)
std_ctrl_lh_th =  np.std(distances_ctrl_lh,axis=0)

mean_ctrl_ra_th =  np.mean(distances_ctrl_ra,axis=0)
std_ctrl_ra_th =  np.std(distances_ctrl_ra,axis=0)

mean_ctrl_la_th =  np.mean(distances_ctrl_la,axis=0)
std_ctrl_la_th =  np.std(distances_ctrl_la,axis=0)

mean_ctrl_tr_th =  np.mean(distances_ctrl_tr,axis=0)
std_ctrl_tr_th =  np.std(distances_ctrl_tr,axis=0)



def calculate_abnormalities(mean, std, distances):         
    abnormal = np.zeros(np.shape(distances))
    for a in range(len(distances)):
        for b in range(len(distances[a])):
            if (distances[a][b] > mean[b]+std[b]) or (distances[a][b] < mean[b]-std[b]):
                abnormal[a][b] = 1
    return abnormal
            
abnormal_md_rk = calculate_abnormalities(mean_ctrl_rk_th, std_ctrl_rk_th, distances_md_rk)
abnormal_ms_rk = calculate_abnormalities(mean_ctrl_rk_th, std_ctrl_rk_th, distances_ms_rk)
abnormal_sci_rk = calculate_abnormalities(mean_ctrl_rk_th, std_ctrl_rk_th, distances_sci_rk)
abnormal_park_rk = calculate_abnormalities(mean_ctrl_rk_th, std_ctrl_rk_th, distances_park_rk)

abnormal_md_lk = calculate_abnormalities(mean_ctrl_lk_th, std_ctrl_lk_th, distances_md_lk)
abnormal_ms_lk = calculate_abnormalities(mean_ctrl_lk_th, std_ctrl_lk_th, distances_ms_lk)
abnormal_sci_lk = calculate_abnormalities(mean_ctrl_lk_th, std_ctrl_lk_th, distances_sci_lk)
abnormal_park_lk = calculate_abnormalities(mean_ctrl_lk_th, std_ctrl_lk_th, distances_park_lk)

abnormal_md_rh = calculate_abnormalities(mean_ctrl_rh_th, std_ctrl_rh_th, distances_md_rh)
abnormal_ms_rh = calculate_abnormalities(mean_ctrl_rh_th, std_ctrl_rh_th, distances_ms_rh)
abnormal_sci_rh = calculate_abnormalities(mean_ctrl_rh_th, std_ctrl_rh_th, distances_sci_rh)
abnormal_park_rh = calculate_abnormalities(mean_ctrl_rh_th, std_ctrl_rh_th, distances_park_rh)

abnormal_md_lh = calculate_abnormalities(mean_ctrl_lh_th, std_ctrl_lh_th, distances_md_lh)
abnormal_ms_lh = calculate_abnormalities(mean_ctrl_lh_th, std_ctrl_lh_th, distances_ms_lh)
abnormal_sci_lh = calculate_abnormalities(mean_ctrl_lh_th, std_ctrl_lh_th, distances_sci_lh)
abnormal_park_lh = calculate_abnormalities(mean_ctrl_lh_th, std_ctrl_lh_th, distances_park_lh)

abnormal_md_ra = calculate_abnormalities(mean_ctrl_ra_th, std_ctrl_ra_th, distances_md_ra)
abnormal_ms_ra = calculate_abnormalities(mean_ctrl_ra_th, std_ctrl_ra_th, distances_ms_ra)
abnormal_sci_ra = calculate_abnormalities(mean_ctrl_ra_th, std_ctrl_ra_th, distances_sci_ra)
abnormal_park_ra = calculate_abnormalities(mean_ctrl_ra_th, std_ctrl_ra_th, distances_park_ra)

abnormal_md_la = calculate_abnormalities(mean_ctrl_la_th, std_ctrl_la_th, distances_md_la)
abnormal_ms_la = calculate_abnormalities(mean_ctrl_la_th, std_ctrl_la_th, distances_ms_la)
abnormal_sci_la = calculate_abnormalities(mean_ctrl_la_th, std_ctrl_la_th, distances_sci_la)
abnormal_park_la = calculate_abnormalities(mean_ctrl_la_th, std_ctrl_la_th, distances_park_la)

abnormal_md_tr = calculate_abnormalities(mean_ctrl_tr_th, std_ctrl_tr_th, distances_md_tr)
abnormal_ms_tr = calculate_abnormalities(mean_ctrl_tr_th, std_ctrl_tr_th, distances_ms_tr)
abnormal_sci_tr = calculate_abnormalities(mean_ctrl_tr_th, std_ctrl_tr_th, distances_sci_tr)
abnormal_park_tr = calculate_abnormalities(mean_ctrl_tr_th, std_ctrl_tr_th, distances_park_tr)

def get_phases(abnormalities, knee, hip):
    if len(abnormalities) > len(knee):
        phases = np.zeros(np.shape(knee))
        max_len = len(knee)
    else:
        phases = np.zeros(np.shape(abnormalities))
        max_len = len(abnormalities)
    for sample in  range(max_len):
        deviations = np.where(abnormalities[sample]!=0)
        for dev in deviations[0]:
            knee_s = knee[sample][dev]
            hip_s = hip[sample][dev]
            thigh_s = dev/len(knee[sample])
            time_s = dev/len(knee[sample])
            phases[sample,dev] = fuzzy_gait_improved.detect_phase(knee_s, hip_s, thigh_s, time_s, visualize=False, mode='swst')
    return np.round(phases)

def percentage(phases_mtx):
    percents = np.zeros(np.shape(phases_mtx))
    for i in range(len(phases_mtx)):
        stance = np.array(np.where(phases_mtx[i]==1))
        if np.max(stance)>154:
            st_perc = (stance/np.max(stance))*100
            percents[i,stance]= st_perc
        else:
            st_perc = (stance/154)*100
            percents[i, stance]= st_perc
        swing = np.array(np.where(phases_mtx[i]==2))
        sw_perc = np.interp(swing,[155,257],[0,100])
        percents[i,swing]= sw_perc      
    return percents

save_path = '/cluster/home/corteshg/gaitIdentification/phases_results_swst/'
phases_rk_md = get_phases(abnormal_md_rk, r_knee_md, r_hip_md)
np.save(save_path+'phases_rk_md', phases_rk_md)
phases_rk_ms = get_phases(abnormal_ms_rk, r_knee_ms, r_hip_ms)
np.save(save_path+'phases_rk_ms', phases_rk_ms)
phases_rk_sci = get_phases(abnormal_sci_rk, r_knee_sci, r_hip_sci)
np.save(save_path+'phases_rk_sci', phases_rk_sci)
phases_rk_park = get_phases(abnormal_park_rk, r_knee_park, r_hip_park)
np.save(save_path+'phases_rk_park', phases_rk_park)

phases_lk_md = get_phases(abnormal_md_lk, l_knee_md, l_hip_md)
np.save(save_path+'phases_lk_md', phases_lk_md)
phases_lk_ms = get_phases(abnormal_ms_lk, l_knee_ms, l_hip_ms)
np.save(save_path+'phases_lk_ms', phases_lk_ms)
phases_lk_sci = get_phases(abnormal_sci_lk, l_knee_sci, l_hip_sci)
np.save(save_path+'phases_lk_sci', phases_lk_sci)
phases_lk_park = get_phases(abnormal_park_lk, l_knee_park, l_hip_park)
np.save(save_path+'phases_lk_park', phases_lk_park)

phases_rh_md = get_phases(abnormal_md_rh, r_knee_md, r_hip_md)
np.save(save_path+'phases_rh_md', phases_rh_md)
phases_rh_ms = get_phases(abnormal_ms_rh, r_knee_ms, r_hip_ms)
np.save(save_path+'phases_rh_ms', phases_rh_ms)
phases_rh_sci = get_phases(abnormal_sci_rh, r_knee_sci, r_hip_sci)
np.save(save_path+'phases_rh_sci', phases_rh_sci)
phases_rh_park = get_phases(abnormal_park_rh, r_knee_park, r_hip_park)
np.save(save_path+'phases_rh_park', phases_rh_park)

phases_lh_md = get_phases(abnormal_md_lh, l_knee_md, l_hip_md)
np.save(save_path+'phases_lh_md', phases_lh_md)
phases_lh_ms =  get_phases(abnormal_ms_lh, l_knee_ms, l_hip_ms)
np.save(save_path+'phases_lh_ms', phases_lh_ms)
phases_lh_sci = get_phases(abnormal_sci_lh, l_knee_sci, l_hip_sci)
np.save(save_path+'phases_lh_sci', phases_lh_sci)
phases_lh_park = get_phases(abnormal_park_lh, l_knee_park, l_hip_park)
np.save(save_path+'phases_lh_park', phases_lh_park)

phases_ra_md = get_phases(abnormal_md_ra, r_knee_md, r_hip_md)
np.save(save_path+'phases_ra_md', phases_ra_md)
phases_ra_ms = get_phases(abnormal_ms_ra, r_knee_ms, r_hip_ms)
np.save(save_path+'phases_ra_ms', phases_ra_ms)
phases_ra_sci = get_phases(abnormal_sci_ra, r_knee_sci, r_hip_sci)
np.save(save_path+'phases_ra_sci', phases_ra_sci)
phases_ra_park = get_phases(abnormal_park_ra, r_knee_park, r_hip_park)
np.save(save_path+'phases_ra_park', phases_ra_park)

phases_la_md = get_phases(abnormal_md_la, r_knee_md, r_hip_md)
np.save(save_path+'phases_la_md', phases_la_md)
phases_la_ms = get_phases(abnormal_ms_la, r_knee_ms, r_hip_ms)
np.save(save_path+'phases_la_ms', phases_la_ms)
phases_la_sci = get_phases(abnormal_sci_la, r_knee_sci, r_hip_sci)
np.save(save_path+'phases_la_sci', phases_la_sci)
phases_la_park = get_phases(abnormal_park_la, r_knee_park, r_hip_park)
np.save(save_path+'phases_la_park', phases_la_park)

phases_tr_md = get_phases(abnormal_md_tr, r_knee_md, r_hip_md)
np.save(save_path+'phases_tr_md', phases_tr_md)
phases_tr_ms = get_phases(abnormal_ms_tr, r_knee_ms, r_hip_ms)
np.save(save_path+'phases_tr_ms', phases_tr_ms)
phases_tr_sci = get_phases(abnormal_sci_tr, r_knee_sci, r_hip_sci)
np.save(save_path+'phases_tr_sci', phases_tr_sci)
phases_tr_park = get_phases(abnormal_park_tr, r_knee_park, r_hip_park)
np.save(save_path+'phases_tr_park', phases_tr_park)

# percentage calculations
pc_rk_md = percentage(phases_rk_md)
np.save(save_path+'pc_rk_md', pc_rk_md)
pc_rk_ms = percentage(phases_rk_ms)
np.save(save_path+'pc_rk_ms', pc_rk_ms)
pc_rk_sci = percentage(phases_rk_sci)
np.save(save_path+'pc_rk_sci', pc_rk_sci)
pc_rk_park = percentage(phases_rk_park)
np.save(save_path+'pc_rk_park', pc_rk_park)

pc_lk_md = percentage(phases_lk_md)
np.save(save_path+'pc_lk_md', pc_lk_md)
pc_lk_ms = percentage(phases_lk_ms)
np.save(save_path+'pc_lk_ms', pc_lk_ms)
pc_lk_sci = percentage(phases_lk_sci)
np.save(save_path+'pc_lk_sci', pc_lk_sci)
pc_lk_park = percentage(phases_lk_park)
np.save(save_path+'pc_lk_park', pc_lk_park)

pc_rh_md = percentage(phases_rh_md)
np.save(save_path+'pc_rh_md', pc_rh_md)
pc_rh_ms = percentage(phases_rh_ms)
np.save(save_path+'pc_rh_ms', pc_rh_ms)
pc_rh_sci = percentage(phases_rh_sci)
np.save(save_path+'pc_rh_sci', pc_rh_sci)
pc_rh_park = percentage(phases_rh_park)
np.save(save_path+'pc_rh_park', pc_rh_park)

pc_lh_md = percentage(phases_lh_md)
np.save(save_path+'pc_lh_md', pc_lh_md)
pc_lh_ms = percentage(phases_lh_ms)
np.save(save_path+'pc_lh_ms', pc_lh_ms)
pc_lh_sci = percentage(phases_lh_sci)
np.save(save_path+'pc_lh_sci', pc_lh_sci)
pc_lh_park = percentage(phases_lh_park)
np.save(save_path+'pc_lh_park', pc_lh_park)

pc_ra_md = percentage(phases_ra_md)
np.save(save_path+'pc_ra_md', pc_ra_md)
pc_ra_ms = percentage(phases_ra_ms)
np.save(save_path+'pc_ra_ms', pc_ra_ms)
pc_ra_sci = percentage(phases_ra_sci)
np.save(save_path+'pc_ra_sci', pc_ra_sci)
pc_ra_park = percentage(phases_ra_park)
np.save(save_path+'pc_ra_park', pc_ra_park)

pc_la_md = percentage(phases_la_md)
np.save(save_path+'pc_la_md', pc_la_md)
pc_la_ms = percentage(phases_la_ms)
np.save(save_path+'pc_la_ms', pc_la_ms)
pc_la_sci = percentage(phases_la_sci)
np.save(save_path+'pc_la_sci', pc_la_sci)
pc_la_park = percentage(phases_la_park)
np.save(save_path+'pc_la_park', pc_la_park)

pc_tr_md = percentage(phases_tr_md)
np.save(save_path+'pc_tr_md', pc_tr_md)
pc_tr_ms = percentage(phases_tr_ms)
np.save(save_path+'pc_tr_ms', pc_tr_ms)
pc_tr_sci = percentage(phases_tr_sci)
np.save(save_path+'pc_tr_sci', pc_tr_sci)
pc_tr_park = percentage(phases_tr_park)
np.save(save_path+'pc_tr_park', pc_tr_park)
  
    
