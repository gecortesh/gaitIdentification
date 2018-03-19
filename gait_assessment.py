# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 16:55:15 2018

@author: gabych
"""

import helpers
import matplotlib.pyplot as plt
#from scipy.signal import freqz
#import numpy as np

#all_data_l, labels_l, data, dataset, labels_dataset = helpers.dataAndLabels()
#l_shank, r_shank, l_thigh, r_thigh, com = helpers.each_sensor_data(dataset)
#R_g_l_s, R_a_l_s = helpers.factor_extraction(l_shank)
#R_g_r_s, R_a_r_s = helpers.factor_extraction(r_shank)
#R_g_l_t, R_a_l_t = helpers.factor_extraction(l_thigh)
#R_g_r_t, R_a_r_t = helpers.factor_extraction(r_thigh)
#R_g_c, R_a_c = helpers.factor_extraction(com)

#plt.figure()       
#plt.subplot(2,1,1)
#plt.plot(R_g_l_s[0:10000,:], 'r', linewidth=1.5)
##plt.plot(R_g_l_t, 'b', linewidth=1.5)
#plt.plot(R_g_r_s[0:10000,:], 'y', linewidth=1.5)
##plt.plot(R_g_r_t, 'm', linewidth=1.5)
##plt.plot(R_g_c, 'g', linewidth=1.5)
#plt.xlabel('Time [ms]')
#plt.grid()
##plt.title('Acc. along {} [g]'.format(measAx[i]))
#plt.title('Angular Vel. left leg and com [deg/s]')
#plt.subplot(2,1,2)
#plt.plot(R_a_l_s[0:10000,:], 'r', linewidth=1.5)
##plt.plot(R_a_l_t, 'b', linewidth=1.5)
#plt.plot(R_a_r_s[0:10000,:], 'y', linewidth=1.5)
##plt.plot(R_a_r_t, 'm', linewidth=1.5)
##plt.plot(R_a_c, 'g', linewidth=1.5)
#plt.xlabel('Time [ms]')
#plt.grid()
#plt.title('Acc. right leg and com [g]')
#

#Am = R_a_l_s[0:10000,:]*9.81

#    
#    
# Sample rate and desired cutoff frequencies (in Hz).
#fs = 133.0
#lowcut = 0.05
#highcut = 20
##   
## Plot the frequency response for a few different orders.
##plt.figure(1)
##plt.clf()
##for order in [2, 10]:
##    b, a = helpers.butter_bandpass(lowcut, highcut, fs, order=order)
##    w, h = freqz(b, a, worN=2000)
##    plt.plot((fs * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)
##  
##plt.plot([0, 0.5 * fs], [np.sqrt(0.5), np.sqrt(0.5)],'--', label='sqrt(0.5)')
##plt.xlabel('Frequency (Hz)')
##plt.ylabel('Gain')
##plt.grid(True)
##plt.legend(loc='best')
##   
## Filter a noisy signal.
#plt.figure(2)
##plt.subplot(2,1,1)
#plt.plot(Am, 'b', label='Raw signal')
##plt.grid(True)
##plt.title('Noisy signal')
#
##plt.subplot(2,1,2)
#y = helpers.butter_bandpass_filter(Am, lowcut, highcut, fs, order=2)
##y2 = butter_bandpass_filter(R_a_l_s[0:10000,:], lowcut, highcut, fs, order=10)
#plt.plot(y, 'r', label='Filtered signal')
##plt.plot(y2, 'g', label='Filtered signal (50 Hz)')
#plt.xlabel('time (seconds)')
#plt.grid(True)
#plt.legend(loc='upper left')


plt.figure(2)
[x,y,z]  = plt.plot(r_shank[0:00,3:6])
plt.legend([x,y,z],['x','y','z'], loc='upper left')
plt.title('right shank acc')
plt.show()

plt.figure(3)
[x2,y2,z2]  = plt.plot(r_shank[450:650,3:6])
plt.legend([x2,y2,z2],['x','y','z'], loc='upper left')
plt.title('left shank acc')
plt.show()