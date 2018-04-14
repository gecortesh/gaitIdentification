# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 11:25:27 2018

@author: gabych
"""

import numpy as np
import matplotlib.pyplot as plt
from detect_peaks import detect_peaks
import helpers
from scipy.signal import butter, lfilter, sosfilt, sosfiltfilt, filtfilt
import btk
from scipy import signal

dataset = np.load('dataset.npy')
l_shank, r_shank, l_thigh, r_thigh, com = helpers.each_sensor_data(dataset)
R_g_l_s, R_a_l_s, R_g_r_s, R_a_r_s, R_g_l_t, R_a_l_t, R_g_r_t, R_a_r_t, R_g_c, R_a_c = helpers.factor_extraction(dataset)

#detect_peaks(R_g_r_s[0:1200], mph=0.5, mpd=2, threshold=0, valley=False, show=True)
#detect_peaks(R_a_r_s[0:1200], mph=0.5, mpd=2, threshold=0, valley=False, show=True)

# to get numerator and denominator of the IIR filter
def butter_highpass(cutoff, fs, order):
    nyq = 0.5 * fs
    normal_cutoff = highcut / nyq
    b, a = butter(order, normal_cutoff, btype='highpass', output ='ba', analog=False)
    return b,a

def butter_highpass_filter(data, cutoff, fs, order):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def butter_lowpass(cutoff, fs, order):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='lowpass', output ='ba', analog=False)
    return b,a

def butter_lowpass_filter(data, cutoff, fs, order):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y


# Sample rate and desired cutoff frequencies (in Hz).
fs = 133.0
lowcut = 5.0
highcut = 5.0

y0= (R_a_r_s- np.mean(R_a_r_s))/np.std(R_a_r_s)
#B = np.fft.fft(y0)


#plt.plot(B, 'b', label='Raw signal')

# Filter a noisy signal.
#plt.figure(2)
#plt.subplot(2,1,1)
#plt.plot(y0, 'b', label='Raw signal')
#plt.grid(True)
#plt.title('Noisy signal')


#y_abs = np.absolute(y0)
#plt.subplot(2,1,2)
#y = butter_highpass_filter(y0, highcut, fs, order=4)
y2 = butter_lowpass_filter(y0, lowcut, fs, order =4)
normalize = helpers.normalization(y2)
y_abs = np.absolute(normalize)
#normalize2 = helpers.normalization(y)
#y_abs2 = np.absolute(normalize2)
#plt.plot(y_abs, 'r', label='Filtered signal abs')
#plt.plot(y_abs2[:400], 'g', label='Filtered high signal')
#plt.plot(y_abs, 'y', label='Filtered low signal')
#plt.plot(centered, 'b', label='centered signal')
#plt.plot(y0[:400], 'b', label='Raw signal')
#plt.plot(y_abs[:400], 'g', label='Filtered signal (50 Hz)')
#plt.legend(loc='best')
#plt.show()
loc = detect_peaks(normalize[410000:419300], threshold=0.2/np.max(normalize[410000:419300]), mph=0.04, mpd=100, show=False)
loc2 = detect_peaks(y_abs,mpd=50, valley=True,show=False)

gait_cycles=[]
for g in range(0,len(loc2)-2):
    gait_cycles.append(normalize[loc2[g]:loc2[g+2]])

#plt.plot(gait_cycles[5600])

#plt.xlabel('time (seconds)')
#plt.grid(True)
#plt.legend(loc='upper left')

#tree = spatial.KDTree(loc)
#res = tree.query_ball_point([2,0],1)
#plt.figure()
##plt.subplot(2,1,1)
#x = plt.plot(R_a_l_s[:1200], label='left')
#y = plt.plot(R_a_r_s[:1200], label='right')
##[x,y,z] = plt.plot(l_shank[:,:3])
#plt.legend(loc='upper left')
##plt.subplot(2,1,2)
##[x2,y2,z2] = plt.plot(l_shank[:,3:])
##plt.legend([x2,y2,z2],['x','y','z'], loc='upper left')
#plt.show()

# distance between peaks
distance = []
for l in range(0,len(loc)-1):
    distance.append(loc[l+1]-loc[l])

#One cycle is defined as the fragment between two
#peaks and the average cycle length cl computed using all detected peaks
cl=(1.0/(len(loc)-1))*np.sum(distance)
d2 =[]
for c in range(0,len(distance)):
    d2.append(np.square(distance[c]-cl))
# variance of cycle length (var)
var = (1.0/(len(loc)-1))*np.sum(d2)


f, t, Sxx = signal.spectrogram(y_abs[:1000], fs)
plt.pcolormesh(t, f, Sxx)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()