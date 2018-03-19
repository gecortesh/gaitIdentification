# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 16:55:15 2018

@author: gabych
"""

#import matplotlib.pyplot as plt
#from skimage.util.shape import view_as_windows
import numpy as np
import keras
from keras.models import Model #, Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Input

train_data_ls_window = np.load('train_data_ls_window.npy')
train_data_lt_window = np.load('train_data_lt_window.npy')
train_data_c_window = np.load('train_data_c_window.npy')
test_data_ls_window = np.load('test_data_ls_window.npy')
test_data_lt_window = np.load('test_data_lt_window.npy')
test_data_c_window = np.load('test_data_c_window.npy')
train_labels_encoded = np.load('train_labels_encoded.npy')
test_labels_encoded = np.load('test_labels_encoded.npy')

epochs =10
batch_size = 20
input_shape = (140,1)

#parallel ip for different sections of image
inp1 = Input(shape=train_data_ls_window.shape[1:])
inp2 = Input(shape=train_data_lt_window.shape[1:])
inp3 = Input(shape=train_data_c_window.shape[1:])

# paralle conv and pool layer which process each section of input independently
conv1 = Conv1D(8, 5, activation='relu')(inp1)
conv2 = Conv1D(8, 5, activation='relu')(inp2)
conv3 = Conv1D(8,5, activation='relu')(inp3)

maxp1 = MaxPooling1D(pool_size=2)(conv1)
maxp2 =MaxPooling1D(pool_size=2)(conv2)
maxp3 =MaxPooling1D(pool_size=2)(conv3)

conv4 = Conv1D(4, 5, activation='relu')(maxp1)
conv5 = Conv1D(4, 5, activation='relu')(maxp2)
conv6 = Conv1D(4,5, activation='relu')(maxp3)

maxp4 = MaxPooling1D(pool_size=2)(conv4)
maxp5 =MaxPooling1D(pool_size=2)(conv5)
maxp6 =MaxPooling1D(pool_size=2)(conv6)

# can add multiple parallel conv, pool layes to reduce size

flt1 = Flatten()(maxp4)
flt2 = Flatten()(maxp5)
flt3 = Flatten()(maxp6)

mrg = keras.layers.concatenate([flt1,flt2,flt3])

dense = Dense(723, activation='relu')(mrg)

op = Dense(1, activation='softmax')(dense)

model = Model(input=[inp1, inp2, inp3], output=op)
model.summary()
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
              
history = model.fit([train_data_ls_window,train_data_lt_window,train_data_c_window], train_labels_encoded,  epochs=10, batch_size=28)       



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


#plt.figure(2)
#plt.plot(data_uh)
##plt.plot(R_a_l_s2[0:1500])
##plt.legend([x,y,z],['x','y','z'], loc='upper left')
#plt.title('left shank acc unhealthy')
#plt.show()
#
#plt.figure(3)
#plt.plot(R_a_l_s[0:1500])
##[x2,y2,z2]  = plt.plot(R_a_l_s[0:1500])
##plt.legend([x2,y2,z2],['x','y','z'], loc='upper left')
#plt.title('left shank acc healthy')
#plt.show()