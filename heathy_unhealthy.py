# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 16:55:15 2018

@author: gabych
"""
#import helpers
import matplotlib.pyplot as plt
#from skimage.util.shape import view_as_windows
import numpy as np
import keras
from keras.models import Model #, Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Input, Dropout

train_x_ls = np.load('train_x_ls.npy')
train_y_ls = np.load('train_y_ls.npy')
train_z_ls = np.load('train_z_ls.npy')
test_x_ls = np.load('test_x_ls.npy')
test_y_ls = np.load('test_y_ls.npy')
test_z_ls = np.load('test_z_ls.npy')

train_x_lt = np.load('train_x_lt.npy')
train_y_lt = np.load('train_y_lt.npy')
train_z_lt = np.load('train_z_lt.npy')
test_x_lt = np.load('test_x_lt.npy')
test_y_lt = np.load('test_y_lt.npy')
test_z_lt = np.load('test_z_lt.npy')

train_x_c = np.load('train_x_c.npy')
train_y_c = np.load('train_y_c.npy')
train_z_c = np.load('train_z_c.npy')
test_x_c = np.load('test_x_c.npy')
test_y_c = np.load('test_y_c.npy')
test_z_c = np.load('test_z_c.npy')

train_labels = np.load('train_labels.npy')
test_labels = np.load('test_labels.npy')

epochs =10
batch_size = 28
input_shape = (256,1)

#parallel ip for different sections of image
inp1 = Input(shape=train_x_ls.shape[1:])
inp2 = Input(shape=train_y_ls.shape[1:])
inp3 = Input(shape=train_z_ls.shape[1:])

inp4 = Input(shape=train_x_lt.shape[1:])
inp5 = Input(shape=train_y_lt.shape[1:])
inp6 = Input(shape=train_z_lt.shape[1:])

inp7 = Input(shape=train_x_c.shape[1:])
inp8 = Input(shape=train_y_c.shape[1:])
inp9 = Input(shape=train_z_c.shape[1:])

# paralle conv and pool layer which process each section of input independently
conv1 = Conv1D(8, 5, activation='relu')(inp1)
conv2 = Conv1D(8, 5, activation='relu')(inp2)
conv3 = Conv1D(8,5, activation='relu')(inp3)

conv4 = Conv1D(8, 5, activation='relu')(inp4)
conv5 = Conv1D(8, 5, activation='relu')(inp5)
conv6 = Conv1D(8,5, activation='relu')(inp6)

conv7 = Conv1D(8, 5, activation='relu')(inp7)
conv8 = Conv1D(8, 5, activation='relu')(inp8)
conv9 = Conv1D(8,5, activation='relu')(inp9)

maxp1 = MaxPooling1D(pool_size=2)(conv1)
maxp2 =MaxPooling1D(pool_size=2)(conv2)
maxp3 =MaxPooling1D(pool_size=2)(conv3)

maxp4 = MaxPooling1D(pool_size=2)(conv4)
maxp5 =MaxPooling1D(pool_size=2)(conv5)
maxp6 =MaxPooling1D(pool_size=2)(conv6)

maxp7 = MaxPooling1D(pool_size=2)(conv7)
maxp8 =MaxPooling1D(pool_size=2)(conv8)
maxp9 =MaxPooling1D(pool_size=2)(conv9)

conv10 = Conv1D(4, 5, activation='relu')(maxp1)
conv11 = Conv1D(4, 5, activation='relu')(maxp2)
conv12 = Conv1D(4,5, activation='relu')(maxp3)

conv13 = Conv1D(4, 5, activation='relu')(maxp4)
conv14 = Conv1D(4, 5, activation='relu')(maxp5)
conv15 = Conv1D(4,5, activation='relu')(maxp6)

conv16 = Conv1D(4, 5, activation='relu')(maxp7)
conv17 = Conv1D(4, 5, activation='relu')(maxp8)
conv18 = Conv1D(4,5, activation='relu')(maxp9)

maxp10 = MaxPooling1D(pool_size=2)(conv10)
maxp11 =MaxPooling1D(pool_size=2)(conv11)
maxp12 =MaxPooling1D(pool_size=2)(conv12)

maxp13 = MaxPooling1D(pool_size=2)(conv13)
maxp14 =MaxPooling1D(pool_size=2)(conv14)
maxp15 =MaxPooling1D(pool_size=2)(conv15)

maxp16 = MaxPooling1D(pool_size=2)(conv16)
maxp17 =MaxPooling1D(pool_size=2)(conv17)
maxp18 =MaxPooling1D(pool_size=2)(conv18)

# can add multiple parallel conv, pool layes to reduce size
flt1 = Flatten()(maxp10)
flt2 = Flatten()(maxp11)
flt3 = Flatten()(maxp12)

flt4 = Flatten()(maxp13)
flt5 = Flatten()(maxp14)
flt6 = Flatten()(maxp15)

flt7 = Flatten()(maxp16)
flt8 = Flatten()(maxp17)
flt9 = Flatten()(maxp18)

mrg = keras.layers.concatenate([flt1,flt2,flt3,flt4,flt5,flt6,flt7,flt8,flt9])

dense = Dense(723, activation='relu')(mrg)

op = Dense(1, activation='softmax')(dense)

model = Model(input=[inp1, inp2, inp3, inp4, inp5, inp6, inp7, inp8, inp9], output=op)
model.summary()
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
              
history = model.fit([train_x_ls,train_y_ls,train_z_ls, train_x_lt,train_y_lt,train_z_lt, train_x_c,train_y_c,train_z_c], train_labels,  epochs=epochs, batch_size=batch_size, verbose=1,
          validation_data=([test_x_ls,test_y_ls,test_z_ls, test_x_lt,test_y_lt,test_z_lt, test_x_c,test_y_c,test_z_c],test_labels))       
score = model.evaluate([test_x_ls,test_y_ls,test_z_ls, test_x_lt,test_y_lt,test_z_lt, test_x_c,test_y_c,test_z_c], test_labels, verbose=0)        
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.text(5, 0.5, 'ws=300, ep=30, bs=20, conv64x5, conv20x2, conv16*3, dense')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


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