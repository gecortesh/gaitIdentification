# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 10:17:52 2018

@author: gabych
"""

import numpy as np
from keras.models import Sequential
from keras.layers import Conv1D, AveragePooling1D, Activation, Flatten, Dense
import matplotlib.pyplot as plt
import glob

data_path = "/home/gabych/Documents/ETH/gaitIdentification/gaitpdb/*.txt"
files = glob.glob(data_path)

data_set =[]
for name in files:
    data_set.append(np.loadtxt(name))


#input_shape = (1,1)
#epochs =10
#
## model 2 conv. layers, 1 dense layer
#model = Sequential()  # linear stack of layer
#model.add(Conv1D(6,7, input_shape=input_shape))
#model.add(Activation('relu'))
#model.add(AveragePooling1D())
#
#model.add(Conv1D(17,7))
#model.add(Activation('relu'))
#model.add(AveragePooling1D())
#
#model.add(Flatten())
#model.add(Dense(2))
#model.add(Activation('softmax'))
#
#
#model.summary()
#
#
## learning configuration
#model.compile(optimizer='rmsprop',  # or maybe ADAM check
#              loss='categorical_crossentropy', # multi-class classification
#              metrics=['accuracy'])
#
## train
#history = model.fit(train_data, train_labels_encoded,
#          epochs=epochs,
#          batch_size=batch_size,
#          verbose=1,
#          validation_data=(test_data,test_labels_encoded))
#
#score = model.evaluate(test_data, test_labels_encoded, verbose=0)
#
#print('Test loss:', score[0])
#print('Test accuracy:', score[1])
#
## list all data in history
#print(history.history.keys())
## summarize history for accuracy
#plt.plot(history.history['acc'])
#plt.plot(history.history['val_acc'])
#plt.title('model accuracy')
#plt.text(5, 0.5, 'ws=400, ep=30, bs=20, conv8x2, conv16*3, dense')
#plt.ylabel('accuracy')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
#plt.show()

