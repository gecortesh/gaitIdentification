# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 11:08:18 2018

@author: gabych
"""
import stuff
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Activation, Dropout, Flatten, Dense
import numpy as np

all_data, labels, data, dataset, labels_dataset = utilities.dataAndLabels()
train_x, train_y, test_x, test_y = utilities.splitData(dataset,labels_dataset)

epochs = 30
batch_size = 16
window_size = 1500
nb_samples, nb_series = dataset.shape
input_shape = (nb_series,1) #check

train_data = utilities.dataSegmentation(train_x, window_size)
train_labels= utilities.dataSegmentation(train_y, window_size)
test_data = utilities.dataSegmentation(test_x, window_size)
test_labels= utilities.dataSegmentation(test_y, window_size)

# model 2 conv. layers, 1 dense layer
model = Sequential()  # linear stack of layer
model.add(Conv1D(8,2, input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling1D())


model.add(Conv1D(16,3))
model.add(Activation('relu'))
model.add(MaxPooling1D())

model.add(Flatten())
model.add(Dense(3))  #32?
model.add(Activation('softmax'))
model.summary()

# learning configuration
model.compile(optimizer='rmsprop',  # or maybe ADAM check
              loss='categorical_crossentropy', # multi-class classification
              metrics=['accuracy'])

# train
model.fit(train_x, train_y,
          epochs=epochs,
          batch_size=batch_size,
          validation_data=(test_x,test_y))
model.save_weights(model_weights_path)