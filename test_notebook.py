# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 11:08:18 2018

@author: gabych
"""
import helpers
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import Conv1D, MaxPooling1D, Activation, Dropout, Flatten, Dense
import numpy as np

model_weights_path = "/home/gabych/Documents/gaitIdentification"
all_data, labels, data, dataset, labels_dataset = helpers.dataAndLabels()
train_x, train_y, test_x, test_y = helpers.splitData(dataset,labels_dataset)

epochs = 30
batch_size = 16
window_size = 1500
nb_samples, nb_series = dataset.shape
input_shape = (1500,30) #check

train_data = helpers.dataSegmentation(train_x, window_size)
train_labels= helpers.dataSegmentation(train_y, window_size)
train_labels= train_labels.astype(int)-1
u, indices = np.unique(train_labels, return_inverse=True)
train_labels=u[np.argmax(np.apply_along_axis(np.bincount, 1, indices.reshape(train_labels.shape),None, np.max(indices)+ 1), axis=1)]
train_labels_encoded = to_categorical(train_labels)
test_data = helpers.dataSegmentation(test_x, window_size)
test_labels= helpers.dataSegmentation(test_y, window_size)
test_labels= test_labels.astype(int)-1
u2, indices2 = np.unique(test_labels, return_inverse=True)
test_labels=u2[np.argmax(np.apply_along_axis(np.bincount, 1, indices2.reshape(test_labels.shape),None, np.max(indices2)+ 1), axis=1)]
test_labels_encoded = to_categorical(test_labels)

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
model.fit(train_data, train_labels_encoded,
          epochs=epochs,
          batch_size=batch_size,
          validation_data=(test_data,test_labels_encoded))
model.save_weights(model_weights_path)