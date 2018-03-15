# -*- coding: utf-8   IMUs = {} -*-
"""
Created on Thu Mar  8 11:08:18 2018

@author: gabych
"""
import helpers
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Activation, Flatten, Dense
import matplotlib.pyplot as plt

model_weights_path = "/home/gabych/Documents/gaitIdentification/weights_model"
window_size = 400
epochs =10
batch_size = 20

input_shape = (window_size,30) 

all_data, labels, data, dataset, labels_dataset = helpers.dataAndLabels()
helpers.plotImuData(dataset,"Left Shank")
l_shank_g, l_shank_a, r_shank_g, r_shank_a, l_thigh_g, l_thing_a, r_thigh_g, r_thing_a, com_g, com_a = helpers.each_sensor_data(dataset)


plt.figure()    
measAx = ['X', 'Y', 'Z']        
plt.subplot(2,3,1)
plt.plot(l_shank_g[:,0], 'r', linewidth=1.5)
plt.title('Angular Vel. along x [deg/s]')
plt.subplot(2,3,2)
plt.plot(l_shank_g[:,1], 'g', linewidth=1.5)
plt.title('Angular Vel. along y [deg/s]')
plt.subplot(2,3,3)
plt.plot(l_shank_g[:,2], 'b', linewidth=1.5)
plt.title('Angular Vel. along z [deg/s]')
plt.xlabel('Time [ms]')
plt.grid()
#plt.title('Acc. along {} [g]'.format(measAx[i]))

plt.suptitle(' IMU - Gyro (top) & Accelerometer (bottom) measurements')  





train_data, train_labels_encoded, test_data, test_labels_encoded = helpers.trainAndTest(window_size)


# model 2 conv. layers, 1 dense layer
model = Sequential()  # linear stack of layer
model.add(Conv1D(64,5, input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling1D())

model.add(Conv1D(20,2))
model.add(Activation('relu'))
model.add(MaxPooling1D())

model.add(Conv1D(16,3))
model.add(Activation('relu'))
model.add(MaxPooling1D())

model.add(Flatten())
model.add(Dense(3))  #32?
model.add(Activation('softmax'))
#model.add(Dropout(0.5))


model.summary()


# learning configuration
model.compile(optimizer='rmsprop',  # or maybe ADAM check
              loss='categorical_crossentropy', # multi-class classification
              metrics=['accuracy'])

# train
history = model.fit(train_data, train_labels_encoded,
          epochs=epochs,
          batch_size=batch_size,
          verbose=1,
          validation_data=(test_data,test_labels_encoded))
          
score = model.evaluate(test_data, test_labels_encoded, verbose=0)        
print('Test loss:', score[0])
print('Test accuracy:', score[1])

## list all data in history
#print(history.history.keys())
## summarize history for accuracy
#plt.plot(history.history['acc'])
#plt.plot(history.history['val_acc'])
#plt.title('model accuracy')
#plt.text(5, 0.5, 'ws=300, ep=30, bs=20, conv64x5, conv20x2, conv16*3, dense')
#plt.ylabel('accuracy')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
#plt.show()
          
a = helpers.get_activations(model, test_data[1:2], print_shape_only=True)  # with just one sample.
helpers.display_activations(a)


#model.save_weights(model_weights_path)


# summarize history for loss
#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
#plt.title('model loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
#plt.show()