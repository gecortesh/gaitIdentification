# -*- coding: utf-8   IMUs = {} -*-
"""
Created on Thu Mar  8 11:08:18 2018

@author: gabych
"""
import helpers
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Activation, Flatten, Dense #Dropout
import matplotlib.pyplot as plt

model_weights_path = "/home/gabych/Documents/gaitIdentification/weights_model"
window_size = 200 #data points, 2seconds (2000ms)
epochs =30
batch_size = 20
input_shape = (window_size,30)

train_data, train_labels_encoded, test_data, test_labels_encoded = helpers.trainAndTest(window_size)


# model 2 conv. layers, 1 dense layer
model = Sequential()  # linear stack of layer
model.add(Conv1D(64,5, input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling1D())

#model.add(Conv1D(8,2, input_shape=input_shape))
#model.add(Activation('relu'))
#model.add(MaxPooling1D())

model.add(Conv1D(16,3))
model.add(Activation('relu'))
model.add(MaxPooling1D())

model.add(Flatten())
model.add(Dense(3))
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

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.text(5, 0.5, 'ws=400, ep=30, bs=20, conv8x2, conv16*3, dense')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#a = helpers.get_activations(model, test_data[1:2], print_shape_only=True)  # with just one sample.
#helpers.display_activations(a)


#model.save_weights(model_weights_path)


#summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()