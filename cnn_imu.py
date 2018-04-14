from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Activation, Dropout, Flatten, Dense
import numpy as np
import dataAndLabels

all_data, labels, data, dataset, labels_dataset = dataAndLabels.dataAndLabels()
train_data = dataset[:,0:39]
#train_data =  np.expand_dims(train_data, axis=2)
train_labels = labels_dataset[:,0:39]
test_data = dataset[:,39:]
test_labels=labels_dataset[:,39:]
epochs = 30
batch_size = 16
window_size = 1500
nb_samples, nb_series = dataset.shape
input_shape = (nb_series,1) #check

# model 2 conv. layers, 1 dense layer
model = Sequential()  # linear stack of layer
model.add(Conv1D(8,2, input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling1D())
print(np.shape(model))

model.add(Conv1D(16,3))
model.add(Activation('relu'))
model.add(MaxPooling1D())

model.add(Flatten())
model.add(Dense(3))  #32?
model.add(Activation('softmax'))


# learning configuration
model.compile(optimizer='rmsprop',  # or maybe ADAM check
			  loss='categorical_crossentropy', # multi-class classification
			  metrics=['accuracy'])

# train
model.fit(train_data, train_labels,
		  epochs=epochs,
		  batch_size=batch_size,
		  validation_data=(test_data,test_labels))
model.save_weights(model_weights_path)






