# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 11:03:49 2018

@author: gabych
"""
import readDataFiles
import glob
import numpy as np
from keras.utils import to_categorical
from keras import backend as K
import matplotlib.pyplot as plt

def splitData(dataset,labels_dataset):

    train_test_split = np.random.rand(len(dataset)) < 0.70
    train_x = dataset[train_test_split]
    #train_x =  np.expand_dims(train_x, axis=2)
    train_y = labels_dataset[train_test_split]
    test_x = dataset[~train_test_split]
    #test_x =  np.expand_dims(test_x, axis=2)
    test_y = labels_dataset[~train_test_split]

    return train_x, train_y, test_x, test_y


def dataAndLabels():

    dataPath = glob.glob('/home/gabych/Documents/ETH/gaitIdentification/data/Balgrist_20170508/first/*.csv')
    files2Read = list(set([dataPath[f][75:len(dataPath[f])-6] for f in range(0,len(dataPath))]))
    data={}
    data['Ascend']={}
    data['Descend']={}
    data['Level']={}
    all_data=[]
    labels=[]

    for r in range(0,len(files2Read)):
        if "Ascend" in files2Read[r]:
            data['Ascend'][files2Read[r]]={}
            data['Ascend'][files2Read[r]]['dataDict'] =readDataFiles.actualReading(files2Read[r])
            if len(data['Ascend'][files2Read[r]]['dataDict'])>2:
                all_data.append(data['Ascend'][files2Read[r]]['dataDict'][files2Read[r]+"_Exp1"])
                labels.append(1)
                all_data.append(data['Ascend'][files2Read[r]]['dataDict'][files2Read[r]+"_Exp2"])
                labels.append(1)
            else:
                all_data.append(data['Ascend'][files2Read[r]]['dataDict'][files2Read[r]+"_Exp1"])
                labels.append(1)
        elif "Descend" in files2Read[r]:
             data['Descend'][files2Read[r]]={}
             data['Descend'][files2Read[r]]['dataDict']=readDataFiles.actualReading(files2Read[r])
             if len(data['Descend'][files2Read[r]]['dataDict'])>2:
                all_data.append(data['Descend'][files2Read[r]]['dataDict'][files2Read[r]+"_Exp1"])
                labels.append(2)
                all_data.append(data['Descend'][files2Read[r]]['dataDict'][files2Read[r]+"_Exp2"])
                labels.append(2)
             else:
                all_data.append(data['Descend'][files2Read[r]]['dataDict'][files2Read[r]+"_Exp1"])
                labels.append(2)
        elif "Level" in files2Read[r]:
            data['Level'][files2Read[r]]={}
            data['Level'][files2Read[r]]['dataDict']=readDataFiles.actualReading(files2Read[r])
            if len(data['Level'][files2Read[r]]['dataDict'])>2:
                all_data.append(data['Level'][files2Read[r]]['dataDict'][files2Read[r]+"_Exp1"])
                labels.append(3)
                all_data.append(data['Level'][files2Read[r]]['dataDict'][files2Read[r]+"_Exp2"])
                labels.append(3)
            else:
                all_data.append(data['Level'][files2Read[r]]['dataDict'][files2Read[r]+"_Exp1"])
                labels.append(3)

    dataset = all_data[0]
    labels_dataset = np.ones((dataset.shape[0], 1))*labels[0]
    for i in range(1,len(all_data)):
        dataset = np.append(dataset,all_data[i],axis=0)
        labels_dataset = np.append(labels_dataset,np.ones((all_data[i].shape[0], 1))*labels[i],axis=0)

    return all_data, labels, data, dataset, labels_dataset

def rollingWindow(a, window):
    a.T    
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    windowedData = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides).T
    print("Data size: {0}, Window size: {1}".format(a.shape, window))    
    return windowedData
    
def dataSegmentation(data, window_size):
    x_data = data[:(len(data)-(len(data) % window_size))]
    batches = x_data.reshape(-1, window_size, data.shape[1])
    return batches
    
def trainAndTest(window_size):
    all_data, labels, data, dataset, labels_dataset = dataAndLabels()
    train_x, train_y, test_x, test_y = splitData(dataset,labels_dataset)
    train_data = dataSegmentation(train_x, window_size)
    train_labels= dataSegmentation(train_y, window_size)
    train_labels= train_labels.astype(int)-1
    u, indices = np.unique(train_labels, return_inverse=True)
    train_labels=u[np.argmax(np.apply_along_axis(np.bincount, 1, indices.reshape(train_labels.shape),None, np.max(indices)+ 1), axis=1)]
    train_labels_encoded = to_categorical(train_labels)
    test_data = dataSegmentation(test_x, window_size)
    test_labels= dataSegmentation(test_y, window_size)
    test_labels= test_labels.astype(int)-1
    u2, indices2 = np.unique(test_labels, return_inverse=True)
    test_labels=u2[np.argmax(np.apply_along_axis(np.bincount, 1, indices2.reshape(test_labels.shape),None, np.max(indices2)+ 1), axis=1)]
    test_labels_encoded = to_categorical(test_labels)
    return train_data, train_labels_encoded, test_data, test_labels_encoded
    
# to visualize the activation function of a layer as in https://github.com/philipperemy/keras-visualize-activations
def get_activations(model, model_inputs, print_shape_only=False, layer_name=None):
    print('----- activations -----')
    activations = []
    inp = model.input

    model_multi_inputs_cond = True
    if not isinstance(inp, list):
        # only one input! let's wrap it in a list.
        inp = [inp]
        model_multi_inputs_cond = False

    outputs = [layer.output for layer in model.layers if
               layer.name == layer_name or layer_name is None]  # all layer outputs

    funcs = [K.function(inp + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions

    if model_multi_inputs_cond:
        list_inputs = []
        list_inputs.extend(model_inputs)
        list_inputs.append(0.)
    else:
        list_inputs = [model_inputs, 0.]

    # Learning phase. 0 = Test mode (no dropout or batch normalization)
    # layer_outputs = [func([model_inputs, 0.])[0] for func in funcs]
    layer_outputs = [func(list_inputs)[0] for func in funcs]
    for layer_activations in layer_outputs:
        activations.append(layer_activations)
        if print_shape_only:
            print(layer_activations.shape)
        else:
            print(layer_activations)
    return activations


def display_activations(activation_maps):
    """
    (1, 26, 26, 32)
    (1, 24, 24, 64)
    (1, 12, 12, 64)
    (1, 12, 12, 64)
    (1, 9216)
    (1, 128)
    (1, 128)
    (1, 10)
    """
    batch_size = activation_maps[0].shape[0]
    assert batch_size == 1, 'One image at a time to visualize.'
    for activation_map in range(0,len(activation_maps)):
        print('Displaying activation map {}'.format(activation_map))
        shape = activation_map.shape
        if len(shape) == 4:
            activations = np.hstack(np.transpose(activation_maps[activation_map][0], (2, 0, 1)))
        elif len(shape) == 2:
            # try to make it square as much as possible. we can skip some activations.
            activations = activation_map[activation_map][0]
            num_activations = len(activations)
            if num_activations > 1024:  # too hard to display it on the screen.
                square_param = int(np.floor(np.sqrt(num_activations)))
                activations = activations[0: square_param * square_param]
                activations = np.reshape(activations, (square_param, square_param))
            else:
                activations = np.expand_dims(activations, axis=0)
        else:
            activations = np.squeeze(activation_maps[activation_map])
            #raise Exception('len(shape) = 3 has not been implemented.')
        plt.imshow(activations)
        plt.show()





