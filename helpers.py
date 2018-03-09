# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 11:03:49 2018

@author: gabych
"""
import readDataFiles
import glob
import numpy as np

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

    dataPath = glob.glob('/home/gabych/Documents/ETH/intention_prediction/data/Balgrist_20170508/first/*.csv')
    files2Read = list(set([dataPath[f][77:len(dataPath[f])-6] for f in range(0,len(dataPath))]))
    data={}
    data['Ascend']={}
    data['Descend']={}
    data['Level']={}
    all_data=[]
    labels=[]

    for r in range(0,len(files2Read)):
        if "Ascend" in files2Read[r]:
            data['Ascend'][files2Read[r]]={}
            data['Ascend'][files2Read[r]]['dataDict']=readDataFiles.actualReading(files2Read[r])
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
    batches = x_data.reshape(-1, window_size, data[1].shape)
    return batches
    