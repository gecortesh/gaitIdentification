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
from scipy.signal import butter, lfilter, sosfilt, sosfiltfilt, filtfilt

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
    all_data_a=[]
    labels_a=[]
    all_data_d=[]
    labels_d=[]
    all_data_l=[]
    labels_l=[]

    for r in range(0,len(files2Read)):
        if "Ascend" in files2Read[r]:
            data['Ascend'][files2Read[r]]={}
            data['Ascend'][files2Read[r]]['dataDict'] =readDataFiles.actualReading(files2Read[r])
            if len(data['Ascend'][files2Read[r]]['dataDict'])>2:
                all_data_a.append(data['Ascend'][files2Read[r]]['dataDict'][files2Read[r]+"_Exp1"])
                labels_a.append(1)
                all_data_a.append(data['Ascend'][files2Read[r]]['dataDict'][files2Read[r]+"_Exp2"])
                labels_a.append(1)
            else:
                all_data_a.append(data['Ascend'][files2Read[r]]['dataDict'][files2Read[r]+"_Exp1"])
                labels_a.append(1)
        elif "Descend" in files2Read[r]:
             data['Descend'][files2Read[r]]={}
             data['Descend'][files2Read[r]]['dataDict']=readDataFiles.actualReading(files2Read[r])
             if len(data['Descend'][files2Read[r]]['dataDict'])>2:
                all_data_d.append(data['Descend'][files2Read[r]]['dataDict'][files2Read[r]+"_Exp1"])
                labels_d.append(2)
                all_data_d.append(data['Descend'][files2Read[r]]['dataDict'][files2Read[r]+"_Exp2"])
                labels_d.append(2)
             else:
                all_data_d.append(data['Descend'][files2Read[r]]['dataDict'][files2Read[r]+"_Exp1"])
                labels_d.append(2)
        elif "Level" in files2Read[r]:
            data['Level'][files2Read[r]]={}
            data['Level'][files2Read[r]]['dataDict']=readDataFiles.actualReading(files2Read[r])
            if len(data['Level'][files2Read[r]]['dataDict'])>2:
                all_data_l.append(data['Level'][files2Read[r]]['dataDict'][files2Read[r]+"_Exp1"])
                labels_l.append(3)
                all_data_l.append(data['Level'][files2Read[r]]['dataDict'][files2Read[r]+"_Exp2"])
                labels_l.append(3)
            else:
                all_data_l.append(data['Level'][files2Read[r]]['dataDict'][files2Read[r]+"_Exp1"])
                labels_l.append(3)

    dataset = all_data_l[0]
    labels_dataset = np.ones((dataset.shape[0], 1))*labels_l[0]
    for i in range(1,len(all_data_l)):
        dataset = np.append(dataset,all_data_l[i],axis=0)
        labels_dataset = np.append(labels_dataset,np.ones((all_data_l[i].shape[0], 1))*labels_l[i],axis=0)

    return all_data_l, labels_l, data, dataset, labels_dataset

def rollingWindow(a, window):
    a = a.T    
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    windowedData = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides).T
    print("Data size: {0}, Window size: {1}".format(a.shape, window))       
    return windowedData
    
def dataSegmentation(data, window_size):
    x_data = data[:(len(data)-(len(data) % window_size))]
    batches = x_data.reshape(-1, window_size, data.shape[1])
    return batches
    
# divide train and test data
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
    
# extracting data per sensor per device (acc, gyro)
def each_sensor_data(all_data):
    l_shank = all_data[:,0:6]
    r_shank = all_data[:,6:12]
    l_thigh = all_data[:,12:18]
    r_thigh = all_data[:,18:24]
    com = all_data[:,24:30]
    return l_shank, r_shank, l_thigh, r_thigh, com
    
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
        shape = np.shape(activation_maps[activation_map])
        if len(shape) == 4:
            activations = np.hstack(np.transpose(activation_maps[activation_map][0], (2, 0, 1)))
        elif len(shape) == 2:
            # try to make it square as much as possible. we can skip some activations.
            activations = activation_maps[activation_map]
        else:
            activations = np.squeeze(activation_maps[activation_map])
            #raise Exception('len(shape) = 3 has not been implemented.')
        plt.title('Activations')
        plt.imshow(activations)
        plt.show()

def plotImuData(data, imu):
    plt.figure()    
    measAx = ['X', 'Y', 'Z', 'X', 'Y', 'Z']        
    for i in range(0, 6):
        plt.subplot(2,3,i+1)
        plt.plot(data[:100,i+3], 'r', linewidth=1.5)
        plt.plot(data[:100,i], 'b', linewidth=1.5)
        plt.xlabel('Time [ms]')
        plt.grid()
        if i > 2:
            plt.title('Acc. along {} [g]'.format(measAx[i]))
        else:
            plt.title('Angular Vel. along {} [deg/s]'.format(measAx[i]))
    plt.suptitle(imu + ' IMU - Gyro (top) & Accelerometer (bottom) measurements') 
    
def factor_extraction(imu_data):
    R_g_l_s = np.sqrt(np.square(imu_data[:,0]) + np.square(imu_data[:,1]) + np.square(imu_data[:,2]))
   # R_g_l_s = R_g_l_s.reshape(len(R_g_l_s),1)
    R_a_l_s = np.sqrt(np.square(imu_data[:,3]) + np.square(imu_data[:,4]) + np.square(imu_data[:,5]))
    #R_a_l_s = R_a_l_s.reshape(len(R_a_l_s),1)
    R_g_r_s = np.sqrt(np.square(imu_data[:,6]) + np.square(imu_data[:,7]) + np.square(imu_data[:,8]))
    #R_g_r_s = R_g_r_s.reshape(len(R_g_r_s),1)
    R_a_r_s = np.sqrt(np.square(imu_data[:,9]) + np.square(imu_data[:,10]) + np.square(imu_data[:,11]))
    #R_a_r_s = R_a_r_s.reshape(len(R_a_r_s),1)
    R_g_l_t = np.sqrt(np.square(imu_data[:,12]) + np.square(imu_data[:,13]) + np.square(imu_data[:,14]))
    #R_g_l_t = R_g_l_t.reshape(len(R_g_l_t),1)
    R_a_l_t = np.sqrt(np.square(imu_data[:,15]) + np.square(imu_data[:,16]) + np.square(imu_data[:,17]))
    #R_a_l_t = R_a_l_t.reshape(len(R_a_l_t),1)
    R_g_r_t = np.sqrt(np.square(imu_data[:,18]) + np.square(imu_data[:,19]) + np.square(imu_data[:,20]))
    #R_g_r_t = R_g_r_t.reshape(len(R_g_r_t),1)
    R_a_r_t = np.sqrt(np.square(imu_data[:,21]) + np.square(imu_data[:,22]) + np.square(imu_data[:,23]))
    #R_a_r_t = R_a_r_t.reshape(len(R_a_r_t),1)
    R_g_c = np.sqrt(np.square(imu_data[:,24]) + np.square(imu_data[:,25]) + np.square(imu_data[:,26]))
    #R_g_c = R_g_c.reshape(len(R_g_c),1)
    R_a_c = np.sqrt(np.square(imu_data[:,27]) + np.square(imu_data[:,28]) + np.square(imu_data[:,29]))
    #R_a_c = R_a_c.reshape(len(R_a_c),1)  
    return R_g_l_s, R_a_l_s, R_g_r_s, R_a_r_s, R_g_l_t, R_a_l_t, R_g_r_t, R_a_r_t, R_g_c, R_a_c
    
def factor_extraction2(imu_data):
    R_a_l_s = np.sqrt(np.sqrt(np.square(imu_data[:,1]) + np.square(imu_data[:,2]) + np.square(imu_data[:,3])))
    #R_a_l_s = R_a_l_s.reshape(len(R_a_l_s),1)    
    R_a_l_t = np.sqrt(np.sqrt(np.square(imu_data[:,4]) + np.square(imu_data[:,5]) + np.square(imu_data[:,6])))
    #R_a_l_t = R_a_l_t.reshape(len(R_a_l_t),1)        
    R_a_c = np.sqrt(np.sqrt(np.square(imu_data[:,7]) + np.square(imu_data[:,8]) + np.square(imu_data[:,9])))
    #R_a_c = R_a_c.reshape(len(R_a_c),1)
    return R_a_l_s, R_a_l_t, R_a_c
    
# to get numerator and denominator of the IIR filter 
def butter_bandpass(lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandpass', output ='ba')
    #sos = butter(order, [low,high], analog='False', btype='band', output='sos')
    return b,a
   
def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    #y = filtfilt(b, a, data, padlen=0)
    #y_out = sosfiltfilt(sos,data)
    return y

def load_data():
    folder = r"/home/gabych/Documents/ETH/gaitIdentification/dataset_fog_release/dataset" 
    files = glob.glob(folder+'/*txt')
    for i in range(0, len(files)):
        data_new = np.loadtxt(files[i], delimiter=" ", unpack=False)
        trigger = np.where(data_new[:,10]==1)
        data_uh = data_new[trigger]       
    return data_uh
    
def normalization(data):
    x = (data - np.mean(data)) / np.std(data)
    return x
    
def slidingWindow(sequence,winSize,step=1):
    """Returns a generator that will iterate through
    the defined chunks of input sequence.  Input sequence
    must be iterable."""
    
    # Verify the inputs
    try: it = iter(sequence)
    except TypeError:
        raise Exception("**ERROR** sequence must be iterable.")
    if not ((type(winSize) == type(0)) and (type(step) == type(0))):
        raise Exception("**ERROR** type(winSize) and type(step) must be int.")
    if step > winSize:
        raise Exception("**ERROR** step must not be larger than winSize.")
    if winSize > len(sequence):
        raise Exception("**ERROR** winSize must not be larger than sequence length.")
    
    # Pre-compute number of chunks to emit
    numOfChunks = ((len(sequence)-winSize)/step)+1
    
    # Do the work
    for i in range(0,numOfChunks*step,step):
        yield sequence[i:i+winSize]
    
def blabla():
    data_uh = helpers.load_data()
    #l_shank, r_shank, l_thigh, r_thigh, com = helpers.each_sensor_data(dataset)
    R_g_l_s, R_a_l_s, R_g_r_s, R_a_r_s, R_g_l_t, R_a_l_t, R_g_r_t, R_a_r_t, R_g_c, R_a_c = helpers.factor_extraction(dataset)
    R_a_l_s2, R_a_l_t2, R_a_c2 = helpers.factor_extraction2(data_uh)
    
    R_a_l_s = helpers.normalization(R_a_l_s)
    R_a_l_t = helpers.normalization(R_a_l_t)
    R_a_c = helpers.normalization(R_a_c)
    R_a_l_s2 = helpers.normalization(R_a_l_s2)
    R_a_l_t2 = helpers.normalization(R_a_l_t2)
    R_a_c2 = helpers.normalization(R_a_c2)
    
    R_left_shank = np.concatenate((R_a_l_s,R_a_l_s2),axis=0)
    R_left_thigh = np.concatenate((R_a_l_t,R_a_l_t2),axis=0)
    R_com = np.concatenate((R_a_c,R_a_c2),axis=0)
    
    labels_ls = np.ones((len(R_left_shank),), dtype=int)
    labels_ls[419320:] = labels_ls[419320:]*2
    labels_lt = np.ones((len(R_left_thigh),), dtype=int)
    labels_lt[419320:] = labels_lt[419320:]*2
    labels_c = np.ones((len(R_com),), dtype=int)
    labels_c[419320:] = labels_c[419320:]*2
    
    train_test_split = np.random.rand(len(R_left_shank)) < 0.70
    train_x_ls = R_left_shank[train_test_split]
    train_x_lt = R_left_thigh[train_test_split]
    train_x_c = R_com[train_test_split]
    train_y = labels_ls[train_test_split]
    test_x_ls = R_left_shank[~train_test_split]
    test_x_lt = R_left_thigh[~train_test_split]
    test_x_c = R_com[~train_test_split]
    test_y = labels_ls[~train_test_split]
    
    #train_x_ls, train_y_ls, test_x_ls, test_y_ls = helpers.splitData(R_left_shank,labels_ls)
    #train_x_lt, train_y_lt, test_x_lt, test_y_lt = helpers.splitData(R_left_thigh,labels_lt)
    #train_x_c, train_y_c, test_x_c, test_y_c = helpers.splitData(R_com,labels_c)
    
    window_size= (140,)
    step = 20
    
    train_data_ls_window = view_as_windows(train_x_ls, window_size, step)
    train_data_ls_window = train_data_ls_window.reshape(train_data_ls_window.shape[0],train_data_ls_window.shape[1],1)
    
    test_data_ls_window = view_as_windows(test_x_ls, window_size, step)
    test_data_ls_window = test_data_ls_window.reshape(test_data_ls_window.shape[0], test_data_ls_window.shape[1],1)
    
    train_data_lt_window = view_as_windows(train_x_lt, window_size, step)
    train_data_lt_window = train_data_lt_window.reshape(train_data_lt_window.shape[0],train_data_lt_window.shape[1],1)
    
    test_data_lt_window = view_as_windows(test_x_lt, window_size, step)
    test_data_lt_window = test_data_lt_window.reshape(test_data_lt_window.shape[0],test_data_lt_window.shape[1],1)
    
    train_data_c_window = view_as_windows(train_x_c, window_size, step)
    train_data_c_window = train_data_c_window.reshape(train_data_c_window.shape[0],train_data_c_window.shape[1],1)
    
    test_data_c_window = view_as_windows(test_x_c, window_size, step)
    test_data_c_window = test_data_c_window.reshape(test_data_c_window.shape[0],test_data_c_window.shape[1],1)
    
    train_labels = view_as_windows(train_y, window_size, step)
    train_labels= train_labels.astype(int)-1
    u, indices = np.unique(train_labels, return_inverse=True)
    train_labels=u[np.argmax(np.apply_along_axis(np.bincount, 1, indices.reshape(train_labels.shape),None, np.max(indices)+ 1), axis=1)]
    train_labels_encoded = to_categorical(train_labels)
    
    test_labels = view_as_windows(test_y, window_size, step)
    test_labels= test_labels.astype(int)-1
    ut, indicest = np.unique(test_labels, return_inverse=True)
    test_labels=ut[np.argmax(np.apply_along_axis(np.bincount, 1, indicest.reshape(test_labels.shape),None, np.max(indicest)+ 1), axis=1)]
    test_labels_encoded = to_categorical(test_labels)