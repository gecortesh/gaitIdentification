# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 10:46:36 2018

@author: gabych
"""
import helpers
from skimage.util.shape import view_as_windows
import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import scipy.signal

#dataset = np.load('dataset.npy')
data_uh = helpers.load_data()
#l_shank, r_shank, l_thigh, r_thigh, com = helpers.each_sensor_data(dataset)
#R_g_l_s, R_a_l_s, R_g_r_s, R_a_r_s, R_g_l_t, R_a_l_t, R_g_r_t, R_a_r_t, R_g_c, R_a_c = helpers.factor_extraction(dataset)
#R_a_l_s2, R_a_l_t2, R_a_c2 = helpers.factor_extraction2(data_uh)


left_shank = data_uh[:,1:4]
left_thigh = data_uh[:,4:7]
com = data_uh[:,7:10]

#left_shank, left_thigh, com = helpers.factor_extraction2(data_uh)

#left_shank =  helpers.standarization(left_shank)
#left_thigh =  helpers.standarization(left_thigh)
#com =  helpers.standarization(com)

left_shank =  helpers.normalization(left_shank)
left_thigh =  helpers.normalization(left_thigh)
com =  helpers.normalization(com)

plt.figure()
#plt.subplot(2,1,1)
[x,y,z] = plt.plot(data_uh[:,1:4])
plt.legend([x,y,z],['x','y','z'], loc='upper left')
#plt.subplot(2,1,2)
#[x2,y2,z2] = plt.plot(dataset[:,3:6])
#plt.legend([x2,y2,z2],['x','y','z'], loc='upper left')
plt.show()
#R_a_l_s = helpers.normalization(R_a_l_s)
#R_a_l_t = helpers.normalization(R_a_l_t)
#R_a_c = helpers.normalization(R_a_c)
#R_a_l_s2 = helpers.normalization(R_a_l_s2)
#R_a_l_t2 = helpers.normalization(R_a_l_t2)
#R_a_c2 = helpers.normalization(R_a_c2)

#R_left_shank = np.concatenate((R_a_l_s,R_a_l_s2),axis=0)
#R_left_thigh = np.concatenate((R_a_l_t,R_a_l_t2),axis=0)
#R_com = np.concatenate((R_a_c,R_a_c2),axis=0)

#labels_ls =  data_uh[:,10]
#
##labels_lt = np.ones((len(R_left_thigh),), dtype=int)
##labels_lt[419320:] = labels_lt[419320:]*2
##labels_c = np.ones((len(R_com),), dtype=int)
##labels_c[419320:] = labels_c[419320:]*2
#
#train_test_split = np.random.rand(len(left_shank)) < 0.80
#train_x_ls1 = left_shank[train_test_split]
#train_x_lt1 = left_thigh[train_test_split]
#train_x_c1 = com[train_test_split]
#train_y = labels_ls[train_test_split]
#test_x_ls1 = left_shank[~train_test_split]
#test_x_lt1 = left_thigh[~train_test_split]
#test_x_c1 = com[~train_test_split]
#test_y = labels_ls[~train_test_split]
#
##train_x_ls, train_y_ls, test_x_ls, test_y_ls = helpers.splitData(R_left_shank,labels_ls)
##train_x_lt, train_y_lt, test_x_lt, test_y_lt = helpers.splitData(R_left_thigh,labels_lt)
##train_x_c, train_y_c, test_x_c, test_y_c = helpers.splitData(R_com,labels_c)
#
#window_size= (400,)
#step = 20
#
#train_x_ls = view_as_windows(train_x_ls1[:,0], window_size, step)
#train_x_ls = train_x_ls.reshape(train_x_ls.shape[0],train_x_ls.shape[1],1)
#train_y_ls = view_as_windows(train_x_ls1[:,1], window_size, step)
#train_y_ls = train_y_ls.reshape(train_y_ls.shape[0],train_y_ls.shape[1],1)
#train_z_ls = view_as_windows(train_x_ls1[:,2], window_size, step)
#train_z_ls = train_z_ls.reshape(train_z_ls.shape[0],train_z_ls.shape[1],1)
#
#test_x_ls = view_as_windows(test_x_ls1[:,0], window_size, step)
#test_x_ls = test_x_ls.reshape(test_x_ls.shape[0],test_x_ls.shape[1],1)
#test_y_ls = view_as_windows(test_x_ls1[:,1], window_size, step)
#test_y_ls = test_y_ls.reshape(test_y_ls.shape[0],test_y_ls.shape[1],1)
#test_z_ls = view_as_windows(test_x_ls1[:,2], window_size, step)
#test_z_ls = test_z_ls.reshape(test_z_ls.shape[0],test_z_ls.shape[1],1)
#
#train_x_lt = view_as_windows(train_x_lt1[:,0], window_size, step)
#train_x_lt = train_x_lt.reshape(train_x_lt.shape[0],train_x_lt.shape[1],1)
#train_y_lt = view_as_windows(train_x_lt1[:,1], window_size, step)
#train_y_lt = train_y_lt.reshape(train_y_lt.shape[0],train_y_lt.shape[1],1)
#train_z_lt = view_as_windows(train_x_lt1[:,2], window_size, step)
#train_z_lt = train_z_lt.reshape(train_z_lt.shape[0],train_z_lt.shape[1],1)
#
#test_x_lt = view_as_windows(test_x_lt1[:,0], window_size, step)
#test_x_lt = test_x_lt.reshape(test_x_lt.shape[0],test_x_lt.shape[1],1)
#test_y_lt = view_as_windows(test_x_lt1[:,1], window_size, step)
#test_y_lt = test_y_lt.reshape(test_y_lt.shape[0],test_y_lt.shape[1],1)
#test_z_lt = view_as_windows(test_x_lt1[:,2], window_size, step)
#test_z_lt = test_z_lt.reshape(test_z_lt.shape[0],test_z_lt.shape[1],1)
#
#train_x_c = view_as_windows(train_x_c1[:,0], window_size, step)
#train_x_c = train_x_c.reshape(train_x_c.shape[0],train_x_c.shape[1],1)
#train_y_c = view_as_windows(train_x_c1[:,1], window_size, step)
#train_y_c = train_y_c.reshape(train_y_c.shape[0],train_y_c.shape[1],1)
#train_z_c = view_as_windows(train_x_c1[:,2], window_size, step)
#train_z_c = train_z_c.reshape(train_z_c.shape[0],train_z_c.shape[1],1)
#
#test_x_c = view_as_windows(test_x_c1[:,0], window_size, step)
#test_x_c = test_x_c.reshape(test_x_c.shape[0],test_x_c.shape[1],1)
#test_y_c = view_as_windows(test_x_c1[:,1], window_size, step)
#test_y_c = test_y_c.reshape(test_y_c.shape[0],test_y_c.shape[1],1)
#test_z_c = view_as_windows(test_x_c1[:,2], window_size, step)
#test_z_c = test_z_c.reshape(test_z_c.shape[0],test_z_c.shape[1],1)
#
#
#train_labels = view_as_windows(train_y, window_size, step)
#train_labels= train_labels.astype(int)-1
#u, indices = np.unique(train_labels, return_inverse=True)
#train_labels=u[np.argmax(np.apply_along_axis(np.bincount, 1, indices.reshape(train_labels.shape),None, np.max(indices)+ 1), axis=1)]
#train_labels = train_labels.reshape(train_labels.shape[0],1)
#train_labels_encoded = to_categorical(train_labels)
#
#test_labels = view_as_windows(test_y, window_size, step)
#test_labels= test_labels.astype(int)-1
#ut, indicest = np.unique(test_labels, return_inverse=True)
#test_labels=ut[np.argmax(np.apply_along_axis(np.bincount, 1, indicest.reshape(test_labels.shape),None, np.max(indicest)+ 1), axis=1)]
#test_labels = test_labels.reshape(test_labels.shape[0],1)
#test_labels_encoded = to_categorical(test_labels)
#
##train_data_ls_window = view_as_windows(train_x_ls, window_size, step)
##train_data_ls_window = train_data_ls_window.reshape(train_data_ls_window.shape[0],train_data_ls_window.shape[1],1)
##
##test_data_ls_window = view_as_windows(test_x_ls, window_size, step)
##test_data_ls_window = test_data_ls_window.reshape(test_data_ls_window.shape[0], test_data_ls_window.shape[1],1)
##
##train_data_lt_window = view_as_windows(train_x_lt, window_size, step)
##train_data_lt_window = train_data_lt_window.reshape(train_data_lt_window.shape[0],train_data_lt_window.shape[1],1)
##
##test_data_lt_window = view_as_windows(test_x_lt, window_size, step)
##test_data_lt_window = test_data_lt_window.reshape(test_data_lt_window.shape[0],test_data_lt_window.shape[1],1)
##
##train_data_c_window = view_as_windows(train_x_c, window_size, step)
##train_data_c_window = train_data_c_window.reshape(train_data_c_window.shape[0],train_data_c_window.shape[1],1)
##
##test_data_c_window = view_as_windows(test_x_c, window_size, step)
##test_data_c_window = test_data_c_window.reshape(test_data_c_window.shape[0],test_data_c_window.shape[1],1)
##
##train_labels = view_as_windows(train_y, window_size, step)
##train_labels= train_labels.astype(int)-1
##u, indices = np.unique(train_labels, return_inverse=True)
##train_labels=u[np.argmax(np.apply_along_axis(np.bincount, 1, indices.reshape(train_labels.shape),None, np.max(indices)+ 1), axis=1)]
##train_labels = train_labels.reshape(train_labels.shape[0],1)
##train_labels_encoded = to_categorical(train_labels)
##
##test_labels = view_as_windows(test_y, window_size, step)
##test_labels= test_labels.astype(int)-1
##ut, indicest = np.unique(test_labels, return_inverse=True)
##test_labels=ut[np.argmax(np.apply_along_axis(np.bincount, 1, indicest.reshape(test_labels.shape),None, np.max(indicest)+ 1), axis=1)]
##test_labels = test_labels.reshape(test_labels.shape[0],1)
##test_labels_encoded = to_categorical(test_labels)
#
#np.save('train_x_ls',train_x_ls)
#np.save('train_y_ls',train_y_ls)
#np.save('train_z_ls',train_z_ls)
#np.save('test_x_ls',test_x_ls)
#np.save('test_y_ls',test_y_ls)
#np.save('test_z_ls',test_z_ls)
#
#np.save('train_x_lt',train_x_lt)
#np.save('train_y_lt',train_y_lt)
#np.save('train_z_lt',train_z_lt)
#np.save('test_x_lt',test_x_lt)
#np.save('test_y_lt',test_y_lt)
#np.save('test_z_lt',test_z_lt)
#
#np.save('train_x_c',train_x_c)
#np.save('train_y_c',train_y_c)
#np.save('train_z_c',train_z_c)
#np.save('test_x_c',test_x_c)
#np.save('test_y_c',test_y_c)
#np.save('test_z_c',test_z_c)
#
#np.save('train_labels', train_labels)
#np.save('test_labels', test_labels)
#np.save('test_labels_encoded', test_labels_encoded)
#np.save('train_labels_encoded', train_labels_encoded)