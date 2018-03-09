# -*- coding: utf-8 -*-
"""
Created on Mon May 29 11:57:07 2017

@author: gkogi
"""
from sklearn.svm import LinearSVC, SVC
from sklearn import preprocessing
import matplotlib.pyplot as plt
import imuProcessor as imuProc
import glob as gleb
import scipy.stats
import numpy as np
import pickle
import time
import os

# Super fast windowing
# As in http://www.rigtorp.se/2011/01/01/rolling-statistics-numpy.html
def rollingWindow(a, window):
    
#    featNames = ["IMU{0}_{1}_{2}".format(i,j,k) for i in range(1,6) for k in ["Mean", "stDev", "skewn", "median", "RMS", "kurtosis", "iqr"] for j in ["GyroX", "GyroY", "GyroZ", "AcclX", "AcclY", "AcclZ"]]
    
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    windowedData = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides).T
    print("Data size: {0}, Window size: {1}".format(a.shape, window))
    means = np.mean(np.gradient(windowedData, 7.5, axis=0), axis = 0) # mean
#    means = np.mean(windowedData, axis=0)
    stDev = np.std(windowedData, axis = 0) # standard deviation
    skewn = scipy.stats.skew(windowedData, axis = 0) # Degree of symmetry of the underlying distribution
    mdian = np.median(windowedData, axis = 0) # median of the data
    rms   = np.sqrt(np.mean(np.square(np.gradient(windowedData, 7.5, axis=0)), axis = 0)) # Root-mean-square
#    rms = np.sqrt(np.mean(np.square(windowedData), axis=0))
    kurts = scipy.stats.kurtosis(windowedData, axis = 0) # Degree of "peakness/flatness" of the underlying distribution
    q75, q25 = np.percentile(windowedData, [75 ,25], axis = 0); iqr = q75 - q25 # interquartile range (IQR) is a measure of variability, based on dividing a data set into quartiles. "Midspread / IQR" is the 1st quartile subtracted from the 3rd quartile
    
    # Stacking for a full feature vector
    featVect = np.hstack((means, stDev, skewn, mdian, rms, kurts, iqr))    
#    featVect = np.hstack((means, means, means, means, means, means, means))    
    return featVect#, windowedData


# Only choose required ranges from every experiment #
def splitData(data, t, fName):
    
#    return data
    
    if "Transition" in fName: # Don't want to cut the transition samples #
        return data
    if "Test" in fName: # Don't want to cut the transition samples #
        return data
    elif "Level" in fName:
#        reqRanges = [[20, 205]]
#        reqRanges = [[15, 27], [90, 100], [160, 168], [220, 225]]
        reqRanges = [[15, 48], [90, 120], [160, 190], [230, 240]] # Whole Dataset (w Gleb)
#        reqRanges = [[25, 60], [90, 110], [160, 175], [240, 250]]#, [300, 308]] # level - seconds
    else:
#        reqRanges = [[20, 195]] # Incline - seconds
        reqRanges = [[22, 45], [90, 110], [165, 180]] # Whole Dataset (w Gleb)
#        reqRanges = [[22, 40], [90, 105], [165, 175]] # Full sample
#        reqRanges = [[25, 43], [100, 115], [170, 180]] # Incline - seconds
        
    dataCut = np.zeros((1,30))
    for i in range(0, len(reqRanges)):
        idx = np.where(np.logical_and(t > reqRanges[i][0], t < reqRanges[i][1]))
        dataCut = np.vstack((dataCut, data[idx[0],:]))
        
    dataCut = np.delete(dataCut, (0), axis=0)
    
    return dataCut


# Stupid shit... #
def lenCorr(data, time, fName, exp):
    time = time/1000.0
    
    if "Transition" in fName or "Test" in fName:
        return data, time
   
    elif "CA40B" in fName or "KA40S" in fName or "FN20D" in fName or "ML20B" in fName or "AA40H" in fName or "TM20F" in fName or "LA40B" in fName:
        if "Level" in fName or "2" in exp:
            reqIdx = np.where(time > 10.0)[0]
        elif ("Ascend" in fName and "1" in exp) or ("Descend" in fName and "1" in exp):
            reqIdx = np.where(time > 20.0)[0]
            
    elif "GB20K" in fName or "LS20N" in fName:
        if "Level" in fName or "2" in exp:
            reqIdx = np.where(np.logical_and(time > 10.0, time < time[-1]-35.0))[0]
        elif "Ascend" in fName and "1" in exp or ("Descend" in fName and "1" in exp):
            reqIdx = np.where(np.logical_and(time > 20.0, time < time[-1]-35.0))[0]
            
    elif "ME40G" in fName:
        if "1" in exp:
            reqIdx = np.where(time > 10.0)[0]
        else:
            reqIdx = np.where(time>0)[0]
            
    else:
        reqIdx = np.where(time > 0.0)[0]

    data = data[reqIdx, :]
    time = time[reqIdx] - time[reqIdx[0]]
    
    return data, time

#==============================================================================
# MAIN LOOP - IMU PROCESSING #
#==============================================================================
    
if __name__ == "__main__":
    
    plt.close('all')

    # Specify the directory and the files 
    #folder     = r"C:\Users\Gleb\OneDrive\ETH Zurich\Study Material\Master Thesis\SMS Lab\GRAIL Tests\Balgrist_20170508"
    folder = r"data/Balgrist_20170508"   
#    files2Read = ["CA40BLevel*.csv", "CA40BAscend*.csv", "CA40BDescend*.csv", # Clara Brockmann
#                  "KA40SLevel*.csv", "KA40SAscend*.csv", "KA40SDescend*.csv", # Katja Staehli
#                  "FN20DLevel*.csv", "FN20DAscend*.csv", "FN20DDescend*.csv", # Fabian Dietschi
#                  "ML20BLevel*.csv", "ML20BAscend*.csv", "ML20BDescend*.csv", # Michael Bertschinger
#                  "AA40HLevel*.csv", "AA40HAscend*.csv", "AA40HDescend*.csv", # Agelika Hagen
##                  "TM20FLevel*.csv", "TM20FAscend*.csv", "TM20FDescend*.csv", # Tim Franzmeyer
#                  "LA40BLevel*.csv", "LA40BAscend*.csv", "LA40BDescend*.csv", # Laura Bischoff
##                  
#                  "JE20DLevel*.csv", "JE20DAscend*.csv", "JE20DDescend*.csv", # Jaime Duarte
#                  "KI20SLevel*.csv", "KI20SAscend*.csv", "KI20SDescend*.csv",#, # Kai Schmidt
#                  "MN20GLevel*.csv", "MN20GAscend*.csv", "MN20GDescend*.csv", # Martin Grimmer
#                  "ME40GLevel*.csv", "ME40GAscend*.csv", "ME40GDescend*.csv", # Marie Georgarakis
#                  
#                  "GB20KLevel*.csv", "GB20KAscend*.csv", "GB20KDescend*.csv", # Gleb Koginov
#                  "LS20NLevel*.csv", "LS20NAscend*.csv", "LS20NDescend*.csv"] # Lukas Neuner
                   
#    files2Read = ["JE20DLevel*.csv", "JE20DAscend*.csv", "JE20DDescend*.csv", # Jaime Duarte
#                  "KI20SLevel*.csv", "KI20SAscend*.csv", "KI20SDescend*.csv", # Kai Schmidt
#                  "MN20GLevel*.csv", "MN20GAscend*.csv", "MN20GDescend*.csv", # Martin Grimmer
#                  "ME40GLevel*.csv", "ME40GAscend*.csv", "ME40GDescend*.csv", # Marie Georgarakis
                  
#                  "GB20KLevel*.csv", "GB20KAscend*.csv", "GB20KDescend*.csv", # Gleb Koginov
#                  "LS20NLevel*.csv", "LS20NAscend*.csv", "LS20NDescend*.csv"] # Lukas Neuner
                  
    files2Read   = ["AA40HAscend*.csv"]
#   testFile   = ["AA40HAscend*.csv"]
    
#    for i in range(0, len(testFile)):
#        files2Read.append(testFile[i])
        
    mlData = np.zeros((1,211))
    allDataDict = {}
    for k in range(0, len(files2Read)):
        
        file2Read = files2Read[k]
        
        tempIMUs = {} # Temporary dictionary of all IMU objects (arbitrary IMU sequences)
        IMUs = {} # Dictionary of all IMU objects (final dictionary)

        
        allFls  = gleb.glob(os.path.join(folder, file2Read))
        
        # Create instance for file & inital data processing
        imuDataRecord = imuProc.imuDataFile()
        nFileIdx,nExpIdx = 1,1
        for i in range(0, len(allFls)):
            
            print("\nProcessing file: {0}".format(allFls[i]))
            
            # Instantiate an IMU object for each individual experiment #
            lstObj = imuDataRecord.processFileRecord("IMU{}".format(str(nFileIdx)),allFls[i])
            
            if len(lstObj) > 1 :
                for i in range(0,len(lstObj)):
                    tempIMUs["Exp{0}_IMU{1}".format(str(nExpIdx+i),str(nFileIdx))] = lstObj[i]
            elif len(lstObj) == 1:
                tempIMUs["Exp{0}_IMU{1}".format(str(nExpIdx),str(nFileIdx))] = lstObj[0]

            nFileIdx += 1 
            
            if nFileIdx == 6: # all files processed of one recording
                nFileIdx = 1
                nExpIdx += 1 if nExpIdx == 1 else 2
        
        # Re-arange the IMUs to always have the same order #
        for keys in tempIMUs:
            imuLoc = tempIMUs[keys].imuLoc
            if "Left Ankle" in imuLoc:
                IMUs[keys[0:8]+"1"] = tempIMUs[keys]
            elif "Right Ankle" in imuLoc:
                IMUs[keys[0:8]+"2"] = tempIMUs[keys]
            elif "Left Thigh" in imuLoc:
                IMUs[keys[0:8]+"3"] = tempIMUs[keys]
            elif "Right Thigh" in imuLoc:
                IMUs[keys[0:8]+"4"] = tempIMUs[  keys]
            elif "Centre of Mass" in imuLoc:
                IMUs[keys[0:8]+"5"] = tempIMUs[keys]
                
        ## Clean that memory! ##
        del imuDataRecord, tempIMUs
            
        
        ## Estimte the minimum and maximum ranges for a particular experiment ##
        allKeys = []    
        minTimeExp1 = 0
        maxTimeExp1 = 1e9
        minTimeExp2 = 0
        maxTimeExp2 = 1e9
        
        # Calculate the time-vector for a specific experiment #
        for key in IMUs:
            allKeys.append(key)
            if "Exp2" in key:
                minTimeExp2 = max(minTimeExp2, min(IMUs[key].timeNum))
                maxTimeExp2 = min(maxTimeExp2, max(IMUs[key].timeNum))
            else:   
                minTimeExp1 = max(minTimeExp1, min(IMUs[key].timeNum))
                maxTimeExp1 = min(maxTimeExp1, max(IMUs[key].timeNum))

        
        ## Get interpolated data with 7.5 ms time interval and common x-axis
        exps = ["Exp1", "Exp2"]
        for j in range(0,len(exps)):
            cnt = 0
            for keys in sorted(IMUs.keys()): # Use sorted to avoid getting scared about your gait pattern....
                
                if exps[j] not in keys:
                    continue
                
                # Interpolate all IMUs to a same x-axis #
                if "Exp2" in keys:
                    IMUs[keys].normaliseData(minTimeExp2, maxTimeExp2)
                else:
                    IMUs[keys].normaliseData(minTimeExp1, maxTimeExp1)
                
                # Pre-define an array to stack all interpolated data #
                if cnt==0:
                    timeLength = len(IMUs[keys].timeInterp)
                    allData = np.zeros((timeLength, 30))
                    
                # Store the interpolated IMU data #
                allData[:, cnt*6:(cnt+1)*6] = IMUs[keys].dataInterp
                cnt+=1
            
            # Put the experiment data and time properties into a dictionary #
            if cnt > 0:
                allData, timeAxis = lenCorr(allData, IMUs[exps[j]+"_IMU1"].timeInterp, file2Read, exps[j])
                
                allDataDict[files2Read[k][:-5]+"_"+exps[j]] = allData
                allDataDict[files2Read[k][:-5]+"_"+exps[j]+"_time"] = timeAxis
#                allDataDict[files2Read[k][:-5]+"_"+exps[j]+"_time"] = IMUs[exps[j]+"_IMU1"].timeInterp