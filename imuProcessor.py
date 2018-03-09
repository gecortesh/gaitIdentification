import csv
import collections
import numpy as np
import math as m
from os import path
import glob as gleb
from datetime import datetime, timedelta
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

#TODO: think about moving the time processing to this class as well
class imuDataFile():
    def __init__(self):
        self.time = []
        
    def processFileRecord(self, sName, sFileName):
        
        with open(sFileName,'r') as test_dataset:
            
            ## Read in the data and remove Trigger-OFF points
            datatest_iter = csv.reader(test_dataset, delimiter = ',', quotechar = '"')
            datatest      = np.asarray([data for data in datatest_iter])
            
            trigVect   = np.asarray(datatest[:,2], dtype = np.dtype(int)) # Get trigger vector
            trigOnIdx  = np.where(trigVect == 1) # Find Trigger-ON data samples
            trig2Stage = np.where(np.diff(trigOnIdx) > 1) # Find if two or one experiment setup
            
            lstImuData = []            
            if not trig2Stage[1]: ## If only one experiment (e.g level walking)
                rawData = datatest[trigOnIdx]
                lstImuData.append(imuData(sName, rawData)) 
            
            elif trig2Stage[1] and len(trig2Stage) == 2: ## If two experiments (e.g Incline/decline walking)
                rawData = datatest[trigOnIdx[0][0:trig2Stage[1][0]],:]
                lstImuData.append(imuData(sName, rawData))
                
                rawData = datatest[trigOnIdx[0][trig2Stage[1][0]+1:],:]
                lstImuData.append(imuData(sName, rawData)) 
                
            else: ## Shouldn't happen, otherwise consider on an individual basis
                raise ValueError("ERROR, TRIGGER MATRIX NOT CONTINUOUS")
                
            for obj in lstImuData:
                obj.processTime()
                
            return lstImuData        
#        trigger = dataset[:,self._TRIGGER_COL].astype(bool) 
#        self.time = [datetime.strptime(ts,'%H:%M:%S.%f') for ts in dataset[trigger == True,self._TIME_COL]] #TODO: convert this into a relative time frame
#        self.allData = dataset[trigger == True, 3:dataset.shape[1]].astype(float)
 

class imuData():
    def __init__(self, imuName, rawData):
        
        ## General information
        self.imuInternalName = imuName 
        self.imuLoc = ''
        self.imuMac = ''
        
        ## Time-related vars - all absolute and relative time-stamps are in [ms] ##
        self.timeNum   = [] # Absolute time-stamps 
        self.timeStamp = [] # Relative time-stamps

        ## Measurement data-related vars ##
        self.allData = rawData
        
        ## Interpolated data variables ##
        self.dataInterp = []
        self.timeInterp = []
                
        self.imuLocation()
    
    def debug(self):
        print("\n---- Debug info ----")
        print("name: {0}, location: {1}, mac: {2}".format(self.imuInternalName,self.imuLoc,self.imuMac))
 
    def imuLocation(self):
        self.imuMac = self.allData[0,1]
        
        imuMacSplit = self.allData[0,1].split(":")
        imuID = imuMacSplit[4]    
        if imuID == '16':
            self.imuLoc = 'Centre of Mass'
        elif imuID == '41':
            self.imuLoc = 'Right Thigh'
        elif imuID == '6e':
            self.imuLoc = 'Left Thigh'
        elif imuID == 'd0':
            self.imuLoc = 'Left Ankle'
        elif imuID == '5b':
            self.imuLoc = 'Right Ankle'
        else:
            print("NO IMU FOUND:")
            print(self.imuMac)
               
    ## Format the raw time-data for further processing ##
    def processTime(self):
        
        time_string = self.allData[:,0]
        
        ## Need to check if hour-change happened during data recording (%H data not recorded) ##
        if time_string[0][2:4] > time_string[-1][2:4]:  # check for hour switch
            timeConverted = [datetime.strptime(ts[2:-1],'%M:%S.%f') if int(ts[2:4])>int(time_string[0][3:4]) else datetime.strptime(ts[2:-1],'%M:%S.%f') +  timedelta(hours=1) for ts in time_string]
        else:
            timeConverted = [datetime.strptime(ts[2:-1], '%M:%S.%f') for ts in time_string]
            
        self.timeStamp = np.empty(self.allData.shape[0])
        self.timeNum   = np.empty(self.allData.shape[0])
        
        for i in range(0,len(timeConverted)):
            timeDiff = timeConverted[i]-timeConverted[0]
            self.timeStamp[i] = (timeDiff.seconds//3600)*3.6e6 + (timeDiff.seconds//60)*60000  + (timeDiff.seconds%60)*1000   + timeDiff.microseconds/1000 
            self.timeNum[i]   = timeConverted[i].hour*3.6e6    + timeConverted[i].minute*60000 + timeConverted[i].second*1000 + timeConverted[i].microsecond/1000  # us -> ms
            
        ## Check for duplicate timestamps and delete these data instances ##
#        checkDuplicates = [item for item, count in collections.Counter(IMUs[self.imu].timeStamp).items() if count > 1]
#        print("Checking for duplicates...")
#        print("allData shape (before): {}".format(self.allData.shape[0]))
#        print("Timestamp shape (before): {}".format(self.timeStamp.shape[0]))

#        if checkDuplicates:
#            print("Duplicates found!")
#            for i in range(len(checkDuplicates)-1,-1,-1):
#                print("removing {}".format(checkDuplicates[i]))
#                duplicateIdx   = np.where(IMUs[self.imu].timeStamp==checkDuplicates[i])[0][1:]
#                self.timeStamp = np.delete(self.timeStamp, duplicateIdx)
#                self.allData   = np.delete(self.allData, duplicateIdx, axis=0)
#                self.timeNum   = np.delete(self.timeNum, duplicateIdx)
#        else:
#            print("No duplicates found!")
#        print("Validation: {}".format([item for item, count in collections.Counter(IMUs[self.imu].timeStamp).items() if count > 1]))

        ## Data plotting ##
    def plotImuData(self, plottype):

        plt.figure()    
        if plottype == 1:
            measAx = ['X', 'Y', 'Z', 'X', 'Y', 'Z']        

            for i in range(0, 6):
                plt.subplot(2,3,i+1)
                plt.plot(self.timeStamp, self.allData[:,i+3], 'r', linewidth=1.5)
                plt.plot(self.timeInterp, self.dataInterp[:,i], 'b', linewidth=1.5)
                plt.xlabel('Time [ms]')
                plt.grid()
                if i > 2:
                    plt.title('Acc. along {} [g]'.format(measAx[i]))
                else:
                    plt.title('Angular Vel. along {} [deg/s]'.format(measAx[i]))
            plt.suptitle(self.imuLoc + ' IMU - Gyro (top) & Accelerometer (bottom) measurements')  
        
        elif plottype == 2:
            plt.plot(self.timeInterp,self.dataInterp[:,2],'r', linewidth=1.0)
            plt.plot(self.timeInterp,self.angle2, 'b', linewidth = 1.0)
            plt.plot(self.timeInterp,self.angle, 'g', linewidth = 1.0)
            plt.xlabel('Time [ms]')
            plt.grid()             
                
    def normaliseData(self, minTime, maxTime): 
        ## Normalize data to happen every 7.5ms ##
        xData = self.timeNum
        yData = self.allData[:,3:]
#        print("Min xData: {}, Max xData: {}".format(min(xData), max(xData)))
#        print("MinTime: {}, MaxTime: {}".format(minTime, maxTime))
        fxInterp = interp1d(xData, yData, axis = 0)
  
        newX = np.arange(minTime, maxTime, 7.5)
        #print("newx",newX)
        self.dataInterp = fxInterp(newX)
        self.timeInterp = newX-newX[0]
        
#        print(len(self.timeStamp),self.timeStamp[0],self.timeStamp[-1])
#        print(len(self.timeInterp),self.timeInterp[0],self.timeInterp[-1])
        
        
    def angleEstimation(self):
            
        # defining process covariance -> bias and accelerometer assumed to be 
        # independent:     
        Q_angle = 0.01;     # acceleromter variance
        Q_bias = 0.005;      # bias variance
           
        # measurement variance
        R_measure = 0.1;

        P = np.zeros((2,2))  # error covariance matrix
        #P[0][0] = 1 # for now I don't know the initial state so i should initialize this with a high number?
        #P[1][1] = 1
        
        dt = 0.0075     # use a fixed time interval as the data is interpolated to this;
        # TODO: consider using the raw data and a varying dt
            
        self.angle2   = np.zeros(len(self.dataInterp))
        self.angle    = np.zeros(len(self.dataInterp))
        self.gyroBias = np.zeros(len(self.dataInterp))
        
        #check how it looks like without a kalman filter
        for i in range(1,len(self.dataInterp)):
            gyroRate = self.dataInterp[i,2]   
            self.angle2[i] = self.angle2[i-1] + gyroRate*dt   
            #if i < 100:
            #    print(i, gyroRate, self.angle2[i])
        
        # iterate over all data points
        for i in range(1,len(self.dataInterp)):
        
            #if (i > 10000):
            #    break
            # Calculate the angle from accelerometers
            angleAcc = m.degrees(m.atan2(self.dataInterp[i,3],self.dataInterp[i,4]))
            
            # Get current gyro value
            gyroRate = self.dataInterp[i,2]
            #print(i,angleAcc,gyroRate)
            
            ## Kalman Filter - Predict ##
                
            #Step 1: Get a priori estimate based on x(k+1) = F*x(k)+B*rate
            rate  = gyroRate - self.gyroBias[i-1]
            self.angle[i] = self.angle[i-1] + dt*rate
            
            #Step 2: update estimation of covariance
            P[0][0] = P[0][0] + dt * (dt*P[1][1] - P[0][1] - P[1][0] + Q_angle);
            P[0][1] = P[0][1] - dt * P[1][1];
            P[1][0] = P[1][0] - dt * P[1][1];
            P[1][1] = P[1][1] + Q_bias * dt;
        
            ## Kalman - Filter Update ##
        
            # Step 3: compute innovation and it's covariance
            y = angleAcc - self.angle[i]
            S = P[0][0] + R_measure
        
            # Step 4: Kalman gain
            K = 1/S*P[0]
        
            # Step 5: Update the posteriori state estimate and error covariance
            self.angle[i]    = self.angle[i] + K[0]*y
            self.gyroBias[i] = self.gyroBias[i] + K[1]*y
            
            P[0][0] = P[0][0] - K[0]*P[0][0]
            P[0][1] = P[0][0] - K[0]*P[0][1]
            P[1][0] = P[0][0] - K[1]*P[0][0]
            P[1][1] = P[1][1] - K[1]*P[0][1]            

#==============================================================================
# MAIN
#==============================================================================
if __name__ == "__main__":
    plt.close('all')
    IMUs = {} # Dictionary of all IMU objects
    
    #specify directory and files 
    folder = r"D:\imuValidation\Try6"
    filesLevel = r"\MartinLevel*.csv" ## File names with all data
    filesAscend = r"\MartinAscend_*.csv"
    filesDescend = r"\MartinDescend_*.csv"
    filesValidation = r"\imuValidation_*.csv"
    
    #convert it to a list of file records
    #allFls  = gleb.glob(path.join(folder,filesLevel))
    allFls  = gleb.glob(folder+filesValidation)
    #allFls += gleb.glob(path.join(folder,filesAscend))
    #allFls += gleb.glob(path.join(folder,filesDescend))
    
    #create instance for file & inital data processing
    imuDataRecord = imuDataFile()
    nFileIdx,nExpIdx = 1,1
    
    maxSamples = 9e9
    
    for i in range(0, len(allFls)):
        
        print("\nProcessing file: {0}".format(allFls[i]))
        # Instantiate an IMU object for each individual experiment #
        lstObj = imuDataRecord.processFileRecord("IMU{}".format(str(nFileIdx)),allFls[i])
        
        print("Exp{0}_IMU{1}".format(str(nExpIdx),str(nFileIdx)))
        if len(lstObj) == 2:
            IMUs["Exp{0}_IMU{1}".format(str(nExpIdx),str(nFileIdx))] = lstObj[0]
            IMUs["Exp{0}_IMU{1}".format(str(nExpIdx+1),str(nFileIdx))] = lstObj[1]
        elif len(lstObj) == 1:
            IMUs["Exp{0}_IMU{1}".format(str(nExpIdx),str(nFileIdx))] = lstObj[0]
        
        nFileIdx += 1 
                
        if nFileIdx == 6: # all files processed of one recording
            nFileIdx = 1
            nExpIdx += 1 if nExpIdx == 1 else 2
    
    del imuDataRecord
    
    #debug purposes to figure out, which IMU belongs to a certain position in a recording 
    #for i in range(0,len(allFls)):
    #    key = list(IMUs.keys())[i]
    #    print(key)
    #    IMUs[key].debug()
    
    # Test everything for one/two object instance(s)
    # Config for Level from Kai: IMU1 = COM IMU3 = left thigh, IMU5 = left ankle
    #IMUs["Exp1_IMU5"].normaliseData()
    #IMUs["Exp1_IMU5"].angleEstimation()
    #IMUs["Exp1_IMU5"].plotImuData(2)
    #plt.show
    
    allKeys = []    
    minTimeExp1 = 0
    maxTimeExp1 = 1e9
    minTimeExp2 = 0
    maxTimeExp2 = 1e9
    
    for key in IMUs:
        allKeys.append(key)
        if "Exp2" in key:
            minTimeExp2 = max(minTimeExp2, min(IMUs[key].timeNum))
            maxTimeExp2 = min(maxTimeExp2, max(IMUs[key].timeNum))
        else:   
            minTimeExp1 = max(minTimeExp1, min(IMUs[key].timeNum))
            maxTimeExp1 = min(maxTimeExp1, max(IMUs[key].timeNum))
    
    exps = ["Exp1", "Exp2"]
    for j in range(0,len(exps)):
        for keys in IMUs:
            if exps[j] not in keys:
                continue
            # Interpolate all IMUs to a same x-axis #
            if "Exp2" in keys:
                IMUs[keys].normaliseData(minTimeExp2, maxTimeExp2)
            else:
                IMUs[keys].normaliseData(minTimeExp1, maxTimeExp1)
    #for i in range(0, len(IMUs)):
         
        # Find the common maximum - minimum time (replace once the trigger works)
    #    if IMUs["IMU{}".format(str(i+1))].twoStageExp:
    #        minTimeExp1 = max(minTimeExp1, min(IMUs["IMU{}".format(str(i+1))].timeNumExp1))
    #        minTimeExp2 = max(minTimeExp2, min(IMUs["IMU{}".format(str(i+1))].timeNumExp2))
    #        maxTimeExp1 = min(maxTimeExp1, max(IMUs["IMU{}".format(str(i+1))].timeNumExp1))
    #        maxTimeExp2 = min(maxTimeExp2, max(IMUs["IMU{}".format(str(i+1))].timeNumExp2))
    #    else:
    #        minTime = max(minTime, min(IMUs["IMU{}".format(str(i+1))].timeNum))
    #        maxTime = min(maxTime, max(IMUs["IMU{}".format(str(i+1))].timeNum))
        
    #for i in range(0, len(IMUs)):
    #    if IMUs["IMU{}".format(str(i+1))].twoStageExp:
    #        IMUs["IMU{}".format(str(i+1))].normaliseData(minTimeExp1, maxTimeExp1, 1)
    #        IMUs["IMU{}".format(str(i+1))].normaliseData(minTimeExp2, maxTimeExp2, 2)
    #    else:
    #        IMUs["IMU{}".format(str(i+1))].normaliseData(minTime, maxTime, 0)
    #
    #    IMUs["IMU{}".format(str(i+1))].plotImuData()
    #plt.show()