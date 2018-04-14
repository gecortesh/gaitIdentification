# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 10:14:23 2018

@author: gabych
"""

import btk

reader = btk.btkAcquisitionFileReader()
reader.SetFilename('Level2.c3d')
reader.Update()
acq = reader.GetOutput()
acq.GetPointFrequency() # give the point frequency
acq.GetPointFrameNumber() # give the number of frames
metadata = acq.GetMetaData()