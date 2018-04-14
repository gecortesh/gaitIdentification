# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 10:17:52 2018

@author: gabych
"""

import numpy as np
from keras.models import Sequential
from keras.layers import Conv1D, Dense
from keras import backend as K
import wfdb, sys

dataset = wfdb.dl_database(db_dir='gaitpdb', dl_dir='/gaitpdb' )
