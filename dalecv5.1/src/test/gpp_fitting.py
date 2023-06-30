# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 14:45:17 2022

@author: hliu

Used for phenological constraints
"""

#import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data = pd.read_csv("../../data/verify/HARV_gpp.csv", na_values="nan") 
data = data[(data['year'] == 2019)]
data[(data['gpp'] < 0)] = 0.1
#plt.plot(data['gpp'], 'r+')

from scipy.signal import savgol_filter
fdata = savgol_filter(data['gpp'], 99, 2)
fdata = np.tile(fdata, 3)
fdata = np.insert(fdata, 730, fdata[730], axis=0)

#plt.plot(fdata)
#plt.plot(savgol_filter(data['gpp'], 121, 2))

np.savetxt("../../data/parameters/fgpp.txt", fdata, delimiter=",")
#test = np.genfromtxt("../../data/parameters/fgpp.txt")


        