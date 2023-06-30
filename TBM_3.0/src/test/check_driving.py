# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 21:02:54 2022

@author: hliu
"""
import pandas as pd 
site = "HARV"
data1 = pd.read_csv("../../data/driving/{0}.csv".format(site), na_values="nan") 
data2 = data1.groupby(['year', 'doy']).mean()

import matplotlib.pyplot as plt
plt.figure(figsize=(12, 3))
data2['TA'].plot()
plt.figure(figsize=(12, 3))
data2['SW'].plot()
plt.figure(figsize=(12, 3))
data2['PAR_up'].plot()
plt.figure(figsize=(12, 3))
data2['VPD'].plot()