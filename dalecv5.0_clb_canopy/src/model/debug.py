# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 09:57:32 2022

@author: hliu
"""
import data_class as dc
import mod_class as mc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ci_flag = 1
site = "HARV"
pars = [0.7, 30]
d = dc.DalecData(2019, 2022, site, ci_flag, pars)
m = mc.DalecModel(d)
refl_y,lai_y = m.mod_list(d.lai, d.sai)
refl_y = refl_y[:,[0,1,5,6]]
 
brf_df = pd.read_csv("../../data/verify/HARV_brf.csv", na_values="nan") 
brf = np.array(brf_df[['sur_refl_b01', 'sur_refl_b02', 'sur_refl_b06', 'sur_refl_b07']])
    
fig = plt.figure(figsize=(8,6))
ax = plt.subplot(1,1,1)
ax.plot(refl_y[:,0],color='black',linestyle='solid', label='sur_refl_b01')
ax.plot(brf[:,0],'r.',markersize=3, label='Observation data')

fig = plt.figure(figsize=(8,6))
ax = plt.subplot(1,1,1)
ax.plot(refl_y[:,1],color='black',linestyle='solid', label='sur_refl_b02')
ax.plot(brf[:,1],'r.',markersize=3, label='Observation data')

fig = plt.figure(figsize=(8,6))
ax = plt.subplot(1,1,1)
ax.plot(refl_y[:,2],color='black',linestyle='solid', label='sur_refl_b06')
ax.plot(brf[:,2],'r.',markersize=3, label='Observation data')

fig = plt.figure(figsize=(8,6))
ax = plt.subplot(1,1,1)
ax.plot(refl_y[:,3],color='black',linestyle='solid', label='sur_refl_b07')
ax.plot(brf[:,3],'r.',markersize=3, label='Observation data')
