# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 20:09:15 2022

@author: hliu
"""

import pandas as pd
import numpy as np
import scipy.stats as stats

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('font',family='Calibri')

import warnings
warnings.filterwarnings('ignore')

import joblib

n = joblib.load("../../../data/output/model/nee_ci1_HARV.pkl")    
v = np.array(pd.read_csv("../../../data/verify/HARV_nee.csv")['nee'])

x = np.array(n[0:26280])
y = np.array(v[0:26280])    

arr = np.hstack((x.reshape(-1,1),y.reshape(-1,1)))
arr = arr[~np.isnan(arr).any(axis=1)]

arr = arr[~np.any(arr == 0, axis=1)]

X = arr[:,0]
y = arr[:,1] 

r, p = stats.pearsonr(X, y) 

abs_sum = sum(abs(X - y))
abs_num = len(X)
mae = (abs_sum / abs_num)

nrows = 1
ncols = 1 
fig, ax = plt.subplots(nrows = nrows, ncols = ncols, figsize=(6, 6))#, sharey=True)

ax.set_ylabel('Simulated NEE', fontsize=12)
ax.set_xlabel('Observed NEE', fontsize=12)
ax.scatter(X, y, facecolors='none', marker='o', edgecolors='black', s = 20) # 把 corlor 设置为空，通过edgecolors来控制颜色
ax.set(xlim=(-40, 15), ylim=(-40, 15))   
ax.set(xlim = ax.get_xlim(), ylim = ax.get_ylim())   

#ax.set_yticks(np.arange(0, 0.5, 0.1))  
#ax.set_xticks(np.arange(0, 0.81, 0.2))  
ax.plot(ax.get_xlim(), ax.get_ylim(), color = "red", ls="--")
ax.text(-20,  20, "$R^2$ = {0}, p<0.01\nMAE={1}".format(round(r*r, 2), round(mae, 2)), fontsize=10)                              
