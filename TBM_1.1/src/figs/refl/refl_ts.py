# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 11:20:30 2022

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
import datetime
from matplotlib.dates import MonthLocator, DateFormatter, drange
import matplotlib.dates as mdates 

n0 = joblib.load("../../../data/output/model/refl_ci1_HARV.pkl")
n1 = joblib.load("../../../data/output/model/refl_ci1_HARV.pkl")

n0r, n0n = n0[:,0], n0[:,1]
n1r, n1n = n1[:,0], n1[:,1]

ns = [n0r, n0n, n1r, n1n]
    
v0 = pd.read_csv("../../../data/verify/HARV_brf.csv")
v1 = pd.read_csv("../../../data/verify/HARV_brf.csv")

v0r = np.array(v0['sur_refl_b01'])
v0n = np.array(v0['sur_refl_b02'])
v1r = np.array(v1['sur_refl_b01'])
v1n = np.array(v1['sur_refl_b02'])

vs = [v0r, v0n, v1r, v1n]

d0 = v0.apply(lambda x: mdates.date2num(datetime.datetime.strptime("{0}{1}{2}".format(int(x['year']), int(x['month']), int(x['day'])), '%Y%m%d')), axis=1)
d1 = v1.apply(lambda x: mdates.date2num(datetime.datetime.strptime("{0}{1}{2}".format(int(x['year']), int(x['month']), int(x['day'])), '%Y%m%d')), axis=1)

ds = [d0, d0, d1, d1]

fig, axs = plt.subplots(1, 2, figsize=(15, 5))    

nx = 3 #y轴刻度个数
ny = 3 #y轴刻度个数

linewidth = 1.8 #边框线宽度
ftsize = 20 #字体大小
axlength = 3.0 #轴刻度长度
axwidth = 2.0 #轴刻度宽度
ftfamily = 'Calibri'
legendcols = 5 #图例一行的个数

labels = ["Red Band", "NIR Band", "UNDE Red", "UNDE NIR"]
for i in range(0, 2):
    
    xd = np.array(ns[i])
    yd = np.array(vs[i])    

    date = ds[i]
    
    arr = np.hstack((xd.reshape(-1,1), yd.reshape(-1,1)))  
    arr = arr[~np.isnan(arr).any(axis=1)]
    x, y = arr[:,0], arr[:,1]    
    
    r, p = stats.pearsonr(x, y)   
    abs_sum = sum(abs(x - y))
    abs_num = len(x)
    mae = (abs_sum / abs_num)
    
    if i == 0:
        min_y = 0.0
        max_y = 0.2       
    else:
        min_y = 0.0
        max_y = 1.0

    R1 = axs[i].scatter(date, xd, marker='x', color='red',  s=40, label='MODIS Reflectance')          
    R2 = axs[i].scatter(date, yd, marker='o', color='None', s=40, edgecolors="black", label='Model Simulations')
    
    if i == 0 :
        axs[i].text(date[10],  0.22, "{0}:\nR\u00B2={1}, MAE={2}".format(labels[i], round(r*r, 2), round(mae, 2)), fontsize=ftsize*1.3)                              
    else:
        axs[i].text(date[10],  0.80, "{0}:\nR\u00B2={1}, MAE={1}".format(labels[i], round(r*r, 2), round(mae, 2)), fontsize=ftsize*1.3)                              


    axs[i].set_ylim(min_y, max_y+0.1)  
    #axs[m, n].set_xlim(date[0], date[len(date)-1])
    
    axs[i].xaxis.set_major_locator(MonthLocator([6,12]))
    axs[i].xaxis.set_major_formatter(DateFormatter('%Y-%m'))
        
    axs[i].fmt_xdata = DateFormatter('%Y-%m')
    fig.autofmt_xdate()
    #fig.autofmt_xdate(bottom=0.3, rotation=0, ha='center')
    
    axs[i].set_yticks(np.linspace(min_y, max_y, ny))
    axs[i].set_ylabel("Reflectance", fontsize = ftsize*1.5, family = ftfamily, labelpad=10)
    #axs[i].set_title("({0}) {1}".format(chr(97+i), labels[i]), fontsize = ftsize*1.6)

    axs[i].spines['left'].set_linewidth(linewidth)
    axs[i].spines['right'].set_linewidth(linewidth)
    axs[i].spines['top'].set_linewidth(linewidth)
    axs[i].spines['bottom'].set_linewidth(linewidth)
    axs[i].tick_params(direction = 'in', axis='both', length = axlength, width = axwidth, labelsize = ftsize)        
    
    if i == 1:
        handles = [R1, R2]
        labels = ["Simulations", "MODIS reflectance"] 
        
    fig.add_subplot(111, frame_on=False)
    plt.tick_params(labelcolor="none", bottom=False, left=False)
    
#plt.xlabel("Date", fontsize = ftsize*1.5, family = ftfamily, labelpad=40)
#plt.ylabel("Reflectance", fontsize = ftsize*1.5, family = ftfamily, labelpad=30)
fig.tight_layout()          

fig.legend(handles, labels, loc ='lower center', fancybox = False, shadow = False,frameon = False, 
          ncol = legendcols, handletextpad = 0.4, columnspacing = 0.5, prop={'family':ftfamily, 'size':ftsize*1.2})  
fig.subplots_adjust(left = None, right = None, bottom = 0.3)
     
plt.show()
plot_path = "../../../figs/refl/refl_ts1.jpg"
fig.savefig(plot_path, dpi=600, bbox_inches = 'tight')    