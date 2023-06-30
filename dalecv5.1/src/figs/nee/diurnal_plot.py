# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 00:35:06 2022

@author: hliu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('font',family='Calibri')

import warnings
warnings.filterwarnings('ignore')

import joblib
import datetime
from matplotlib.dates import HourLocator, DateFormatter, drange
import matplotlib.dates as mdates 

n0 = joblib.load("../../../data/output/model/nee_ci1_HARV.pkl")
n1 = joblib.load("../../../data/output/model/nee_ci1_HARV.pkl")
ns = [n0, n1]
    
v0 = np.array(pd.read_csv("../../../data/verify/HARV_nee.csv")['nee'])
v1 = np.array(pd.read_csv("../../../data/verify/HARV_nee.csv")['nee'])
vs = [v0, v1]

fig, axs = plt.subplots(2, 1, figsize=(12, 10))    

nx = 3 #y轴刻度个数
ny = 3 #y轴刻度个数

linewidth = 1.8 #边框线宽度
ftsize = 20 #字体大小
axlength = 3.0 #轴刻度长度
axwidth = 2.0 #轴刻度宽度
ftfamily = 'Calibri'

startDate = datetime.datetime(2019, 8, 1)
startDays = (startDate-datetime.datetime(2019, 1, 1)).days
Winlength = 20

date1 = startDate
date2 = startDate + datetime.timedelta(days=Winlength) 
delta = datetime.timedelta(hours = 1)
dates = drange(date1, date2, delta)

labels = ["HARV", "UNDE"]
for i in range(0, 2):
    x = np.array(ns[i][0:26280])
    y = np.array(vs[i][0:26280])    
    
    xd = np.array(ns[i][startDays*24:(startDays+Winlength)*24])
    yd = np.array(vs[i][startDays*24:(startDays+Winlength)*24])

    arr = np.hstack((xd.reshape(-1,1), yd.reshape(-1,1)))  
    arr = arr[~np.isnan(arr).any(axis=1)]
    x, y = arr[:,0], arr[:,1]    
    k, b = np.polyfit(x, y, 1)     

    min_y = -15.0
    max_y = 10.0

    axs[i].scatter(dates, yd, marker='o', color='None', s=40, edgecolors="black")             
    axs[i].plot(dates, xd, 'red', linewidth=2)
    
    axs[i].set_ylim(min_y, max_y)  
    axs[i].set_xlim(dates[0], dates[-1])
    
    axs[i].xaxis.set_major_locator(mdates.HourLocator(interval = 3*24))
    axs[i].xaxis.set_major_formatter(DateFormatter('%j'))
        
    axs[i].fmt_xdata = DateFormatter('%j')
    fig.autofmt_xdate(bottom=0.3, rotation=0, ha='center')
    
    axs[i].set_yticks(np.linspace(min_y, max_y, ny))
    axs[i].set_title("({0}) {1}".format(chr(97+i), labels[i]), fontsize = ftsize*1.6)

    axs[i].spines['left'].set_linewidth(linewidth)
    axs[i].spines['right'].set_linewidth(linewidth)
    axs[i].spines['top'].set_linewidth(linewidth)
    axs[i].spines['bottom'].set_linewidth(linewidth)
    axs[i].tick_params(direction = 'in', axis='both', length = axlength, width = axwidth, labelsize = ftsize)        
    
    fig.add_subplot(111, frame_on=False)
    plt.tick_params(labelcolor="none", bottom=False, left=False)
        
plt.xlabel("Day of Year, 2019", fontsize = ftsize*1.5, family = ftfamily, labelpad=30)
plt.ylabel("NEE (gCm\u207B\u00B2d\u207B\u00B9)", fontsize = ftsize*1.5, family = ftfamily, labelpad=35)
fig.tight_layout()          
     
plt.subplots_adjust(wspace =0.3, hspace =0.18)#调整子图间距    
plt.show()
plot_path = "../../../figs/nee/nee_hts.jpg"
fig.savefig(plot_path, dpi=600, bbox_inches = 'tight')    
    