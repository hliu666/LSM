# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 17:13:44 2022

@author: hliu
"""
import scipy.stats as stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('font',family='Calibri')

import warnings
warnings.filterwarnings('ignore')

import joblib
import datetime
from matplotlib.dates import MonthLocator, DateFormatter, drange
import matplotlib.dates as mdates 

n = joblib.load("../../../data/output/model/nee_ci1_HARV.pkl")    
v = np.array(pd.read_csv("../../../data/verify/HARV_nee.csv")['nee'])

date1 = datetime.datetime(2019, 1, 1)
date2 = datetime.datetime(2021, 12, 31)
delta = datetime.timedelta(days = 1)
dates = drange(date1, date2, delta)

fig, axs = plt.subplots(2, 1, figsize=(12, 10))   

nx = 3 #y轴刻度个数
ny = 3 #y轴刻度个数

linewidth = 1.8 #边框线宽度
ftsize = 20 #字体大小
axlength = 3.0 #轴刻度长度
axwidth = 2.0 #轴刻度宽度
legendcols = 5 #图例一行的个数
ftfamily = 'Calibri'

startDate = datetime.datetime(2019, 7, 1)
startDays = (startDate-datetime.datetime(2019, 1, 1)).days
Winlength = 15

label = "HARV"
for i in range(0, 2):
    x = np.array(n[0:26280])
    y = np.array(v[0:26280])    
        
    if i == 0:
        date1 = datetime.datetime(2019, 1, 1)
        date2 = datetime.datetime(2021, 12, 31)
        delta = datetime.timedelta(days = 1)
        dates = drange(date1, date2, delta)
        
        startDay1 = (date1-datetime.datetime(2019, 1, 1)).days
        ebdDay1   = (date2-datetime.datetime(2019, 1, 1)).days
        
        xd = np.array([np.nanmean(x[m: m+24]) for m in range(0, len(x), 24)])
        yd = np.array([np.nanmean(y[m: m+24]) for m in range(0, len(y), 24)])

        xd = xd[startDay1:ebdDay1]
        yd = yd[startDay1:ebdDay1]
        
        min_y = -20.0
        max_y = 15.0
        
    elif i == 1:
        date1 = startDate
        date2 = startDate + datetime.timedelta(days=Winlength) 
        delta = datetime.timedelta(hours = 1)
        dates = drange(date1, date2, delta)
        
        xd = np.array(n[startDays*24:(startDays+Winlength)*24])
        yd = np.array(v[startDays*24:(startDays+Winlength)*24])
        
        min_y = -50.0
        max_y = 25.0
    
    arr = np.hstack((xd.reshape(-1,1), yd.reshape(-1,1)))  
    arr = arr[~np.isnan(arr).any(axis=1)]
    x, y = arr[:,0], arr[:,1]    

    r, p = stats.pearsonr(x, y) 

    abs_sum = sum(abs(x - y))
    abs_num = len(x)
    mae = (abs_sum / abs_num)
    
    R1, = axs[i].plot(dates, xd, 'black', linewidth=2, label='Model Simulations')  
    R2 = axs[i].scatter(dates, yd, marker='o', color='None', s=40, edgecolors="red", label='NEON Observations')                
    
    axs[i].set_ylim(min_y, max_y)  
    axs[i].set_xlim(dates[0], dates[-1])

    if i == 0:    
        axs[i].set_xlabel("Date(Year-Month)", fontsize = ftsize*1.2, family = ftfamily, labelpad=5)
        axs[i].set_ylabel("NEE (gCm\u207B\u00B2d\u207B\u00B9)", fontsize = ftsize*1.2, family = ftfamily, labelpad=5)

        axs[i].xaxis.set_major_locator(MonthLocator([3,6,9,12]))
        axs[i].xaxis.set_major_formatter(DateFormatter('%Y-%m'))
        axs[i].xaxis.set_tick_params(rotation=30, labelsize=ftsize)
        
        axs[i].text(dates[-1]-400, 10, "R\u00B2={0} p<0.01 MAE={1}".format(round(r*r, 2), round(mae, 2)), fontsize=ftsize*1.2)                              

        #axs[i].fmt_xdata = DateFormatter('%Y-%m')
        #fig.autofmt_xdate()
        #axs[i].legend(loc='upper right', fancybox = False, shadow = False,frameon = False, ncol = 2, fontsize=ftsize) 
    elif i == 1: 
        axs[i].set_xlabel("Date(Year-Doy)", fontsize = ftsize*1.2, family = ftfamily, labelpad=5)
        axs[i].set_ylabel("NEE (gCm\u207B\u00B2h\u207B\u00B9)", fontsize = ftsize*1.2, family = ftfamily, labelpad=5)

        axs[i].xaxis.set_major_locator(mdates.HourLocator(interval = 3*24))
        axs[i].xaxis.set_major_formatter(DateFormatter('%Y-%j'))
        axs[i].xaxis.set_tick_params(rotation=30, labelsize=ftsize)
        #fig.autofmt_xdate(bottom=0.3, rotation=0, ha='center')         
        #axs[i].fmt_xdata = DateFormatter('%Y-%j')
        axs[i].text(dates[-1]-5.5, 15, "R\u00B2={0} p<0.01 MAE={1}".format(round(r*r, 2), round(mae, 2)), fontsize=ftsize*1.2)                              

    axs[i].set_yticks(np.linspace(min_y, max_y, ny))
    #if i == 0:
        #axs[i].set_title("({0}) {1}, DBF".format(chr(97+i), label), fontsize = ftsize*1.6)
        #axs[i].set_title("{0}, DBF".format(label), fontsize = ftsize*1.6)

    axs[i].spines['left'].set_linewidth(linewidth)
    axs[i].spines['right'].set_linewidth(linewidth)
    axs[i].spines['top'].set_linewidth(linewidth)
    axs[i].spines['bottom'].set_linewidth(linewidth)
    axs[i].tick_params(direction = 'in', axis='both', length = axlength, width = axwidth, labelsize = ftsize)        
    
    handles = [R1, R2]
    labels = ["Simulations", "Observations"] 
    
    fig.add_subplot(111, frame_on=False)
    plt.tick_params(labelcolor="none", bottom=False, left=False)
        
#plt.xlabel("Date", fontsize = ftsize*1.5, family = ftfamily, labelpad=50)
#plt.ylabel("NEE (gCm\u207B\u00B2d\u207B\u00B9)", fontsize = ftsize*1.5, family = ftfamily, labelpad=30)
fig.tight_layout() 
fig.legend(handles, labels, loc ='lower center', fancybox = False, shadow = False,frameon = False, 
          ncol = legendcols, handletextpad = 0.3, columnspacing = 1, prop={'family':ftfamily, 'size':ftsize})  
fig.subplots_adjust(left = None, right = None, bottom = 0.18)
     
plt.subplots_adjust(wspace =0.3, hspace =0.4)#调整子图间距    
plt.show()
plot_path = "../../../figs/nee/nee_ts.jpg"
fig.savefig(plot_path, dpi=600, bbox_inches = 'tight')