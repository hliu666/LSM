# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 11:20:30 2022

@author: hliu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('font',family='Calibri')
import scipy.stats as stats
import warnings
warnings.filterwarnings('ignore')

import joblib
import datetime
from matplotlib.dates import MonthLocator, DateFormatter, drange
import matplotlib.dates as mdates 

n0 = joblib.load("../../../data/output/model/refl_ci1_HARV.pkl")
n0r, n0n, n0sw1, n0sw2 = n0[:,0], n0[:,1], n0[:,-2], n0[:,-1]
n0ndvi = (n0n-n0r)/(n0n+n0r)
n0nirv = (n0n-n0r)/(n0n+n0r)*n0n
ns = [n0r, n0n, n0ndvi, n0nirv, n0sw1, n0sw2]
    
v0 = pd.read_csv("../../../data/verify/HARV_brf.csv")
v0r, v0n, v0sw1, v0sw2 = np.array(v0['sur_refl_b01']), np.array(v0['sur_refl_b02']), np.array(v0['sur_refl_b06']), np.array(v0['sur_refl_b07'])
v0ndvi = (v0n-v0r)/(v0n+v0r)
v0nirv = (v0n-v0r)/(v0n+v0r)*v0n
vs = [v0r, v0n, v0ndvi, v0nirv, v0sw1, v0sw2]

d0 = v0.apply(lambda x: mdates.date2num(datetime.datetime.strptime("{0}{1}{2}".format(int(x['year']), int(x['month']), int(x['day'])), '%Y%m%d')), axis=1)
ds = [d0, d0, d0, d0, d0, d0]

fig, axs = plt.subplots(3, 2, figsize=(12, 12))    

nx = 3 #y轴刻度个数
ny = 3 #y轴刻度个数

linewidth = 1.8 #边框线宽度
ftsize = 18 #字体大小
axlength = 3.0 #轴刻度长度
axwidth = 2.0 #轴刻度宽度
legendcols = 5 #图例一行的个数
ftfamily = 'Calibri'

titles = ["Red", "NIR", "NDVI", "NIRv", "SW1", "SW2"]
for i in range(0, 6):
    m, n = int(i/2), i%2
    
    xd = np.array(ns[i])
    yd = np.array(vs[i])    

    date = ds[i]
    
    arr = np.hstack((xd.reshape(-1,1), yd.reshape(-1,1)))  
    arr = arr[~np.isnan(arr).any(axis=1)]
    x, y = arr[:,0], arr[:,1]    
    """
    statistical indices
    """
    r, p = stats.pearsonr(x, y) 

    abs_sum = sum(abs(x - y))
    abs_num = len(x)
    mae = (abs_sum / abs_num)
    
    if n == 0 and m == 0:
        min_y = 0.0
        max_y = 0.2       
    elif n == 1 and m == 0:
        min_y = 0.0
        max_y = 1.0
    elif m == 1 and n == 0:
        min_y = 0.2
        max_y = 1.0    
    else:
        min_y = 0.0
        max_y = 0.5
        
    R1 = axs[m, n].scatter(date, yd, marker='x', color='red',  s=40, label='MODIS Reflectance')                           
    R2 = axs[m, n].scatter(date, xd, marker='o', color='None', s=40, edgecolors="black", label='Model Simulations')
    
    if n == 0 and m == 0:
        axs[m, n].text(date[10],  0.25, "R\u00B2={0}, p<0.01, MAE={1}".format(round(r*r, 2), round(mae, 2)), fontsize=ftsize*1.3)                              
    elif n == 1 and m == 0:
        axs[m, n].text(date[10],  0.91, "R\u00B2={0}, p<0.01, MAE={1}".format(round(r*r, 2), round(mae, 2)), fontsize=ftsize*1.3)                              
    elif m == 1 and n == 0:
        axs[m, n].text(date[10],  0.95, "R\u00B2={0}, p<0.01, MAE={1}".format(round(r*r, 2), round(mae, 2)), fontsize=ftsize*1.3)                              
    else:
        axs[m, n].text(date[10],  0.5, "R\u00B2={0}, p<0.01, MAE={1}".format(round(r*r, 2), round(mae, 2)), fontsize=ftsize*1.3)                              

    axs[m, n].set_ylim(min_y, max_y+0.1)  
    #axs[m, n].set_xlim(date[0], date[len(date)-1])
    
    axs[m, n].xaxis.set_major_locator(MonthLocator([6,12]))
    axs[m, n].xaxis.set_major_formatter(DateFormatter('%Y-%m'))
        
    axs[m, n].fmt_xdata = DateFormatter('%Y-%m')
    fig.autofmt_xdate()
    #fig.autofmt_xdate(bottom=0.3, rotation=0, ha='center')
    
    #if i == 0:
    #    axs[m, n].legend(loc='upper right', fancybox = False, shadow = False,frameon = False, ncol = 2, fontsize=ftsize/1.2) 
        
    axs[m, n].set_yticks(np.linspace(min_y, max_y+0.1, ny))
    axs[m, n].set_title("({0}) {1}".format(chr(97+i), titles[i]), fontsize = ftsize*1.5)

    axs[m, n].spines['left'].set_linewidth(linewidth)
    axs[m, n].spines['right'].set_linewidth(linewidth)
    axs[m, n].spines['top'].set_linewidth(linewidth)
    axs[m, n].spines['bottom'].set_linewidth(linewidth)
    axs[m, n].tick_params(direction = 'in', axis='both', length = axlength, width = axwidth, labelsize = ftsize)        
    
    handles = [R1, R2]
    labels = ["Simulations", "MODIS reflectance"] 
    
    fig.add_subplot(111, frame_on=False)
    plt.tick_params(labelcolor="none", bottom=False, left=False)
    
plt.xlabel("Date", fontsize = ftsize*1.5, family = ftfamily, labelpad=40)
plt.ylabel("Reflectance", fontsize = ftsize*1.5, family = ftfamily, labelpad=30)
fig.tight_layout()          
plt.subplots_adjust(wspace =0.15, hspace =0.18)#调整子图间距      
fig.legend(handles, labels, loc ='lower center', fancybox = False, shadow = False,frameon = False, 
          ncol = legendcols, handletextpad = 0.4, columnspacing = 0.5, prop={'family':ftfamily, 'size':ftsize*1.2})  
fig.subplots_adjust(left = None, right = None, bottom = 0.15)
 

plt.show()
plot_path = "../../../figs/refl/refl_ts.jpg"
fig.savefig(plot_path, dpi=600, bbox_inches = 'tight')    