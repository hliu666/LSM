# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 11:02:42 2021

@author: hliu

"""
import os 
import pandas as pd
import numpy as np
import datetime

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('font',family='Calibri')

from scipy.stats import gaussian_kde
import warnings
warnings.filterwarnings('ignore')

os.chdir('C:/Users/liuha/Desktop/dalecv4.3/src/sa')

import joblib
s1 = joblib.load("sensitivity_fpar_leaftraits.pkl")
s2 = joblib.load("sensitivity_nee_leaftraits.pkl")
s3 = joblib.load("sensitivity_lai_leaftraits.pkl")

mean1 = np.array([s1['S1'],      s1['ST']])
err1  = np.array([s1['S1_conf'], s1['ST_conf']])
mean2 = np.array([s2['S1'],      s2['ST']])
err2  = np.array([s2['S1_conf'], s2['ST_conf']])
mean3 = np.array([s3['S1'],      s3['ST']])
err3  = np.array([s3['S1_conf'], s3['ST_conf']])

fig, axs = plt.subplots(3, 1, figsize=(4, 4))

for mean, err, ylabel, ax in zip([mean1, mean2, mean3], [err1, err2, err3], ['fPAR', 'NEE', 'LAI'], [axs[0], axs[1], axs[2]]):
    
    x_labels = ["Cab", "Car", "Cbrown", "Cw", "Cm", "Ant", "Alpha"]
    x = np.arange(int(len(x_labels)))
    
    bar_width = 0.3 #柱状体宽度
    capsize = 1.2 #柱状体标准差参数1
    capthick = 0.8  #柱状体标准差参数2
    elinewidth = 0.8 #柱状体标准差参数3
    linewidth = 1.0 #边框线宽度
    axlength = 2.0 #轴刻度长度
    axwidth = 1.2 #轴刻度宽度
    legendcols = 5 #图例一行的个数
    ftsize = 12 #字体大小
    ftfamily = "Calibri"
    
    #min_y1 = 0
    #max_y1 = 1
    
    S1_Color = 'red'
    ST_Color = 'blue'
    
    R1 = ax.bar(x + 0 * bar_width, mean[0,:], bar_width, yerr = err[0,:],  error_kw = {'ecolor' : '0.2', 'elinewidth':elinewidth, 'capsize' :capsize, 'capthick' :capthick}, color=S1_Color, label = "First order index", align="center", alpha=1)
    R2 = ax.bar(x + 1.2 * bar_width, mean[1,:], bar_width, yerr = err[1,:],  error_kw = {'ecolor' : '0.2', 'elinewidth':elinewidth, 'capsize' :capsize, 'capthick' :capthick}, color=ST_Color,  label = "Total order index", align="center", alpha=1)
    
    if ax == axs[0]:
        ax.set_title("Sensitivity Analysis", fontsize = ftsize/1.2, family = ftfamily)  
    #axes.set_ylabel('GPP mol CO₂m\u207B\u00B2yr\u207B\u00B9', fontsize = ftsize/1.5, family=ftfamily)
    ax.set_ylabel(ylabel, fontsize = ftsize/1.5, family=ftfamily)
    if ax == axs[2]:
        ax.set_xlabel('Parameters', fontsize = ftsize/1.5, family=ftfamily)
    
    ax.set_xticks(x + 1 * bar_width)
    ax.set_xticklabels(x_labels, fontsize = ftsize/1.5, family = ftfamily)
    
    ax.spines['left'].set_linewidth(linewidth)
    ax.spines['right'].set_linewidth(linewidth)
    ax.spines['top'].set_linewidth(linewidth)
    ax.spines['bottom'].set_linewidth(linewidth)
    ax.tick_params(axis='both', length = axlength, width = axwidth, labelsize = ftsize/1.5)
    
    handles = [R1, R2]
       
    labels = ["First order index",\
              "Total order index"] 
    
fig.tight_layout() 
fig.legend(handles, labels, loc ='lower center', fancybox = False, shadow = False,frameon = False, 
          ncol = legendcols, handletextpad = 0.2, columnspacing = 1.5, prop={'family':ftfamily, 'size':ftsize/1.5})  
fig.subplots_adjust(left = None, right = None, bottom = 0.16)

plot_path = "C:/Users/liuha/Desktop/dalecv4.3/figs/sa/hist3.jpg"
plt.show()
fig.savefig(plot_path, dpi=600, quality=100,bbox_inches='tight')
