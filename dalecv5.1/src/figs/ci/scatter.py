# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 11:02:42 2021

@author: hliu

scatter plots of the reflectance based on different ci

"""

import os 
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('font',family='Calibri')

from scipy.stats import gaussian_kde
import warnings
warnings.filterwarnings('ignore')

import joblib
import scipy.stats as stats 

r0 = joblib.load("../../../data/output/model/refl_ci0_UNDE.pkl")
r1 = joblib.load("../../../data/output/model/refl_ci1_UNDE.pkl")
r2 = joblib.load("../../../data/output/model/refl_ci2_UNDE.pkl")
rs = [r0, r1, r2]
    
brf_data = joblib.load("../../../data/output/verify/d_refl0_UNDE.pkl")

n0 = joblib.load("../../../data/output/model/nee_ci0_UNDE.pkl")
n1 = joblib.load("../../../data/output/model/nee_ci1_UNDE.pkl")
n2 = joblib.load("../../../data/output/model/nee_ci2_UNDE.pkl")
ns = [n0, n1, n2]
    
nee = joblib.load("../../../data/output/verify/d_nee0_UNDE.pkl")

fig, axs = plt.subplots(3, 3, figsize=(10, 10))    

nx = 3 #y轴刻度个数
ny = 3 #y轴刻度个数

linewidth = 1.0 #边框线宽度
ftsize = 15 #字体大小
axlength = 2.0 #轴刻度长度
axwidth = 1.2 #轴刻度宽度
ftfamily = 'Calibri'

labels = [["No clumping Red", "Clumping Red", "Angle clumping Red"], 
          ["No clumping NIR", "Clumping NIR", "Angle clumping NIR"],
          ["No clumping NEE", "Clumping NEE", "Angle clumping NEE"]]
for i in range(0, 3):
    for j in range(0, 3):
        if i == 0 or i == 1:
            x = np.array(brf_data["sur_refl_b0{0}".format(i+1)])
            y = np.array(rs[j][:,i])
        elif i == 2:
            x = np.array(nee["nee"][0:26280])
            y = np.array(ns[j])    
        arr = np.hstack((x.reshape(-1,1), y.reshape(-1,1)))  
        arr = arr[~np.isnan(arr).any(axis=1)]
        x, y = arr[:,0], arr[:,1]    
        k, b = np.polyfit(x, y, 1)

        if i == 0:
            min_x = 0
            min_y = 0
            max_x = 0.1        
            max_y = 0.1
        elif i == 1:
            min_x = 0
            min_y = 0
            max_x = 0.7        
            max_y = 0.7    
        elif i == 2:
            min_x = -30.0
            min_y = -30.0
            max_x = 30.0       
            max_y = 30.0

        axs[i][j].plot(range(int(min_x),int(max_x)+2), k*range(int(min_x),int(max_x)+2) + b, 'red')
        axs[i][j].scatter(x, y, marker='o', color = 'None', s = 20, edgecolors="black")     
            
        axs[i][j].set_xlim(min_x, max_x)
        axs[i][j].set_ylim(min_y, max_y)  
        axs[i][j].plot(axs[i][j].get_xlim(), axs[i][j].get_ylim(), ls="--", c=".1") 
        
        axs[i][j].set_xticks(np.linspace(min_x, max_x, nx))
        axs[i][j].set_yticks(np.linspace(min_y, max_y, ny))
        axs[i][j].set_title("({0}) {1}".format(chr(97+i*3+j), labels[i][j]), fontsize = ftsize*1.2)

        axs[i][j].annotate('k={0}, R\u00b2={1}\nRMSE={2}'.format(round(k,2), 
                                                                round(stats.pearsonr(x, y)[0]**2,2), 
                                                                round(np.sqrt(((x-y)**2).mean()),2)), 
                                                                (0.02, 0.72), xycoords='axes fraction', fontsize = ftsize)    
        axs[i][j].spines['left'].set_linewidth(linewidth)
        axs[i][j].spines['right'].set_linewidth(linewidth)
        axs[i][j].spines['top'].set_linewidth(linewidth)
        axs[i][j].spines['bottom'].set_linewidth(linewidth)
        axs[i][j].tick_params(direction = 'in', axis='both', length = axlength, width = axwidth, labelsize = ftsize)        
        
        fig.add_subplot(111, frame_on=False)
        plt.tick_params(labelcolor="none", bottom=False, left=False)
        
plt.xlabel("MODIS reflectance/NEON NEE", fontsize = ftsize*1.5, family = ftfamily, labelpad=15)
plt.ylabel("Model simulations", fontsize = ftsize*1.5, family = ftfamily, labelpad=15)

fig.tight_layout()          
     
plt.subplots_adjust(wspace =0.3, hspace =0.3)#调整子图间距    
plt.show()
plot_path = "../../../figs/ci/scatter.jpg"
fig.savefig(plot_path, dpi=600, bbox_inches = 'tight')
