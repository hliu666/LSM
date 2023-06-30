# -*- coding: utf-8 -*-
"""
Created on Sun Jan  1 21:37:01 2023

@author: hliu
"""
import scipy.stats as stats
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def func(x, a, b, c, d):
    return a * np.log(b * x + c) - d

"""
Read photosynthesis, and stomata conductance 
"""
an  = joblib.load("../../../data/output/model/An_ci1_HARV.pkl")
re  = joblib.load("../../../data/output/model/Re_ci1_HARV.pkl")
gs  = 1/re
"""
Read air temperature and land surface temperature 
"""
lst = joblib.load("../../../data/output/model/lst_ci1_HARV.pkl")
flux_data = pd.read_csv("../../../data/driving/HARV.csv", na_values="nan") 
ata = np.array(flux_data['TA'])[0:26280]

date = np.array(flux_data[['hour', 'doy', 'month']])[0:26280]

det = lst-ata

y_lists, x_lists = [an, gs], [det, lst, ata]
y_labels = ["Photosynthesis ({0}mol CO₂m\u207B\u00B2s\u207B\u00B9)".format(chr(956)), 'Stomata Conductance (mmol m\u207B\u00B2s\u207B\u00B9)']
x_labels = ['Temperature differences (LST-TA)', 'Land surface temperature (LST)', 'Air temperature (TA)' ]


fig, ax = plt.subplots(2, 3, figsize=(28,16))

for i in range(len(y_lists)):
    for j in range(len(x_lists)):
        x, y = x_lists[j], y_lists[i]
        arr = np.hstack((x.reshape(-1,1), y.reshape(-1,1)))
        arr = np.hstack((arr, date))
        
        #arr = arr[(arr[:,4] >= 9) & (arr[:,4] <= 11)]

        arr = arr[arr[:,1] > 0]
        arr = arr[arr[:,1].argsort()] 
        #2: hour, 3: doy, 4:month
        xobs, yobs, cobs = arr[:,0], arr[:,1], arr[:,2]
      
        #popt, pcov = curve_fit(func, xobs, yobs)
        #yfits = func(xobs, *popt)
        
        r, p = stats.pearsonr(xobs, yobs) 
        
        nx = 3 #y轴刻度个数
        ny = 3 #y轴刻度个数
        
        linewidth = 1.8 #边框线宽度
        ftsize = 28 #字体大小
        axlength = 6.0 #轴刻度长度
        axwidth = 2.0 #轴刻度宽度
        legendcols = 5 #图例一行的个数
        ftfamily = 'Calibri'
        
        #ax[i,j].scatter(xobs, yobs, marker='o', color='None', s=40, edgecolors="black", label='Observations')                
        #p1 = ax[i,j].scatter(xobs, yobs, c=cobs, marker='o', s=40, cmap=plt.cm.coolwarm) 
        if i == 1:
            p1 = ax[i,j].hexbin(xobs, yobs*1000, C=cobs, cmap=plt.cm.coolwarm, gridsize=25, mincnt=3, reduce_C_function=np.mean)
        else:
            p1 = ax[i,j].hexbin(xobs, yobs, C=cobs, cmap=plt.cm.coolwarm, gridsize=25, mincnt=3, reduce_C_function=np.mean)
            
        cb = fig.colorbar(p1, ax=ax[i,j])
        cb.ax.yaxis.set_tick_params(labelsize=ftsize)
        
        #axs.set_xlabel("Photosynthesis ({0}mol CO₂m\u207B\u00B2s\u207B\u00B9)".format(chr(956)), fontsize = ftsize*1.2, family = ftfamily, labelpad=5)
        #axs.set_ylabel("Temperature differences (canopy-air)", fontsize = ftsize*1.2, family = ftfamily, labelpad=5)

        ax[i,j].set_xlabel(x_labels[j], fontsize = ftsize*1.2, family = ftfamily, labelpad=5)
        ax[i,j].set_ylabel(y_labels[i], fontsize = ftsize*1.2, family = ftfamily, labelpad=5)
         
        ax[i,j].spines['left'].set_linewidth(linewidth)
        ax[i,j].spines['right'].set_linewidth(linewidth)
        ax[i,j].spines['top'].set_linewidth(linewidth)
        ax[i,j].spines['bottom'].set_linewidth(linewidth)
        ax[i,j].tick_params(direction = 'in', axis='both', length = axlength, width = axwidth, labelsize = ftsize, color="black")        
        #print(x_labels[i], y_labels[j], round(r*r, 2))
        #axs.text(0, 5, "R\u00B2={0}, p<0.01".format(round(r*r, 2)), fontsize=ftsize*1.2) 
                             
fig.tight_layout()          
plot_path = "../../../figs/lst_ta/plot3.jpg"
fig.savefig(plot_path, dpi=600, bbox_inches = 'tight')    
