# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 17:04:46 2022

@author: 16072
"""
import numpy as np
import joblib
import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('font',family='Calibri')

import warnings
warnings.filterwarnings('ignore')

sites = ['UNDE, DBF', 'HARV, DBF']
sitens = ['1_HARV', '1_HARV']
fig, axs = plt.subplots(2, 1, figsize=(10, 10))    

for i in range(0, 2):
    site = sites[i]
    print(sitens[i])
    lst_m = joblib.load("../../../data/output/model/lst_ci{0}.pkl".format(sitens[i]))
    lst_s = joblib.load("../../../data/output/verify/d_lst{0}.pkl".format(sitens[i]))
    
    linewidth = 2.0 #边框线宽度
    ftsize = 16 #字体大小
    axlength = 2.0 #轴刻度长度
    axwidth = 2.0 #轴刻度宽度
    ftfamily = 'Calibri'
                   
    subplt = axs[i]
    subplt.set_ylim(-20,40) 

    subplt.spines['left'].set_linewidth(linewidth)
    subplt.spines['right'].set_linewidth(linewidth)
    subplt.spines['top'].set_linewidth(linewidth)
    subplt.spines['bottom'].set_linewidth(linewidth)
    subplt.tick_params(direction = 'in', axis='both', length = axlength, width = axwidth, labelsize = ftsize*1.1)
     
    date = [datetime.datetime(int(x[0]), int(x[1]), int(x[2]), int(x[3]), 0)  for x in np.array(lst_s)]
    subplt.set_title('({0}) {1}'.format(chr(97+i), site), fontsize=ftsize*1.6, family=ftfamily)                             
    subplt.scatter(date,  lst_m[lst_s['index'].astype(int)], marker="o",  c = "black", zorder=1, label='Model Simulations')  
    subplt.scatter(date,  lst_s["LST_Day_1km"],  marker="x", c = "r", s = 60,  zorder=3, label='MODIS LST')            
    if  i == 0:
        subplt.xaxis.set_visible(False)
        subplt.legend(loc='upper right', fancybox = False, shadow = False,frameon = False, ncol = 2, fontsize=ftsize) 
    for tick in subplt.get_xticklabels():
        tick.set_rotation(30)

fig.add_subplot(111, frame_on=False)
plt.tick_params(labelcolor="none", bottom=False, left=False)  
plt.xlabel("Date", fontsize=ftsize*1.6, family = ftfamily, labelpad=50) 
plt.ylabel("Land surface temperature (degree)", fontsize=ftsize*1.6, family = ftfamily, labelpad=15)  
fig.tight_layout() 
fig.subplots_adjust(bottom=0.12)  
    
plot_path = "../../../figs/lst/lst_ts.jpg"
plt.show()
fig.savefig(plot_path, dpi=300, bbox_inches = 'tight')