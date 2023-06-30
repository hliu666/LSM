# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 17:04:46 2022

@author: 16072
"""
import numpy as np
import joblib
import datetime
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('font',family='Calibri')

import warnings
warnings.filterwarnings('ignore')

sites = ['UNDE, DBF', 'HARV, DBF']
sitens = ['1_HARV', '1_HARV']
fig, axs = plt.subplots(1, 1, figsize=(10, 6))    

site = 'HARV, DBF'
lst_m = joblib.load("../../../data/output/model/lst_ci1_HARV.pkl")
lst_s = joblib.load("../../../data/output/verify/d_lst1_HARV.pkl")

linewidth = 2.0 #边框线宽度
ftsize = 18 #字体大小
axlength = 2.0 #轴刻度长度
axwidth = 2.0 #轴刻度宽度
legendcols = 5 #图例一行的个数
ftfamily = 'Calibri'
               
subplt = axs
subplt.set_ylim(-20,40) 

subplt.spines['left'].set_linewidth(linewidth)
subplt.spines['right'].set_linewidth(linewidth)
subplt.spines['top'].set_linewidth(linewidth)
subplt.spines['bottom'].set_linewidth(linewidth)
subplt.tick_params(direction = 'in', axis='both', length = axlength, width = axwidth, labelsize = ftsize*1.1)
 
date = [datetime.datetime(int(x[0]), int(x[1]), int(x[2]), int(x[3]), 0)  for x in np.array(lst_s)]
#subplt.set_title('({0}) {1}'.format(chr(97), site), fontsize=ftsize*1.6, family=ftfamily)                             
R1 = subplt.scatter(date,  lst_m[lst_s['index'].astype(int)], marker="o",  c = "black", zorder=1, label='Simulations')  
R2 = subplt.scatter(date,  lst_s["LST_Day_1km"],  marker="x", c = "r", s = 60,  zorder=3, label='MODIS LST')            
"""
statistical indices
"""
x = np.array(lst_m[lst_s['index'].astype(int)])
y = np.array(lst_s["LST_Day_1km"])

r, p = stats.pearsonr(x, y) 

abs_sum = sum(abs(x - y))
abs_num = len(x)
mae = (abs_sum / abs_num)

subplt.text(date[80], 34, "R\u00B2={0}, p<0.01, MAE={1}".format(round(r*r, 2), round(mae, 2)), fontsize=ftsize*1.3)                              

#subplt.plot(date,  lst_m[lst_s['index'].astype(int)]-lst_s["LST_Day_1km"], linewidth=4,  c = "blue", zorder=1, label='Model Simulations')  

#subplt.hlines(y=0, xmin=date[0], xmax=date[-1], colors='black', linestyles='--', lw=4)
                        
#subplt.legend(loc='upper right', fancybox = False, shadow = False,frameon = False, ncol = 3, fontsize=ftsize) 
handles = [R1, R2]
labels = ["Simulations", "MODIS LST"] 
for tick in subplt.get_xticklabels():
    tick.set_rotation(30)

fig.add_subplot(111, frame_on=False)
plt.tick_params(labelcolor="none", bottom=False, left=False)  
plt.xlabel("Date", fontsize=ftsize*1.4, family = ftfamily, labelpad=50) 
plt.ylabel("LST (degree)", fontsize=ftsize*1.4, family = ftfamily, labelpad=15)  
#fig.tight_layout() 
fig.legend(handles, labels, loc ='lower center', fancybox = False, shadow = False,frameon = False, 
          ncol = legendcols, handletextpad = 0.3, columnspacing = 1, prop={'family':ftfamily, 'size':ftsize})  
fig.subplots_adjust(left = None, right = None, bottom = 0.3)
    
plot_path = "../../../figs/lst/lst_ts1.jpg"
plt.show()
fig.savefig(plot_path, dpi=300, bbox_inches = 'tight')