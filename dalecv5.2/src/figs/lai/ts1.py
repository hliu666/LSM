# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 17:04:46 2022

@author: 16072
"""
import numpy as np
import os 
import pandas as pd
import scipy.stats as stats
import joblib
import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('font',family='Calibri')

import warnings
warnings.filterwarnings('ignore')

sites = ['HARV, DBF']
sitens = ['1_HARV']
fig, axs = plt.subplots(1, 1, figsize=(10, 6.5))    

site = 'HARV, DBF'
lai_m = joblib.load("../../../data/output/model/out_ci1_HARV.pkl")[:,-1]
lai_s = joblib.load("../../../data/output/verify/d_lai1_HARV.pkl")

linewidth = 2.0 #边框线宽度
ftsize = 18 #字体大小
axlength = 2.0 #轴刻度长度
axwidth = 2.0 #轴刻度宽度
ftfamily = 'Calibri'
legendcols = 5 #图例一行的个数
        
subplt = axs
subplt.set_ylim(-0.3,8) 

subplt.spines['left'].set_linewidth(linewidth)
subplt.spines['right'].set_linewidth(linewidth)
subplt.spines['top'].set_linewidth(linewidth)
subplt.spines['bottom'].set_linewidth(linewidth)
subplt.tick_params(direction = 'in', axis='both', length = axlength, width = axwidth, labelsize = ftsize*1.1)
 
t_index = pd.date_range(start='2019-01-01', end='2021-12-31', freq="1d")
#subplt.set_title('({0}) {1}'.format(chr(97), site), fontsize=ftsize*1.6, family=ftfamily)                             
R1, = subplt.plot(t_index,  lai_m, linewidth=4,  c = "black", zorder=1, label='Simulations')  
date = [datetime.datetime(int(x[0]), int(x[1]), int(x[2]), 0, 0)  for x in np.array(lai_s)]
#subplt.scatter(date,  lai_s['LAI'],  marker="x", c = "r", s = 60,  zorder=3, label='NEON Observations')            
R2 = subplt.errorbar(date,  lai_s['LAI'], yerr=lai_s['errLAI'], c = "r",  fmt='o', fillstyle='none', capsize=6, label='Observations')    

"""
statistical indices
"""
lai_s['date'] = date
lai_m_pd = pd.DataFrame(lai_m, columns=['mLAI'])
lai_m_pd['date'] = t_index
lai_df = pd.merge(lai_s, lai_m_pd, on='date')

x = np.array(lai_df['LAI'])
y = np.array(lai_df['mLAI'])

r, p = stats.pearsonr(x, y) 

abs_sum = sum(abs(x - y))
abs_num = len(x)
mae = (abs_sum / abs_num)

subplt.text(t_index[580],  7, "R\u00B2={0}, p<0.01, MAE={1}".format(round(r*r, 2), round(mae, 2)), fontsize=ftsize*1.3)                              

handles = [R1, R2]
labels = ["Simulations", "Observations"] 
#subplt.legend(loc='upper right', fancybox = False, shadow = False,frameon = False, ncol = 2, fontsize=ftsize) 

for tick in subplt.get_xticklabels():
    tick.set_rotation(30)

fig.add_subplot(111, frame_on=False)
plt.tick_params(labelcolor="none", bottom=False, left=False)   
plt.xlabel("Date", fontsize=ftsize*1.4, family = ftfamily, labelpad=50)   
plt.ylabel("LAI",  fontsize=ftsize*1.4, family = ftfamily, labelpad=15)   
#fig.tight_layout() 
fig.legend(handles, labels, loc ='lower center', fancybox = False, shadow = False,frameon = False, 
          ncol = legendcols, handletextpad = 0.4, columnspacing = 0.5, prop={'family':ftfamily, 'size':ftsize*1.2})  
fig.subplots_adjust(left = None, right = None, bottom = 0.3)
 
    
plot_path = "../../../figs/lai/lai_ts1.jpg"
plt.show()
fig.savefig(plot_path, dpi=600, bbox_inches = 'tight')