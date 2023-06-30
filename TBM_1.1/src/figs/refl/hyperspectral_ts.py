# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 12:08:42 2023

@author: hliu
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('font',family='Calibri')

import joblib
brf = joblib.load("../../../data/output/model/brf_ci1_HARV.pkl")    
brf_year = brf[12::24,:][0:365,0:2000]
brf_year = brf_year.T

# generate 2 2d grids for the x & y bounds
y, x = np.meshgrid(np.linspace(0, 365, num=365), np.linspace(400, 2400, num=2000))

z = brf_year
z_min, z_max = 0, np.abs(z).max()

def_fontsize = 18
def_linewidth = 3.5

linewidth = 2.0 #边框线宽度
ftsize = 15 #字体大小
axlength = 5.0 #轴刻度长度
axwidth = 3.0 #轴刻度宽度
legendcols = 5 #图例一行的个数
ftfamily = 'Calibri'

fig, ax = plt.subplots(1, 1, figsize=(11,4))

c = ax.pcolormesh(x, y, z, cmap='Spectral_r', vmin=z_min, vmax=z_max)
ax.set_title('Seasonal Hyperspectral Reflectance', fontsize=def_fontsize*1.2)

ax.spines['left'].set_linewidth(linewidth)
ax.spines['right'].set_linewidth(linewidth)
ax.spines['top'].set_linewidth(linewidth)
ax.spines['bottom'].set_linewidth(linewidth)
ax.tick_params(direction = 'in', axis='both', length = axlength, width = axwidth, labelsize = ftsize)        
    
ax.set_xlabel("Wavelength (nm)", fontsize = ftsize*1.2, family = ftfamily, labelpad=5)
ax.set_ylabel("Day of year (2019)", fontsize = ftsize*1.2, family = ftfamily, labelpad=5)

# set the limits of the plot to the limits of the data
ax.axis([x.min(), x.max(), y.min(), y.max()])
fig.colorbar(c, ax=ax)

plt.show()
fig.tight_layout()          

plot_path = "../../../figs/refl/seasonal_refl.jpg"
fig.savefig(plot_path, dpi=600, bbox_inches = 'tight')    

