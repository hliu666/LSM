# -*- coding: utf-8 -*-
"""
Created on Sat Jan  7 11:50:19 2023

@author: hliu
"""

import joblib
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

out_x = joblib.load("../../../figs/plsr/out_x.pkl")
out_y1 = joblib.load("../../../figs/plsr/out_y1.pkl")
out_apar = joblib.load("../../../figs/plsr/out_apar.pkl")
out_par = joblib.load("../../../figs/plsr/out_par.pkl")
out_gpp = joblib.load("../../../figs/plsr/out_gpp.pkl")

arr_x = np.array(out_x)
arr_apar = np.array(out_apar)
arr_par = np.array(out_par)
arr_y1 = np.array(out_y1)

arr_gpp = np.array(out_gpp)
idx = np.where(arr_gpp < 0.5)[0]

arr_x = np.delete(arr_x, idx, axis=0)
arr_apar = np.delete(arr_apar, idx, axis=0)
arr_par = np.delete(arr_par, idx, axis=0)
arr_y1 = np.delete(arr_y1, idx, axis=0)

arr_fpar = arr_apar.flatten()/arr_par.flatten()
arr_y1 = arr_y1.flatten()

data = pd.read_csv("../../../figs/plsr/feature_importance.csv", na_values="nan")
fpar_importance1 = np.array(data)[:,0]

fpar_high_idx = fpar_importance1.argmax() 
fpar_wide_idx = np.argwhere(fpar_importance1 > 1)

fpar_high_idx = fpar_high_idx[fpar_high_idx < 1200]
fpar_wide_idx = fpar_wide_idx[fpar_wide_idx < 1200]

fpar_high = arr_x[:, fpar_high_idx].flatten() 
fpar_wide = (arr_x[:,fpar_wide_idx]).mean(axis=1).flatten() 

x, y = fpar_high, arr_fpar
fpar_high_r, fpar_high_p = stats.pearsonr(x, y) 

x, y = fpar_wide, arr_fpar
fpar_wide_r, fpar_wide_p = stats.pearsonr(x, y) 

lue_importance2 = np.array(data)[:,1]

lue_high_idx = lue_importance2.argmax() 
lue_wide_idx = np.argwhere(lue_importance2 > 1).flatten()

lue_high_idx = lue_high_idx[lue_high_idx < 1200]
lue_wide_idx = lue_wide_idx[lue_wide_idx < 1200]

lue_high = arr_x[:, lue_high_idx].flatten() 
lue_wide = (arr_x[:,lue_wide_idx]).mean(axis=1) 

x, y = lue_high, arr_y1
lue_high_r, lue_high_p = stats.pearsonr(x, y) 

x, y = lue_wide, arr_y1
lue_wide_r, lue_wide_p = stats.pearsonr(x, y) 

fig, axs = plt.subplots(2, 2, figsize=(22,18))

def_fontsize = 35
def_linewidth = 3.5

linewidth = 2.0 #边框线宽度
ftsize = 35 #字体大小
axlength = 5.0 #轴刻度长度
axwidth = 3.0 #轴刻度宽度

labels = ["fpar_single", "fpar_wide", "lue_single", "lue_wide"]
x_arrs = [fpar_high, fpar_wide, lue_high, lue_wide]
y_arrs = [arr_fpar, arr_fpar, arr_y1, arr_y1]
for i in range(0, 2):
    for j in range(0, 2):
        idx = i*2 + j

        x, y = x_arrs[idx], y_arrs[idx]
        k, b = np.polyfit(x, y, 1)
        #axs[i][j].plot(range(0,2), k*range(0,2) + b, 'black')
        axs[i][j].scatter(x, y, marker='o', color = 'None', s = 20, edgecolors="black")    

        axs[i][j].set_title("({0}) {1}".format(chr(97+i*3+j), labels[idx]), fontsize = ftsize*1.2)

        axs[i][j].annotate('k={0}, R\u00b2={1}\nRMSE={2}'.format(round(k,2), 
                                                                round(stats.pearsonr(x, y)[0]**2,2), 
                                                                round(np.sqrt(((x-y) ** 2).mean()),2)), 
                                                                (0.02, 0.72), xycoords='axes fraction', fontsize = ftsize, color='red')    
        axs[i][j].spines['left'].set_linewidth(linewidth)
        axs[i][j].spines['right'].set_linewidth(linewidth)
        axs[i][j].spines['top'].set_linewidth(linewidth)
        axs[i][j].spines['bottom'].set_linewidth(linewidth)
        axs[i][j].tick_params(direction = 'in', axis='both', length = axlength, width = axwidth, labelsize = ftsize)        
        
        fig.add_subplot(111, frame_on=False)
        plt.tick_params(labelcolor="none", bottom=False, left=False)
        
#plt.xlabel("NEON observations", fontsize = ftsize*1.5, family = ftfamily, labelpad=15)
#plt.ylabel("Model simulations", fontsize = ftsize*1.5, family = ftfamily, labelpad=15)

fig.tight_layout()          
     
plt.subplots_adjust(wspace =0.15, hspace =0.15)#调整子图间距    
plt.show()
plot_path = "../../../figs/plsr/scatter2.jpg"
fig.savefig(plot_path, dpi=600, bbox_inches = 'tight')
