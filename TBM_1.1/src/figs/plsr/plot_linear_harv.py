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

out_x = joblib.load("../../../data/output/model/brf_ci1_HARV.pkl")

arr_x = np.array(out_x)

out_gpp = joblib.load("../../../data/output/model/An_ci1_HARV.pkl")
out_apar = joblib.load("../../../data/output/model/apar_ci1_HARV.pkl")
out_par = joblib.load("../../../data/output/model/par_ci1_HARV.pkl")

arr_gpp = np.array(out_gpp)
arr_apar = np.array(out_apar)
arr_par = np.array(out_par)

idx = np.where(arr_gpp < 0.5)[0]

arr_x = np.delete(arr_x, idx, axis=0)

arr_gpp = np.delete(arr_gpp, idx, axis=0)
arr_apar = np.delete(arr_apar, idx, axis=0)
arr_par = np.delete(arr_par, idx, axis=0)

arr_y1 = (arr_gpp/arr_apar).flatten()
arr_y2 = (arr_gpp/arr_par).flatten()

data = pd.read_csv("../../../figs/plsr/feature_importance.csv", na_values="nan")

"""
gpp/PAR
"""
feature_importance1 = np.array(data)[:,0]

red_importance = feature_importance1[220:271]
nir_importance = feature_importance1[441:477]

red_idx = red_importance.argmax() + 220
nir_idx = nir_importance.argmax() + 441

red_high = arr_x[:, red_idx]
nir_high = arr_x[:, nir_idx]

nirv_high = nir_high*(nir_high-red_high)/(nir_high+red_high)

red_wide = (arr_x[:, 220:271]).mean(axis=1) 
nir_wide = (arr_x[:, 441:477]).mean(axis=1)

nirv_wide = nir_wide*(nir_wide-red_wide)/(nir_wide+red_wide)

x, y = nirv_high, arr_y1
nirv_high_r, nirv_high_p = stats.pearsonr(x, y) 

x, y = nirv_wide, arr_y1
nirv_wide_r, nirv_wide_p = stats.pearsonr(x, y) 

"""
gpp/APAR
"""
feature_importance2 = np.array(data)[:,1]

b1_importance = feature_importance2[50:79]
b2_importance = feature_importance2[220:271]
b3_importance = feature_importance2[441:477]

b1_idx = b1_importance.argmax() + 50
b2_idx = b2_importance.argmax() + 220
b3_idx = b3_importance.argmax() + 441

b1_high = arr_x[:, b1_idx]
b2_high = arr_x[:, b2_idx]
b3_high = arr_x[:, b3_idx]

evi_high = 2.5*(b3_high-b2_high)/(b3_high + 6.0*b2_high - 7.5*b1_high + 1.0)

b1_wide = (arr_x[:, 50:79]).mean(axis=1) 
b2_wide = (arr_x[:, 220:271]).mean(axis=1)
b3_wide = (arr_x[:, 441:477]).mean(axis=1)

evi_wide = 2.5*(b3_wide-b2_wide)/(b3_wide + 6.0*b2_wide - 7.5*b1_wide + 1.0)

x, y = evi_high, arr_y2
evi_high_r, evi_high_p = stats.pearsonr(x, y) 

x, y = evi_wide, arr_y2
evi_wide_r, evi_wide_p = stats.pearsonr(x, y) 

"""
GPP
"""
feature_importance3 = np.array(data)[:,2]

red_importance = feature_importance1[220:271]
nir_importance = feature_importance1[441:477]

red_idx = red_importance.argmax() + 220
nir_idx = nir_importance.argmax() + 441

red_high = arr_x[:, red_idx]
nir_high = arr_x[:, nir_idx]

nirv_high = nir_high*(nir_high-red_high)/(nir_high+red_high)

red_wide = (arr_x[:, 220:271]).mean(axis=1) 
nir_wide = (arr_x[:, 441:477]).mean(axis=1)

nirv_wide = nir_wide*(nir_wide-red_wide)/(nir_wide+red_wide)

x, y = nirv_high, arr_y1
gpp_high_r, gpp_high_p = stats.pearsonr(x, y) 

x, y = nirv_wide, arr_y1
gpp_wide_r, gpp_wide_p = stats.pearsonr(x, y) 


fig, axs = plt.subplots(3, 2, figsize=(22,24))

def_fontsize = 35
def_linewidth = 3.5

linewidth = 2.0 #边框线宽度
ftsize = 35 #字体大小
axlength = 5.0 #轴刻度长度
axwidth = 3.0 #轴刻度宽度

labels = ["gpp/par_single", "gpp/par_wide", "gpp/apar_single", "gpp/apar_wide", "gpp_single", "gpp_wide"]
xlabels = ["NIRv", "NIRv", "EVI", "EVI", "NIRv", "NIRv"]
ylabels = ["gpp/par_single", "gpp/par_wide", "gpp/apar_single", "gpp/apar_wide", "gpp_single", "gpp_wide"]
x_arrs = [nirv_high, nirv_wide, evi_high, evi_wide, nirv_high, nirv_wide]
y_arrs = [arr_y1, arr_y1, arr_y2, arr_y2, arr_gpp, arr_gpp]
for i in range(0, 3):
    for j in range(0, 2):
        idx = i*2 + j

        x, y = x_arrs[idx], y_arrs[idx]
        k, b = np.polyfit(x, y, 1)
        #axs[i][j].plot(range(0,2), k*range(0,2) + b, 'black')
        axs[i][j].scatter(x, y, marker='o', color = 'None', s = 20, edgecolors="black")    

        #axs[i][j].set_title("({0}) {1}".format(chr(97+i*3+j), labels[idx]), fontsize = ftsize*1.2)
        axs[i][j].set_xlabel(xlabels[idx], fontsize = ftsize*1.2)
        axs[i][j].set_ylabel(ylabels[idx], fontsize = ftsize*1.2)

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
     
plt.subplots_adjust(wspace =0.3, hspace =0.3)#调整子图间距    
plt.show()
plot_path = "../../../figs/plsr/scatter_harv.jpg"
fig.savefig(plot_path, dpi=600, bbox_inches = 'tight')
