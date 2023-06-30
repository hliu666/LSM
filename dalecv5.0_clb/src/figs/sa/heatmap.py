# -*- coding: utf-8 -*-
"""
Created on Sun Jul 17 15:39:12 2022

@author: 16072
"""
import seaborn as sb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('font',family='Calibri')

import joblib
s1 = joblib.load("sensitivity_fPAR.pkl")
s2 = joblib.load("sensitivity_nee.pkl")

df_corr1 = pd.DataFrame(s1['S2'])
df_corr2 = pd.DataFrame(s2['S2'])

fig, axs = plt.subplots(1, 2, figsize=(24, 10))
ftsize = 20 #字体大小
ftfamily = "Calibri"
labels = ["Cab", "Car", "Cbrown", "Cw", "Cm", "Ant", "Alpha"]  
for df_corr, ax, title, vmax in zip([df_corr1, df_corr2], [axs[0], axs[1]], ['fPAR', 'NEE'], [0.15, 0.04]):


    df_corr.columns = ["Cab", "Car", "Cbrown", "Cw", "Cm", "Ant", "Alpha"]
    df_corr.index = ["Cab", "Car", "Cbrown", "Cw", "Cm", "Ant", "Alpha"]
    corr = df_corr.iloc[:-1,1:].copy()
    
    # mask
    #mask = np.triu(np.ones_like(corr, dtype=np.bool))
    # adjust mask and df
    #mask = mask[1:, :-1]
    
    # color map
    cmap = sb.diverging_palette(0, 230, 90, 60, as_cmap=True)
    # plot heatmap
    hax = sb.heatmap(corr, annot=True, annot_kws=(dict(fontsize = 12, fontfamily = "Calibri")), fmt=".5f", ax=ax,
               linewidths=10, cmap=cmap, vmin=0, vmax=vmax, 
               cbar_kws={"shrink": .8, 'label': 'Second order index'}, square=True)
    hax.figure.axes[-1].yaxis.label.set_size(ftsize*1.5)
    hax.figure.axes[-1].tick_params(labelsize=ftsize)
    
    # ticks
    xticks = [i.upper() for i in corr.columns]
    yticks = [i.upper() for i in corr.index]

    x = np.arange(int(len(xticks)))
    y = np.arange(int(len(yticks)))   
    
    ax.set_xticks(x+0.5)
    ax.set_xticklabels(xticks, fontsize = ftsize, family = ftfamily)
    ax.set_yticks(x+0.5)
    ax.set_yticklabels(yticks, fontsize = ftsize, family = ftfamily)    
    
    ax.set_title(title, loc='center', fontsize=ftsize*2)

plot_path = "../../../figs/sa/heatmap.jpg"
plt.show()
fig.savefig(plot_path, dpi=600, quality=100,bbox_inches='tight')
