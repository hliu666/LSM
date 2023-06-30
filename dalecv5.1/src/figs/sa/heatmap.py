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

types = ['Icar','Dalec','Leaf']
#types = ['year','year','year']
for i in range(len(types)):
    s1 = joblib.load("./../../../data/output/sa/sensitivity_nee_{0}.pkl".format(types[i]))
    s2 = joblib.load("./../../../data/output/sa/sensitivity_lai_{0}.pkl".format(types[i]))
    s3 = joblib.load("./../../../data/output/sa/sensitivity_red_{0}.pkl".format(types[i]))
    s4 = joblib.load("./../../../data/output/sa/sensitivity_nir_{0}.pkl".format(types[i]))

    if i == 0:
        df_corr1 = pd.DataFrame(s1['S2'])
        df_corr2 = pd.DataFrame(s2['S2'])
        df_corr3 = pd.DataFrame(s3['S2'])
        df_corr4 = pd.DataFrame(s4['S2'])
    else:
        df_corr1 = pd.concat([df_corr1, pd.DataFrame(s1['S2'])], axis=1)
        df_corr2 = pd.concat([df_corr2, pd.DataFrame(s2['S2'])], axis=1)
        df_corr3 = pd.concat([df_corr3, pd.DataFrame(s3['S2'])], axis=1)
        df_corr4 = pd.concat([df_corr4, pd.DataFrame(s4['S2'])], axis=1)      


fig, axs = plt.subplots(2, 2, figsize=(24, 10))
ftsize = 20 #字体大小
ftfamily = "Calibri"
labels = ["p1",  "p2",  "p3",  "p4",  "p5",  "p6",\
             "p7",  "p8",  "p9",  "p10", "p11", "p12", "p13",\
             "p14", "p15", "p16", "p17", "p18", "p19", "p20",\
             "p21", "p22", "p23", "p24", "p25", "p26", "p27",\
             "p28", "p29", "p30", "p31", "p32", "p33", "p34"]
for df_corr, ax, title, vmax in zip([df_corr1, df_corr2, df_corr3, df_corr4], [axs[0,0], axs[0,1], axs[1,0], axs[1,1]], ['NEE', 'LAI', 'Red', "NIR"], [1.0, 1.0, 1.0, 1.0]):


    df_corr.columns = labels
    df_corr.index = labels
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
