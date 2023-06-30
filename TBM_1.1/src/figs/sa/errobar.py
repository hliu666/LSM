# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 11:02:42 2021

@author: hliu

"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
import matplotlib
matplotlib.rc('font',family='Calibri')

import warnings
warnings.filterwarnings('ignore')

#import os 
#os.chdir(r'C:\Users\liuha\Desktop\dalecv4.3_htc\src\sa')

import joblib

types = ['year']
#types = ['year','year','year']
for i in range(len(types)):
    s1 = joblib.load("./../../../src/sa/sensitivity_nee_{0}.pkl".format(types[i]))
    s2 = joblib.load("./../../../src/sa/sensitivity_lai_{0}.pkl".format(types[i]))
    s3 = joblib.load("./../../../src/sa/sensitivity_lst_{0}.pkl".format(types[i]))
    s4 = joblib.load("./../../../src/sa/sensitivity_red_{0}.pkl".format(types[i]))
    s5 = joblib.load("./../../../src/sa/sensitivity_nir_{0}.pkl".format(types[i]))

    if i == 0:
        mean1 = np.array([s1['S1'],      s1['ST']])
        err1  = np.array([s1['S1_conf'], s1['ST_conf']])
        mean2 = np.array([s2['S1'],      s2['ST']])
        err2  = np.array([s2['S1_conf'], s2['ST_conf']])
        mean3 = np.array([s3['S1'],      s3['ST']])
        err3  = np.array([s3['S1_conf'], s3['ST_conf']])
        mean4 = np.array([s4['S1'],      s4['ST']])
        err4  = np.array([s4['S1_conf'], s4['ST_conf']])
        mean5 = np.array([s5['S1'],      s5['ST']])
        err5  = np.array([s5['S1_conf'], s5['ST_conf']])        
    else:
        mean1 = np.hstack((mean1, np.array([s1['S1'],      s1['ST']])))
        err1  = np.hstack((err1,  np.array([s1['S1_conf'], s1['ST_conf']])))
        mean2 = np.hstack((mean2, np.array([s2['S1'],      s2['ST']])))
        err2  = np.hstack((err2,  np.array([s2['S1_conf'], s2['ST_conf']])))
        mean3 = np.hstack((mean3, np.array([s3['S1'],      s3['ST']])))
        err3  = np.hstack((err3,  np.array([s3['S1_conf'], s3['ST_conf']])))
        mean4 = np.hstack((mean4, np.array([s4['S1'],      s4['ST']])))
        err4  = np.hstack((err4,  np.array([s4['S1_conf'], s4['ST_conf']])))        
        mean5 = np.hstack((mean4, np.array([s5['S1'],      s5['ST']])))
        err5  = np.hstack((err4,  np.array([s5['S1_conf'], s5['ST_conf']])))     



fig, axs = plt.subplots(5, 1, figsize=(8, 8))

for mean, err, ylabel, ax in zip([mean1, mean2, mean3, mean4, mean5], [err1, err2, err3, err4, err5], ['NEE', 'LAI', 'LST', 'red', 'nir'], [axs[0], axs[1], axs[2], axs[3], axs[4]]):
    
    x_labels =  ["p0", "p1",  "p2",  "p3",  "p4",  "p5",  "p6",\
                 "p7",  "p8",  "p9",  "p10", "p11", "p12", "p13",\
                 "p14", "p15", "p16", "p17", "p18", "p19", "p20",\
                 "p21", "p22", "p23", "p24", "p25", "p26", "p27",\
                 "p28", "p29"]
     #"p0",   
    x = np.arange(int(len(x_labels)))
    
    bar_width = 0.3 #柱状体宽度
    capsize = 1.2 #柱状体标准差参数1
    capthick = 0.8  #柱状体标准差参数2
    elinewidth = 0.8 #柱状体标准差参数3
    linewidth = 1.0 #边框线宽度
    axlength = 2.0 #轴刻度长度
    axwidth = 1.2 #轴刻度宽度
    legendcols = 5 #图例一行的个数
    ftsize = 12 #字体大小
    ftfamily = "Calibri"
    
    #min_y1 = 0
    #max_y1 = 1
    
    S1_Color = 'red'
    ST_Color = 'black'
    
    trans1 = Affine2D().translate(-0.2, 0.0) + ax.transData
    trans2 = Affine2D().translate(+0.2, 0.0) + ax.transData
    R1 = ax.errorbar(x, mean[0,:], yerr=err[0,:], color=S1_Color, marker='o', markerfacecolor='white', markersize=ftsize/3, linestyle="none", transform=trans1)
    R2 = ax.errorbar(x, mean[1,:], yerr=err[1,:], color=ST_Color, marker='^', markerfacecolor='white', markersize=ftsize/2, linestyle="none", transform=trans2)
    
    ax.set_ylim(-0.2, np.amax(mean)+np.amax(err)+0.05)
    
    if ax == axs[0]:
        ax.set_title("Sensitivity Analysis", fontsize = ftsize*2, family = ftfamily)  
    #axes.set_ylabel('GPP mol CO₂m\u207B\u00B2yr\u207B\u00B9', fontsize = ftsize/1.5, family=ftfamily)
    ax.set_ylabel(ylabel, fontsize = ftsize*1.5, family=ftfamily)
    if ax == axs[3]:
        ax.set_xlabel('Parameters', fontsize = ftsize*1.5, family=ftfamily)
    
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize = ftsize*1.5, family = ftfamily, rotation=45)
    #ax.set_ylim(0, 0.15)
    
    ax.spines['left'].set_linewidth(linewidth)
    ax.spines['right'].set_linewidth(linewidth)
    ax.spines['top'].set_linewidth(linewidth)
    ax.spines['bottom'].set_linewidth(linewidth)
    ax.tick_params(axis='both', length = axlength, width = axwidth, labelsize = ftsize)
    
    handles = [R1, R2]
       
    labels = ["First order index",\
              "Total order index"] 
    
fig.tight_layout() 
fig.legend(handles, labels, loc ='lower center', fancybox = False, shadow = False,frameon = False, 
          ncol = legendcols, handletextpad = 0.3, columnspacing = 5, prop={'family':ftfamily, 'size':ftsize*1.5})  
fig.subplots_adjust(left = None, right = None, bottom = 0.13)

plot_path = "./../../../figs/sa/hist_all.jpg"
plt.show()
fig.savefig(plot_path, dpi=600, quality=100,bbox_inches='tight')
