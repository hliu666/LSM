# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 18:15:02 2022

@author: hliu
"""

import os 
print(os.getcwd())
import sys 
sys.path.append("../../model")
from RTM_initial import sip_leaf

os.chdir("../")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('font',family='Calibri')

""" 
Parameters of Leaf-level Radiative Transfer Model  
"""           
lma = 65.18  # clma, leaf mass per area          (81 - 120) g C m-2

Cab    = 28.11851723
Car    = 5.563160774
Cm     = lma/10000.0  
Cbrown = 0.185385 #brown pigments concentration (unitless).
Cw     = 0.00597  #equivalent water thickness (g cm-2 or cm).
Ant    = 1.96672  #Anthocianins concentration (mug cm-2). 
Alpha  = 600     #constant for the the optimal size of the leaf scattering element   
fLMA_k = 2519.65
gLMA_k = -631.54
gLMA_b = 0.0086 

[refl_sim, tran_sim] = sip_leaf(Cab, Car, Cbrown, Cw, Cm, Ant, Alpha, fLMA_k, gLMA_k, gLMA_b)
refl_sim, tran_sim = refl_sim[0:2100], tran_sim[0:2100]
data = pd.read_csv("../../data/parameters/HARV_spectral.csv", na_values="nan")
refl_obs, tran_obs = data['refl'], data['tran']
refl_obs, tran_obs = refl_obs[0:2100], tran_obs[0:2100]

fig, ax = plt.subplots(2, figsize=(6,8))

def_fontsize = 12
def_linewidth = 2.5


ax[0].plot(np.arange(400, 2500), refl_sim, linewidth=def_linewidth,  label="SIP-based leaf Reflectance", color="black")
ax[0].plot(np.arange(400, 2500), refl_obs, '--', linewidth=def_linewidth,  label="Observed leaf Reflectance", color="red")
ax[0].set_ylim(0.0, 0.5)
ax[0].set_xlabel('Wavelength (nm)', fontsize=def_fontsize)
ax[0].set_ylabel('Leaf reflectance (unitless)', fontsize=def_fontsize)
ax[0].legend(loc='upper right', fancybox = False, shadow = False,frameon = False, ncol = 2, fontsize=def_fontsize/1.2) 

ax[1].plot(np.arange(400, 2500), tran_sim,  linewidth=def_linewidth, label="SIP-based leaf Transmittance",     color="black")
ax[1].plot(np.arange(400, 2500), tran_obs,  '--', linewidth=def_linewidth, label="Observed leaf Transmittance", color="red")
ax[1].set_ylim(0.0, 0.6)
ax[1].set_xlabel('Wavelength (nm)', fontsize=def_fontsize)
ax[1].set_ylabel('Leaf reflectance (unitless)', fontsize=def_fontsize)
ax[1].legend(loc='upper right', fancybox = False, shadow = False,frameon = False, ncol = 2, fontsize=def_fontsize/1.2)  
print(os.getcwd())
plot_path = "../../figs/refl/leaf_sip.jpg"
fig.savefig(plot_path, dpi=600, bbox_inches = 'tight')    
