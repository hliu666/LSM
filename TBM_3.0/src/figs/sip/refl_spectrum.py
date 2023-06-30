# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 08:36:08 2022

@author: hliu
"""
import os 
print(os.getcwd())
import sys 
sys.path.append("../../model")

import numpy as np
import prosail 

from RTM_initial import sip_leaf, soil_spectra
from RTM_initial import cal_lidf, weighted_sum_over_lidf_vec, CIxy
from RTM_initial import hemi_initial, dif_initial, hemi_dif_initial
from RTM_Optical import i_hemi, hotspot_calculations_vec, BRF_hemi_func, BRF_dif_func, define_geometric_constant
from RTM_Optical import BRF_hemi_dif_func
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('font',family='Calibri')

os.chdir("../")
def Opt_Refl_VZA(leaf, soil, tts, tto, psi, lai, lidfa, ci_flag):
    
    rho, tau = leaf
    rho, tau = rho.flatten()[0:2100], tau.flatten()[0:2100]

    rg = soil[0:2100]
    
    #lidfa = 1    # float Leaf Inclination Distribution at regular angle steps. 
    lidfb = np.inf # float Leaf Inclination Distribution at regular angle steps. 
    lidf  = cal_lidf(lidfa, lidfb)
    
    CI_flag = ci_flag
    CI_thres = 0.7
    CIs = CIxy(CI_flag, tts, CI_thres)
    CIo = CIxy(CI_flag, tto, CI_thres)

    _, _, ks, ko, _, sob, sof = weighted_sum_over_lidf_vec(lidf, tts, tto, psi)

    hemi_pars = hemi_initial(CI_flag, tts, lidf, CI_thres)
    dif_pars = dif_initial(CI_flag, tto, lidf, CI_thres)
    hemi_dif_pars = hemi_dif_initial(CI_flag, lidf, CI_thres)

    #soil and canopy properties
    w = rho + tau   #leaf single scattering albedo
    sur_refl_b01, sur_refl_b02, fPAR_list = [], [], []
   
    #计算lai    
    i0 = 1 - np.exp(-ks * lai * CIs)
    iv = 1 - np.exp(-ko * lai * CIo)
    
    t0 = 1 - i0
    tv = 1 - iv
    
    dso = define_geometric_constant(tts, tto, psi)

    [kc, kg]    =  hotspot_calculations_vec(np.array([lai]), ko, ks, CIo, CIs, dso)   

    [sob_vsla,          sof_vsla,          kgd]     = BRF_hemi_func(hemi_pars, lai, 0)       
    
    [sob_vsla_dif,      sof_vsla_dif,      kg_dif]  = BRF_dif_func(dif_pars,   lai, 0)  
    
    [sob_vsla_hemi_dif, sof_vsla_hemi_dif, kgd_dif] = BRF_hemi_dif_func(hemi_dif_pars, lai) 
   
    
    rho2 = iv/2/lai
    
    iD = i_hemi(CI_flag,lai,lidf, CI_thres)    
    
    p  = 1 - iD/lai  

    rho_hemi = iD/2/lai        
 
    wso  = sob*rho + sof*tau

    Tdn   = t0+i0*w*rho_hemi/(1-p*w)
    Tup_o = tv+iD*w*rho2/(1-p*w)
    Rdn   = iD*w*rho_hemi/(1-p*w)
    
    BRFv = wso*kc/ko + i0*w*w*p*rho2/(1-p*w)      
    BRFs = kg*rg
    BRFm = rg*Tdn*Tup_o/(1-rg*Rdn)-t0*rg*tv       
    BRF  = BRFv + BRFs + BRFm

    #absorption
    Av  = i0*(1-w)/(1-p*w)
    Aup = iD*(1-w)/(1-p*w)
    Am  = rg*(Tdn)*(Aup)/(1-rg*(Rdn))
    A   = Av + Am    #absorption
    
    return BRF

def cal_reflectance(tto, lai):
    Cab    = 28.12  #chlorophyll a+b content (mug cm-2).
    Car    = 5.56   #carotenoids content (mug cm-2).
    Cm     = 0.0065 #dry matter content (g cm-2).
    Cbrown = 0.185  #brown pigments concentration (unitless).
    Cw     = 0.00597  #equivalent water thickness (g cm-2 or cm).
    Ant    = 1.967  #Anthocianins concentration (mug cm-2). 
    Alpha  = 600   #constant for the the optimal size of the leaf scattering element   
    fLMA_k = 2519.65
    gLMA_k = -631.54 
    gLMA_b = 0.0086
    
    
    tts = np.array([30.0])
    psi = np.array([0.0]) 
    tto = np.array([tto])
    leaf = sip_leaf(Cab, Car, Cbrown, Cw, Cm, Ant, Alpha, fLMA_k, gLMA_k, gLMA_b)
    soil = soil_spectra()        

    lidfa = 63.6 
    hspot = 0.05
    ci_flag = 1 #Clumping Index is a constant 
    
    rho_canopy1 = Opt_Refl_VZA(leaf, soil, tts, tto, psi, lai, lidfa, ci_flag)

    sza, vza, raa = tts, tto, psi 
    rho_canopy2 = prosail.run_prosail(2.0, Cab, Car, Cbrown, Cw, Cm, lai, lidfa, hspot, sza, vza, raa, Ant, Alpha, rsoil0=soil[0:2101])
    
    rho, tau = leaf
    rho, tau = rho.flatten()[0:2100], tau.flatten()[0:2100]

    rg = soil[0:2100]
    
    return np.array(rho_canopy1), np.array(rho_canopy2[0:2100]), rho, tau, rg

linewidth = 1.8 #边框线宽度
ftsize = 20 #字体大小
axlength = 5.0 #轴刻度长度
axwidth = 3.0 #轴刻度宽度

def_fontsize = 18
def_linewidth = 3.5

fig, ax = plt.subplots(2, 2, figsize=(12,9))
"""
ax[0,0].plot(np.arange(400, 2500), rho, linewidth=def_linewidth,  label="Leaf Reflectance", color="black")
ax[0,0].plot(np.arange(400, 2500), tau, linewidth=def_linewidth,  label="Leaf Transmittance", color="red")
ax[0,0].plot(np.arange(400, 2500), soil, linewidth=def_linewidth, label="Soil Spectrum", color="blue")
ax[0,0].set_xlabel('Wavelength (nm)', fontsize=def_fontsize)
ax[0,0].set_ylabel('Leaf/Soil reflectance (unitless)', fontsize=def_fontsize)
ax[0,0].legend(loc='upper right', fancybox = False, shadow = False,frameon = False, ncol = 2, fontsize=def_fontsize/1.2) 
"""
tto, lai = 0.0, 0.5
rho_canopy1_00, rho_canopy2_00, rho, tau, soil = cal_reflectance(tto, lai)
tto, lai = 30.0, 0.5
rho_canopy1_30, rho_canopy2_30, _, _, _ = cal_reflectance(tto, lai)

ax[0,0].set_title("(a) LAI=0.5", fontsize=def_fontsize*1.2)
ax[0,0].plot(np.arange(400, 2500), rho_canopy1_00,  linewidth=def_linewidth, label="SIP for VZA=0",     color="black")
ax[0,0].plot(np.arange(400, 2500), rho_canopy2_00,  '--', linewidth=def_linewidth, label="PROSAIL for VZA=0", color="red")
ax[0,0].plot(np.arange(400, 2500), rho_canopy1_30,  linewidth=def_linewidth, label="SIP for VZA=30", color="blue")
ax[0,0].plot(np.arange(400, 2500), rho_canopy2_30,  '--', linewidth=def_linewidth, label="PROSAIL for VZA=30", color="darkorange")
ax[0,0].set_xlabel('Wavelength (nm)', fontsize=def_fontsize)
ax[0,0].set_ylabel('Canopy reflectance (unitless)', fontsize=def_fontsize)
ax[0,0].legend(loc='upper right', fancybox = False, shadow = False,frameon = False, ncol = 2, fontsize=def_fontsize/1.3)  
ax[0,0].set_ylim(0, 0.4)  

ax[0,0].spines['left'].set_linewidth(linewidth)
ax[0,0].spines['right'].set_linewidth(linewidth)
ax[0,0].spines['top'].set_linewidth(linewidth)
ax[0,0].spines['bottom'].set_linewidth(linewidth)
ax[0,0].tick_params(direction = 'in', axis='both', length = axlength, width = axwidth, labelsize = ftsize)        
  
tto, lai = 0.0, 1.0
rho_canopy1_00, rho_canopy2_00, rho, tau, soil = cal_reflectance(tto, lai)
tto, lai = 30.0, 1.0
rho_canopy1_30, rho_canopy2_30, _, _, _ = cal_reflectance(tto, lai)

ax[0,1].set_title("(b) LAI=1.0", fontsize=def_fontsize*1.2)
ax[0,1].plot(np.arange(400, 2500), rho_canopy1_00,  linewidth=def_linewidth, label="SIP for VZA=0",     color="black")
ax[0,1].plot(np.arange(400, 2500), rho_canopy2_00,  '--', linewidth=def_linewidth, label="PROSAIL for VZA=0", color="red")
ax[0,1].plot(np.arange(400, 2500), rho_canopy1_30,  linewidth=def_linewidth, label="SIP for VZA=30", color="blue")
ax[0,1].plot(np.arange(400, 2500), rho_canopy2_30,  '--', linewidth=def_linewidth, label="PROSAIL for VZA=30", color="darkorange")
ax[0,1].set_xlabel('Wavelength (nm)', fontsize=def_fontsize)
ax[0,1].set_ylabel('Canopy reflectance (unitless)', fontsize=def_fontsize)
ax[0,1].legend(loc='upper right', fancybox = False, shadow = False,frameon = False, ncol = 2, fontsize=def_fontsize/1.3)  
ax[0,1].set_ylim(0, 0.5)  

ax[0,1].spines['left'].set_linewidth(linewidth)
ax[0,1].spines['right'].set_linewidth(linewidth)
ax[0,1].spines['top'].set_linewidth(linewidth)
ax[0,1].spines['bottom'].set_linewidth(linewidth)
ax[0,1].tick_params(direction = 'in', axis='both', length = axlength, width = axwidth, labelsize = ftsize)        
  
tto, lai = 0.0, 3.0
rho_canopy1_00, rho_canopy2_00, _, _, _ = cal_reflectance(tto, lai)
tto, lai = 30.0, 3.0
rho_canopy1_30, rho_canopy2_30, _, _, _ = cal_reflectance(tto, lai)

ax[1,0].set_title("(c) LAI=3.0", fontsize=def_fontsize*1.2)
ax[1,0].plot(np.arange(400, 2500), rho_canopy1_00,  linewidth=def_linewidth, label="SIP for VZA=0",     color="black")
ax[1,0].plot(np.arange(400, 2500), rho_canopy2_00,  '--', linewidth=def_linewidth, label="PROSAIL for VZA=0", color="red")
ax[1,0].plot(np.arange(400, 2500), rho_canopy1_30,  linewidth=def_linewidth, label="SIP for VZA=30", color="blue")
ax[1,0].plot(np.arange(400, 2500), rho_canopy2_30,  '--', linewidth=def_linewidth, label="PROSAIL for VZA=30", color="darkorange")
ax[1,0].set_xlabel('Wavelength (nm)', fontsize=def_fontsize)
ax[1,0].set_ylabel('Canopy reflectance (unitless)', fontsize=def_fontsize)
ax[1,0].legend(loc='upper right', fancybox = False, shadow = False,frameon = False, ncol = 2, fontsize=def_fontsize/1.3) 
ax[1,0].set_ylim(0, 0.7)  

ax[1,0].spines['left'].set_linewidth(linewidth)
ax[1,0].spines['right'].set_linewidth(linewidth)
ax[1,0].spines['top'].set_linewidth(linewidth)
ax[1,0].spines['bottom'].set_linewidth(linewidth)
ax[1,0].tick_params(direction = 'in', axis='both', length = axlength, width = axwidth, labelsize = ftsize)        
  
tto, lai = 0.0, 5.0
rho_canopy1_00, rho_canopy2_00, _, _, _ = cal_reflectance(tto, lai)
tto, lai = 30.0, 5.0
rho_canopy1_30, rho_canopy2_30, _, _, _ = cal_reflectance(tto, lai)

ax[1,1].set_title("(d) LAI=5.0", fontsize=def_fontsize*1.2)
ax[1,1].plot(np.arange(400, 2500), rho_canopy1_00,  linewidth=def_linewidth, label="SIP for VZA=0",     color="black")
ax[1,1].plot(np.arange(400, 2500), rho_canopy2_00,  '--', linewidth=def_linewidth, label="PROSAIL for VZA=0", color="red")
ax[1,1].plot(np.arange(400, 2500), rho_canopy1_30,  linewidth=def_linewidth, label="SIP for VZA=30", color="blue")
ax[1,1].plot(np.arange(400, 2500), rho_canopy2_30,  '--', linewidth=def_linewidth, label="PROSAIL for VZA=30", color="darkorange")
ax[1,1].set_xlabel('Wavelength (nm)', fontsize=def_fontsize)
ax[1,1].set_ylabel('Canopy reflectance (unitless)', fontsize=def_fontsize)
ax[1,1].legend(loc='upper right', fancybox = False, shadow = False,frameon = False, ncol = 2, fontsize=def_fontsize/1.3) 
ax[1,1].set_ylim(0, 0.8)  

ax[1,1].spines['left'].set_linewidth(linewidth)
ax[1,1].spines['right'].set_linewidth(linewidth)
ax[1,1].spines['top'].set_linewidth(linewidth)
ax[1,1].spines['bottom'].set_linewidth(linewidth)
ax[1,1].tick_params(direction = 'in', axis='both', length = axlength, width = axwidth, labelsize = ftsize)        
          
fig.tight_layout()                      
plot_path = "../../figs/refl/refl_spectrum.jpg"
fig.savefig(plot_path, dpi=600, bbox_inches = 'tight')    
