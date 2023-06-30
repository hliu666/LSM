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
from RTM_initial import cal_lidf, weighted_sum_over_lidf_vec, dir_gap_initial_vec, CIxy
from RTM_initial import single_hemi_initial, single_dif_initial, single_hemi_dif_initial

from Optical_RTM import sunshade, i_hemi, A_BRFv2_single_hemi, A_BRFv2_single_dif, A_BRFv2_single_hemi_dif
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('font',family='Calibri')

def Opt_Refl_VZA(leaf, soil, tts_arr, tto_arr, psi_arr, lai, lidfa, ci_flag):
    
    rho, tau = leaf
    rho, tau = rho.flatten()[0:2101], tau.flatten()[0:2101]

    rg = soil[0:2101]
    
    #lidfa = 1    # float Leaf Inclination Distribution at regular angle steps. 
    lidfb = -0.15 # float Leaf Inclination Distribution at regular angle steps. 
    lidf  = cal_lidf(lidfa, lidfb)
    
    CI_flag = ci_flag
    CIs_arr = CIxy(CI_flag, tts_arr)
    CIo_arr = CIxy(CI_flag, tto_arr)

    _, _, ks_arr, ko_arr, _, sob_arr, sof_arr = weighted_sum_over_lidf_vec(lidf, tts_arr, tto_arr, psi_arr)
    Ps_arrs, Po_arrs, int_res_arrs, nl = dir_gap_initial_vec(tts_arr, tto_arr, psi_arr, ks_arr, ko_arr, CIs_arr, CIo_arr)

    hemi_pars = single_hemi_initial(CI_flag, tts_arr, lidf)
    dif_pars = single_dif_initial(CI_flag, tto_arr, lidf)
    hemi_dif_pars = single_hemi_dif_initial(CI_flag, lidf)
      
    #soil and canopy properties
    w = rho + tau   #leaf single scattering albedo
    sur_refl_b01, sur_refl_b02, fPAR_list = [], [], []
    for x in range(len(tto_arr)):
        CIs, CIo, ks, ko, sob, sof = CIs_arr[x], CIo_arr[x], ks_arr[x], ko_arr[x], sob_arr[x], sof_arr[x] 
        tts, tto, psi = tts_arr[x], tto_arr[x], psi_arr[x] 
        
        Ps_arr, Po_arr, int_res_arr = Ps_arrs[x], Po_arrs[x], int_res_arrs[:,x,:]

        #计算lai    
        i0 = 1 - np.exp(-ks * lai * CIs)
        iv = 1 - np.exp(-ko * lai * CIo)
        
        t0 = 1 - i0
        tv = 1 - iv
        
        [kc, kg]    =   sunshade(tts, tto, psi, ks, ko, CIs, CIo, Ps_arr, Po_arr, int_res_arr, nl, lai)       
       
        [sob_vsla,          sof_vsla,          kgd]     = A_BRFv2_single_hemi(hemi_pars, lai, x)       
        
        [sob_vsla_dif,      sof_vsla_dif,      kg_dif]  = A_BRFv2_single_dif(dif_pars,   lai, x)  
        
        [sob_vsla_hemi_dif, sof_vsla_hemi_dif, kgd_dif] = A_BRFv2_single_hemi_dif(hemi_dif_pars, lai)      
    
        
        rho2 = iv/2/lai
        
        iD = i_hemi(CI_flag,lai,lidf)  
        
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
    Cab    = 38.55 #chlorophyll a+b content (mug cm-2).
    Car    = 6.77  #carotenoids content (mug cm-2).
    Cbrown = 0.348 #brown pigments concentration (unitless).
    Cw     = 0.348 #equivalent water thickness (g cm-2 or cm).
    Cm     = 0.036 #dry matter content (g cm-2).
    Ant    = 0.001 #Anthocianins concentration (mug cm-2). 
    Alpha  = 115   #constant for the the optimal size of the leaf scattering element   
    
    tts = np.array([30.0])
    psi = np.array([0.0]) 
    tto = np.array([tto])
    leaf = sip_leaf(Cab, Car, Cbrown, Cw, Cm, Ant, Alpha)
    soil = soil_spectra()        

    lidfa = 1
    hspot = 0.05
    ci_flag = 1 #Clumping Index is a constant 

    rho_canopy1 = Opt_Refl_VZA(leaf, soil, tts, tto, psi, lai, lidfa, ci_flag)

    sza, vza, raa = tts, tto, psi 
    rho_canopy2 = prosail.run_prosail(2.25, Cab, Car, Cbrown, Cw, Cm, lai, lidfa, hspot, sza, vza, raa, Ant, Alpha, rsoil0=soil[0:2101])
    
    rho, tau = leaf
    rho, tau = rho.flatten()[0:2101], tau.flatten()[0:2101]

    rg = soil[0:2101]
    
    return np.array(rho_canopy1), np.array(rho_canopy2), rho, tau, rg

def_fontsize = 15
def_linewidth = 2.5

tto, lai = 0.0, 1.0
rho_canopy1_00, rho_canopy2_00, rho, tau, soil = cal_reflectance(tto, lai)
tto, lai = 30.0, 1.0
rho_canopy1_30, rho_canopy2_30, _, _, _ = cal_reflectance(tto, lai)

fig, ax = plt.subplots(2, 2, figsize=(16,8))
ax[0,0].plot(np.arange(400, 2501), rho, linewidth=def_linewidth,  label="Leaf Reflectance", color="black")
ax[0,0].plot(np.arange(400, 2501), tau, linewidth=def_linewidth,  label="Leaf Transmittance", color="red")
ax[0,0].plot(np.arange(400, 2501), soil, linewidth=def_linewidth, label="Soil Spectrum", color="blue")
ax[0,0].set_xlabel('Wavelength (nm)', fontsize=def_fontsize)
ax[0,0].set_ylabel('Leaf/Soil reflectance (unitless)', fontsize=def_fontsize)
ax[0,0].legend(loc='upper right', fancybox = False, shadow = False,frameon = False, ncol = 2, fontsize=10) 

ax[0,1].plot(np.arange(400, 2501), rho_canopy1_00,  linewidth=def_linewidth, label="SIP for VZA=0",     color="black")
ax[0,1].plot(np.arange(400, 2501), rho_canopy2_00,  '--', linewidth=def_linewidth, label="PROSAIL for VZA=0", color="red")
ax[0,1].plot(np.arange(400, 2501), rho_canopy1_30,  linewidth=def_linewidth, label="SIP for VZA=30", color="blue")
ax[0,1].plot(np.arange(400, 2501), rho_canopy2_30,  '--', linewidth=def_linewidth, label="PROSAIL for VZA=30", color="darkorange")
ax[0,1].set_xlabel('Wavelength (nm)', fontsize=def_fontsize)
ax[0,1].set_ylabel('Canopy reflectance (unitless)', fontsize=def_fontsize)
ax[0,1].legend(loc='upper right', fancybox = False, shadow = False,frameon = False, ncol = 2, fontsize=10) 

tto, lai = 0.0, 3.0
rho_canopy1_00, rho_canopy2_00, _, _, _ = cal_reflectance(tto, lai)
tto, lai = 30.0, 3.0
rho_canopy1_30, rho_canopy2_30, _, _, _ = cal_reflectance(tto, lai)

ax[1,0].plot(np.arange(400, 2501), rho_canopy1_00,  linewidth=def_linewidth, label="SIP for VZA=0",     color="black")
ax[1,0].plot(np.arange(400, 2501), rho_canopy2_00,  '--', linewidth=def_linewidth, label="PROSAIL for VZA=0", color="red")
ax[1,0].plot(np.arange(400, 2501), rho_canopy1_30,  linewidth=def_linewidth, label="SIP for VZA=30", color="blue")
ax[1,0].plot(np.arange(400, 2501), rho_canopy2_30,  '--', linewidth=def_linewidth, label="PROSAIL for VZA=30", color="darkorange")
ax[1,0].set_xlabel('Wavelength (nm)', fontsize=def_fontsize)
ax[1,0].set_ylabel('Canopy reflectance (unitless)', fontsize=def_fontsize)
ax[1,0].legend(loc='upper right', fancybox = False, shadow = False,frameon = False, ncol = 2, fontsize=10) 

tto, lai = 0.0, 5.0
rho_canopy1_00, rho_canopy2_00, _, _, _ = cal_reflectance(tto, lai)
tto, lai = 30.0, 5.0
rho_canopy1_30, rho_canopy2_30, _, _, _ = cal_reflectance(tto, lai)

ax[1,1].plot(np.arange(400, 2501), rho_canopy1_00,  linewidth=def_linewidth, label="SIP for VZA=0",     color="black")
ax[1,1].plot(np.arange(400, 2501), rho_canopy2_00,  '--', linewidth=def_linewidth, label="PROSAIL for VZA=0", color="red")
ax[1,1].plot(np.arange(400, 2501), rho_canopy1_30,  linewidth=def_linewidth, label="SIP for VZA=30", color="blue")
ax[1,1].plot(np.arange(400, 2501), rho_canopy2_30,  '--', linewidth=def_linewidth, label="PROSAIL for VZA=30", color="darkorange")
ax[1,1].set_xlabel('Wavelength (nm)', fontsize=def_fontsize)
ax[1,1].set_ylabel('Canopy reflectance (unitless)', fontsize=def_fontsize)
ax[1,1].legend(loc='upper right', fancybox = False, shadow = False,frameon = False, ncol = 2, fontsize=10) 
                               
plot_path = "../../figs/refl/refl_spectrum.jpg"
fig.savefig(plot_path, dpi=600, bbox_inches = 'tight')    
