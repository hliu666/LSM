# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 13:40:51 2022

@author: 16072
"""
import sys 
sys.path.append("../model")

import numpy as np
import prosail 

from RTM_initial import sip_leaf, soil_spectra
from RTM_initial import cal_lidf, weighted_sum_over_lidf_vec, dir_gap_initial_vec, CIxy
from RTM_initial import single_hemi_initial, single_dif_initial, single_hemi_dif_initial

from Optical_RTM import sunshade, i_hemi, A_BRFv2_single_hemi, A_BRFv2_single_dif, A_BRFv2_single_hemi_dif
import matplotlib.pyplot as plt

def Opt_Refl_VZA(leaf, soil, tts_arr, tto_arr, psi_arr, lai, lidfa, ci_flag):
    
    rho, tau = leaf
    rho, tau = rho.flatten()[0:2000], tau.flatten()[0:2000]

    rg = soil[0:2000]
    
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
    
        fPAR = sum(A[0:301])/301
    
        sur_refl_b01.append(np.mean(BRF[220:271]))
        sur_refl_b02.append(np.mean(BRF[441:477]))
        fPAR_list.append(fPAR)
    
    return [sur_refl_b01, sur_refl_b02, fPAR_list]

Cab    = 38.55 #chlorophyll a+b content (mug cm-2).
Car    = 6.77  #carotenoids content (mug cm-2).
Cbrown = 0.348 #brown pigments concentration (unitless).
Cw     = 0.348 #equivalent water thickness (g cm-2 or cm).
Cm     = 0.005 #dry matter content (g cm-2).
Ant    = 0.001 #Anthocianins concentration (mug cm-2). 
Alpha  = 136   #constant for the the optimal size of the leaf scattering element   

leaf = sip_leaf(Cab, Car, Cbrown, Cw, Cm, Ant, Alpha)
soil = soil_spectra()  
      
lai_list = np.arange(0, 5, 0.1)
lidfa = 1
hspot = 0.05

tts = np.full(1, 30.0)
tto = np.full(1, 30.0)
psi = np.full(1, 0.0)

fpar1L, fpar2L, fpar3L = [], [], []
for lai in lai_list:
    r_red, r_nir, fpar1 = Opt_Refl_VZA(leaf, soil, tts, tto, psi, lai, lidfa, 0)
    r_red, r_nir, fpar2 = Opt_Refl_VZA(leaf, soil, tts, tto, psi, lai, lidfa, 1)
    r_red, r_nir, fpar3 = Opt_Refl_VZA(leaf, soil, tts, tto, psi, lai, lidfa, 2)
    
    fpar1L.append(fpar1[0])
    fpar2L.append(fpar2[0])
    fpar3L.append(fpar3[0])

fig, ax = plt.subplots(figsize=(4,4))
ax.set_xlabel("LAI")
ax.set_ylabel("fPAR")

#ax.plot(lai_list, fpar1L, linewidth=5,label="CI_with angle effect")
ax.plot(lai_list, fpar2L, label="CI_constant")
ax.plot(lai_list, fpar3L, label="CI_non")
ax.legend( fancybox = False, shadow = False,frameon = False, ncol = 2, fontsize=10) 
