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
from RTM_Optical import i_hemi, hotspot_calculations, BRF_hemi_func, BRF_dif_func, define_geometric_constant
from RTM_Optical import BRF_hemi_dif_func
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('font',family='Calibri')

os.chdir("../")

rsr_red = np.genfromtxt("../../data/parameters/rsr_red.txt")
rsr_nir = np.genfromtxt("../../data/parameters/rsr_nir.txt")
rsr_sw1 = np.genfromtxt("../../data/parameters/rsr_swir1.txt")
rsr_sw2 = np.genfromtxt("../../data/parameters/rsr_swir2.txt")

def Opt_Refl_VZA(leaf, soil, tts_arr, tto_arr, psi_arr, lai, lidfa, ci_flag):
    
    rho, tau = leaf
    rho, tau = rho.flatten()[0:2100], tau.flatten()[0:2100]

    rg = soil[0:2100]
    
    #lidfa = 1    # float Leaf Inclination Distribution at regular angle steps. 
    lidfb = np.inf # float Leaf Inclination Distribution at regular angle steps. 
    lidf  = cal_lidf(lidfa, lidfb)
    
    CI_flag = ci_flag
    CI_thres = 0.7
    CIs_arr = CIxy(CI_flag, tts_arr, CI_thres)
    CIo_arr = CIxy(CI_flag, tto_arr, CI_thres)

    _, _, ks_arr, ko_arr, _, sob_arr, sof_arr = weighted_sum_over_lidf_vec(lidf, tts_arr, tto_arr, psi_arr)

    hemi_pars = hemi_initial(CI_flag, tts_arr, lidf, CI_thres)
    dif_pars = dif_initial(CI_flag, tto_arr, lidf, CI_thres)
    hemi_dif_pars = hemi_dif_initial(CI_flag, lidf, CI_thres)

    #soil and canopy properties
    w = rho + tau   #leaf single scattering albedo
    sur_refl_b01, sur_refl_b02, fPAR_list = [], [], []
    
    for x in range(len(CIs_arr)):
        CIs, CIo = CIs_arr[x], CIo_arr[x]  
        ks, ko   = ks_arr[x],  ko_arr[x]
        sob, sof = sob_arr[x], sof_arr[x]
        tts, tto, psi = tts_arr[x], tto_arr[x], psi_arr[x] 
        if ks < 0 or ko <0:
            print(tto)
      
        #计算lai    
        i0 = 1 - np.exp(-ks * lai * CIs)
        iv = 1 - np.exp(-ko * lai * CIo)
        
        t0 = 1 - i0
        tv = 1 - iv
                
        dso = define_geometric_constant(tts, tto, psi)
        
        [kc, kg]    =  hotspot_calculations(lai, ko, ks, CIo, CIs, dso)
       
        [sob_vsla,          sof_vsla,          kgd]     = BRF_hemi_func(hemi_pars, lai, x)       
        
        [sob_vsla_dif,      sof_vsla_dif,      kg_dif]  = BRF_dif_func(dif_pars,   lai, x)  
        
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
        
        fPAR = sum(A[0:301])/301
      
        sur_refl_b01.append(float(np.nansum(BRF[220:271].flatten()*rsr_red.flatten())/np.nansum(rsr_red.flatten())))
        sur_refl_b02.append(float(np.nansum(BRF[441:477].flatten()*rsr_nir.flatten())/np.nansum(rsr_nir.flatten())))
        #sur_refl_b01.append(np.nansum(BRF[220:271]))
        #sur_refl_b02.append(np.nansum(BRF[441:477]))

        fPAR_list.append(fPAR)
    
    return [sur_refl_b01, sur_refl_b02, fPAR_list]

def cal_reflectance(tto, tts, psi):
    Cab    = 28.12  #chlorophyll a+b content (mug cm-2).
    Car    = 5.56   #carotenoids content (mug cm-2).
    Cbrown = 0.185  #brown pigments concentration (unitless).
    Cw     = 0.005  #equivalent water thickness (g cm-2 or cm).
    Cm     = 0.0065 #dry matter content (g cm-2).
    Ant    = 1.967  #Anthocianins concentration (mug cm-2). 
    Alpha  = 600   #constant for the the optimal size of the leaf scattering element   
    fLMA_k = 3500
    gLMA_k =-631.54 
    gLMA_b = 0.01
    
    leaf = sip_leaf(Cab, Car, Cbrown, Cw, Cm, Ant, Alpha, fLMA_k, gLMA_k, gLMA_b)
    soil = soil_spectra()        
    lai = 1
    lidfa = 30
    hspot = 0.05
    ci_flag = 1 #Clumping Index is a constant 
    
    r_red, r_nir, fpar = Opt_Refl_VZA(leaf, soil, tts, tto, psi, lai, lidfa, ci_flag)
    r_red_ps, r_nir_ps = [], []
    for x in range(len(tto)):
        sza, vza, raa = tts[x], tto[x], psi[x]
        rho_canopy = prosail.run_prosail(0.9, Cab, Car, Cbrown, Cw, Cm, lai, lidfa, hspot, sza, vza, raa, Ant, Alpha, rsoil0=soil[0:2101])
        
        r_red_ps.append(float(np.nansum(rho_canopy[220:271].flatten()*rsr_red.flatten())/np.nansum(rsr_red.flatten())))
        r_nir_ps.append(float(np.nansum(rho_canopy[441:477].flatten()*rsr_nir.flatten())/np.nansum(rsr_nir.flatten())))
        #r_red_ps.append(np.nansum(rho_canopy[220:271]))
        #r_nir_ps.append(np.nansum(rho_canopy[441:477]))
                
    return np.array(r_red), np.array(r_nir), np.array(r_red_ps), np.array(r_nir_ps)

linewidth = 1.8 #边框线宽度
ftsize = 20 #字体大小
axlength = 5.0 #轴刻度长度
axwidth = 3.0 #轴刻度宽度

de_fontsize = 18
sangle, eangle, step = -75, 75, 1    
tto = np.arange(sangle, eangle, step)
tto = abs(tto)
tts = np.full(len(tto), 0.0)
psi = np.full(len(tto), 0.0)
psi[int(len(tto)/2):] = 180
r_red, r_nir, r_red_ps, r_nir_ps = cal_reflectance(tto, tts, psi)

fig, ax = plt.subplots(2, 2, figsize=(8,8))
ax[0,0].set_title("RED_00", fontsize=de_fontsize*1.2)
p1, = ax[0,0].plot(np.arange(sangle, eangle, step), r_red_ps, label="PROSAIL Red", color="black")
p2 = ax[0,0].fill_between(np.arange(sangle, eangle, step), r_red_ps-r_red_ps*0.01, r_red_ps+r_red_ps*0.01,   alpha=0.9, facecolor="black", label='1%')
p3 = ax[0,0].fill_between(np.arange(sangle, eangle, step), r_red_ps-r_red_ps*0.025, r_red_ps+r_red_ps*0.025, alpha=0.6, facecolor="black", label='2.5%')
p4 = ax[0,0].fill_between(np.arange(sangle, eangle, step), r_red_ps-r_red_ps*0.05, r_red_ps+r_red_ps*0.05,   alpha=0.3, facecolor="black", label='5%')
p5, = ax[0,0].plot(np.arange(sangle, eangle, step), r_red,    label="SIP Red", color="red")
ax[0,0].set_xlabel('View Zenith Angle [degree]', fontsize=de_fontsize)
ax[0,0].set_ylabel('brftop', fontsize=de_fontsize)
#ax[0,0].legend(loc='upper right', fancybox = False, shadow = False,frameon = False, ncol = 2, fontsize=10) 

ax[0,0].spines['left'].set_linewidth(linewidth)
ax[0,0].spines['right'].set_linewidth(linewidth)
ax[0,0].spines['top'].set_linewidth(linewidth)
ax[0,0].spines['bottom'].set_linewidth(linewidth)
ax[0,0].tick_params(direction = 'in', axis='both', length = axlength, width = axwidth, labelsize = ftsize)        
  
ax[0,1].set_title("NIR_00", fontsize=de_fontsize*1.2)
ax[0,1].plot(np.arange(sangle, eangle, step), r_nir_ps, label="PROSAIL NIR", color="black")
ax[0,1].fill_between(np.arange(sangle, eangle, step), r_nir_ps-r_nir_ps*0.01,  r_nir_ps+r_nir_ps*0.01,   alpha=0.9, facecolor="black", label='1%')
ax[0,1].fill_between(np.arange(sangle, eangle, step), r_nir_ps-r_nir_ps*0.025, r_nir_ps+r_nir_ps*0.025, alpha=0.6, facecolor="black", label='2.5%')
ax[0,1].fill_between(np.arange(sangle, eangle, step), r_nir_ps-r_nir_ps*0.05,  r_nir_ps+r_nir_ps*0.05,   alpha=0.3, facecolor="black", label='5%')
ax[0,1].plot(np.arange(sangle, eangle, step), r_nir,    label="SIP NIR", color="red")
ax[0,1].set_xlabel('View Zenith Angle [degree]', fontsize=de_fontsize)
ax[0,1].set_ylabel('brftop', fontsize=de_fontsize)

ax[0,1].spines['left'].set_linewidth(linewidth)
ax[0,1].spines['right'].set_linewidth(linewidth)
ax[0,1].spines['top'].set_linewidth(linewidth)
ax[0,1].spines['bottom'].set_linewidth(linewidth)
ax[0,1].tick_params(direction = 'in', axis='both', length = axlength, width = axwidth, labelsize = ftsize)        
  
tts = np.full(len(tto), 30.0)
r_red, r_nir, r_red_ps, r_nir_ps = cal_reflectance(tto, tts, psi)

ax[1,0].set_title("NIR_30", fontsize=de_fontsize*1.2)
ax[1,0].plot(np.arange(sangle, eangle, step), r_nir_ps, label="PROSAIL NIR", color="black")
ax[1,0].fill_between(np.arange(sangle, eangle, step), r_nir_ps-r_nir_ps*0.01,  r_nir_ps+r_nir_ps*0.01,   alpha=0.9, facecolor="black", label='1%')
ax[1,0].fill_between(np.arange(sangle, eangle, step), r_nir_ps-r_nir_ps*0.025, r_nir_ps+r_nir_ps*0.025, alpha=0.6, facecolor="black", label='2.5%')
ax[1,0].fill_between(np.arange(sangle, eangle, step), r_nir_ps-r_nir_ps*0.05,  r_nir_ps+r_nir_ps*0.05,   alpha=0.3, facecolor="black", label='5%')
ax[1,0].plot(np.arange(sangle, eangle, step), r_nir,    label="SIP NIR", color="red")
ax[1,0].set_xlabel('View Zenith Angle [degree]', fontsize=de_fontsize)
ax[1,0].set_ylabel('brftop', fontsize=de_fontsize)

ax[1,0].spines['left'].set_linewidth(linewidth)
ax[1,0].spines['right'].set_linewidth(linewidth)
ax[1,0].spines['top'].set_linewidth(linewidth)
ax[1,0].spines['bottom'].set_linewidth(linewidth)
ax[1,0].tick_params(direction = 'in', axis='both', length = axlength, width = axwidth, labelsize = ftsize)        
  
tts = np.full(len(tto), 60.0)
r_red, r_nir, r_red_ps, r_nir_ps = cal_reflectance(tto, tts, psi)

ax[1,1].set_title("NIR_60", fontsize=de_fontsize*1.2)
ax[1,1].plot(np.arange(sangle, eangle, step), r_nir_ps, label="PROSAIL NIR", color="black")
ax[1,1].fill_between(np.arange(sangle, eangle, step), r_nir_ps-r_nir_ps*0.01,  r_nir_ps+r_nir_ps*0.01,   alpha=0.9, facecolor="black", label='1%')
ax[1,1].fill_between(np.arange(sangle, eangle, step), r_nir_ps-r_nir_ps*0.025, r_nir_ps+r_nir_ps*0.025, alpha=0.6, facecolor="black", label='2.5%')
ax[1,1].fill_between(np.arange(sangle, eangle, step), r_nir_ps-r_nir_ps*0.05,  r_nir_ps+r_nir_ps*0.05,   alpha=0.3, facecolor="black", label='5%')
ax[1,1].plot(np.arange(sangle, eangle, step), r_nir,    label="SIP NIR", color="red")
ax[1,1].set_xlabel('View Zenith Angle [degree]', fontsize=de_fontsize)
ax[1,1].set_ylabel('brftop', fontsize=de_fontsize)

ax[1,1].spines['left'].set_linewidth(linewidth)
ax[1,1].spines['right'].set_linewidth(linewidth)
ax[1,1].spines['top'].set_linewidth(linewidth)
ax[1,1].spines['bottom'].set_linewidth(linewidth)
ax[1,1].tick_params(direction = 'in', axis='both', length = axlength, width = axwidth, labelsize = ftsize)        
    
handles = [p5, p1, p2, p3, p4]
labels = ['SIP', 'PROSAIL', "1% bias", "2.5% bias", "5% bias"]  

fig.tight_layout() 
fig.subplots_adjust(bottom=0.16)
    
fig.legend(handles, labels, loc ='lower center', bbox_to_anchor=(0.5, 0.02),
          fancybox = False, shadow = False,frameon = False, ncol = 5, 
          handletextpad = 0.2, columnspacing = 1.2, prop={'family':"Calibri", 'size':ftsize/1.2})  
                               
plot_path = "../../figs/refl/refl_vza.jpg"
fig.savefig(plot_path, dpi=600, bbox_inches = 'tight')    
