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

    #soil and canopy properties
    w = rho + tau   #leaf single scattering albedo
    sur_refl_b01, sur_refl_b02, fPAR_list = [], [], []
    
    #计算lai    
   
    dso = define_geometric_constant(tts_arr, tto_arr, psi_arr)
    
    [kc_arr, kg_arr]    =  hotspot_calculations_vec(lai, ko_arr, ks_arr, CIo_arr, CIs_arr, dso)
      
    for x in range(len(tto_arr)):
        kc, kg   = kc_arr[x], kg_arr[x]
        ks, ko   = ks_arr[x], ko_arr[x]
        CIs, CIo = CIs_arr[x], CIo_arr[x]
        sob, sof = sob_arr[x], sof_arr[x]
        
        i0 = 1 - np.exp(-ks * lai[x] * CIs)
        iv = 1 - np.exp(-ko * lai[x] * CIo)
        
        t0 = 1 - i0
        tv = 1 - iv
             
        rho2 = iv/2/lai[x]
        
        iD = i_hemi(CI_flag,lai[x],lidf, CI_thres)    
       
        p  = 1 - iD/lai[x]  
    
        rho_hemi = iD/2/lai[x]        
     
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

def cal_reflectance(lai, tto, tts, psi):
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
    
    leaf = sip_leaf(Cab, Car, Cbrown, Cw, Cm, Ant, Alpha, fLMA_k, gLMA_k, gLMA_b)
    soil = soil_spectra()        
    ci_flag = 1 #Clumping Index is a constant 
    hspot = 0.05
    lidfa = 63.6   
    
    r_red_sip, r_nir_sip, fpar = Opt_Refl_VZA(leaf, soil, tts, tto, psi, lai, lidfa, ci_flag)
    r_red_prs, r_nir_prs = [], []
    for x in range(len(tto)):
        sza, vza, raa = tts[x], tto[x], psi[x]
        rho_canopy = prosail.run_prosail(0.9, Cab, Car, Cbrown, Cw, Cm, lai[x], lidfa, hspot, sza, vza, raa, Ant, Alpha, rsoil0=soil[0:2101])
        
        r_red_prs.append(float(np.nansum(rho_canopy[220:271].flatten()*rsr_red.flatten())/np.nansum(rsr_red.flatten())))
        r_nir_prs.append(float(np.nansum(rho_canopy[441:477].flatten()*rsr_nir.flatten())/np.nansum(rsr_nir.flatten())))
        #r_red_ps.append(np.nansum(rho_canopy[220:271]))
        #r_nir_ps.append(np.nansum(rho_canopy[441:477]))
    
    return np.array(r_red_sip), np.array(r_nir_sip), np.array(r_red_prs), np.array(r_nir_prs)

def cal_reflectance_point(lai, tto, tts, psi):
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
    
    leaf = sip_leaf(Cab, Car, Cbrown, Cw, Cm, Ant, Alpha, fLMA_k, gLMA_k, gLMA_b)
    soil = soil_spectra()        
    ci_flag = 1 #Clumping Index is a constant 
    hspot = 0.05
    lidfa = 30   
    
    sza, vza, raa = tts, tto, psi
    rho_canopy = prosail.run_prosail(0.9, Cab, Car, Cbrown, Cw, Cm, lai, lidfa, hspot, sza, vza, raa, Ant, Alpha, rsoil0=soil[0:2101])
    
    r_red_prs = float(np.nansum(rho_canopy[220:271].flatten()*rsr_red.flatten())/np.nansum(rsr_red.flatten()))
    r_nir_prs = float(np.nansum(rho_canopy[441:477].flatten()*rsr_nir.flatten())/np.nansum(rsr_nir.flatten()))
    
    return r_red_prs, r_nir_prs


de_fontsize = 15
num_tto = 10#90
num_psi = 30#360

import numpy as np
#-- Generate Data -----------------------------------------
# Using linspace so that the endpoint of 360 is included...
azimuths = abs(np.linspace(-180, 180, num_psi))
zeniths  = np.linspace(0, 70, num_tto)
xarr, yarr = np.meshgrid(zeniths, azimuths)
zarr0 = np.zeros_like(xarr)
zarr1 = np.zeros_like(xarr)

"""
zarr2 = np.zeros_like(xarr)
zarr3 = np.zeros_like(xarr)
for i in range(len(xarr)):
    for j in range(len(xarr[0])):
        tto, psi = xarr[i,j], yarr[i,j]
        print(tto, psi)
        lai = 1.0
        tts = 30
        r_red_prs, r_nir_prs = cal_reflectance_point(lai, tto, tts, psi)
        zarr2[i,j] = r_nir_prs*(r_nir_prs-r_red_prs)/(r_nir_prs+r_red_prs)
        zarr3[i,j] = r_nir_prs
"""

tto = xarr.flatten()
psi = yarr.flatten()
tts = np.full(len(tto), 30)
lai = np.full(len(tto), 1.0)
r_red_sip, r_nir_sip, r_red_prs, r_nir_prs = cal_reflectance(lai, tto, tts, psi)

nirv_sip = r_nir_sip*(r_nir_sip-r_red_sip)/(r_nir_sip+r_red_sip)
nirv_prs = r_nir_prs*(r_nir_prs-r_red_prs)/(r_nir_prs+r_red_prs)

zarr0 = nirv_sip.reshape(azimuths.size, zeniths.size)
zarr1 = nirv_prs.reshape(azimuths.size, zeniths.size)

r, theta = np.meshgrid(zeniths, np.radians(np.linspace(180, 540, num_psi)%360))
#values = np.full((azimuths.size, zeniths.size), nirv_prs.reshape(azimuths.size, zeniths.size)) 
#sip_list = [r_red_prs, r_nir_prs]
#prs_list = [r_red_prs, r_nir_prs] 
#-- Plot... ------------------------------------------------
linewidth = 2.0 #边框线宽度
ftsize = 16 #字体大小
axlength = 2.0 #轴刻度长度
axwidth = 2.0 #轴刻度宽度
ftfamily = 'Calibri'

fig, ax = plt.subplots(1, 2, figsize=(12,12), subplot_kw=dict(projection='polar'))
ax[0].set_theta_zero_location('N')
ax[0].set_theta_direction(-1)
ax[1].set_theta_zero_location('N')
ax[1].set_theta_direction(-1)  

#values0 = np.full((azimuths.size, zeniths.size), sip_list[i].reshape(azimuths.size, zeniths.size))  
#values1 = np.full((azimuths.size, zeniths.size), prs_list[i].reshape(azimuths.size, zeniths.size))

p1 = ax[0].contourf(theta, r, zarr0, levels = 20, cmap='jet',vmin=0.1, vmax=0.25)
ax[0].contour(theta, r, zarr0, levels = 20, linewidths=1, colors='black')
ax[0].grid(linewidth=1.5, color="black")
ax[0].set_title("(a)", fontsize=ftsize*1.6, family=ftfamily) 
ax[0].tick_params(length = axlength, width = axwidth, labelsize = ftsize*1.1)

p2 = ax[1].contourf(theta, r, zarr1, levels = 20, cmap='jet',vmin=0.1, vmax=0.25)
ax[1].contour(theta, r, zarr1, levels = 20, linewidths=1, colors='black')
ax[1].grid(linewidth=1.5, color="black")
ax[1].set_title("(b)", fontsize=ftsize*1.6, family=ftfamily) 
ax[1].tick_params(length = axlength, width = axwidth, labelsize = ftsize*1.1)

#-- obtaining the colormap limits
vmin,vmax = p2.get_clim()
#-- Defining a normalised scale
cNorm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
#-- Creating a new axes at the right side
ax3 = fig.add_axes([0.9, 0.3, 0.02, 0.4])
ax3.tick_params(length = axlength, width = axwidth, labelsize = ftsize*1.1)

#-- Plotting the colormap in the created axes
cb1 = matplotlib.colorbar.ColorbarBase(ax3, cmap=matplotlib.cm.jet, norm=cNorm)
fig.subplots_adjust(left=0.05,right=0.85)
plt.show()
