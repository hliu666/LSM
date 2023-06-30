# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 00:11:19 2022

@author: hliu
"""

import os 
print(os.getcwd())
import sys 
sys.path.append("../../model")

import numpy as np
from scipy import integrate

from RTM_initial import sip_leaf, soil_spectra, atmoE
from RTM_initial import cal_lidf, weighted_sum_over_lidf_vec, CIxy
from RTM_initial import hemi_initial, dif_initial, hemi_dif_initial
from RTM_Optical import i_hemi, hotspot_calculations_vec, BRF_hemi_func, BRF_dif_func, define_geometric_constant
from RTM_Optical import BRF_hemi_dif_func

from Ebal_initial import calc_extinc_coeff_pars

from TIR import Planck, calc_longwave_irradiance
from PhotoSynth import PhotoSynth

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('font',family='Calibri')

os.chdir("../")


def rtm_o(leaf, soil, tts, tto, psi, lai, lidfa, ci_flag, ci_thres):
    
    rho, tau = leaf
    rho, tau = rho.flatten(), tau.flatten()

    rg = soil
    
    #lidfa = 1    # float Leaf Inclination Distribution at regular angle steps. 
    lidfb = np.inf # float Leaf Inclination Distribution at regular angle steps. 
    lidf  = cal_lidf(lidfa, lidfb)
    
    CI_flag = ci_flag
    CI_thres = ci_thres
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
    td = 1 - iD
    
    p  = 1 - iD/lai  

    rho_hemi     = iD/2/lai        
    rho_dif      = iv/2/lai        
    rho_dif_hemi = iD/2/lai  
 
    wso  = sob*rho + sof*tau

    Tdn   = t0+i0*w*rho_hemi/(1-p*w)
    Tup_o = tv+iD*w*rho2/(1-p*w)
    Rdn   = iD*w*rho_hemi/(1-p*w)
    
    BRFv = wso*kc/ko + i0*w*w*p*rho2/(1-p*w)      
    BRFs = kg*rg
    BRFm = rg*Tdn*Tup_o/(1-rg*Rdn)-t0*rg*tv       
    BRF  = BRFv + BRFs + BRFm

    Tup_hemi = td + iD*w*rho_hemi/(1-p*w)
    
    Rv = sob_vsla*rho + sof_vsla*tau+i0*w**2*p*rho_hemi/(1-p*w) 
    Rs = kgd*rg
    Rm = rg*Tdn*Tup_hemi/(1-rg*Rdn)-t0*rg*td    
    R  = Rv + Rs + Rm   #albedo

    #absorption
    Av    = i0*(1-w)/(1-p*w)
    Aup   = iD*(1-w)/(1-p*w)
    Am    = rg*(Tdn)*(Aup)/(1-rg*(Rdn))
    A_tot = Av + Am    #absorption

    Tdn_dif  = td+iD*w*rho_dif_hemi/(1-p*w)
    Tup_difo = tv+iD*w*rho_dif/(1-p*w)
    Rdn_dif  = iD*w*rho_dif_hemi/(1-p*w)
    
    BRF_difv = sob_vsla_dif*rho+sof_vsla_dif*tau+iD*w**2*p*rho_dif/(1-p*w)          
    BRF_difs = kg_dif*rg    
    BRF_difm = rg*(Tdn_dif)*(Tup_difo)/(1-rg*(Rdn_dif))-td*rg*tv    
    BRF_dif  = BRF_difv + BRF_difs + BRF_difm

    Tup_dif_hemi = td+iD*w*rho_dif_hemi/(1-p*w)
    
    R_difv = sob_vsla_hemi_dif*rho + sof_vsla_hemi_dif*tau+iD*w**2*p*rho_dif_hemi/(1-p*w)    
    R_difs = kgd_dif*rg    
    R_difm = rg*(Tdn_dif)*(Tup_dif_hemi)/(1-rg*(Rdn_dif))-td*rg*td
    R_dif  = R_difv + R_difs + R_difm

    #absorption
    Aup_dif = iD*(1-w)/(1-p*w)
    A_difv  = iD*(1-w)/(1-p*w)
    A_difm  = rg*(Tdn_dif)*(Aup_dif)/(1-rg*(Rdn_dif))
    A_dif   = A_difv + A_difm 

    """    
    [[rs, re], [gs, ge], [bs, be]] = band_Pars
    Red = float(np.nanmean(BRF[rs:re]))
    Green = float(np.nanmean(BRF[gs:ge]))
    Blue = float(np.nanmean(BRF[bs:be])) 
    """
    ebal_pars = [R, R_dif, Rs, A_tot, A_dif, i0, iD] 
    
    red = float(np.nanmean(BRF[220:271]))
    nir = float(np.nanmean(BRF[441:477]))
    nirv = nir*(nir-red)/(nir+red)
    ndvi = (nir-red)/(nir+red)
    
    red_dif = float(np.nanmean(BRF_dif[220:271]))
    nir_dif = float(np.nanmean(BRF_dif[441:477]))
    nirv_dif = nir_dif*(nir_dif-red_dif)/(nir_dif+red_dif)
    ndvi_dif = (nir_dif-red_dif)/(nir_dif+red_dif)

    brf_pars = [nirv, nirv_dif, ndvi, ndvi_dif]
    
    return ebal_pars, brf_pars

def calc_extinction_coeff(lai, cik, sum0):
    ia = np.exp(-cik*lai)
    sum_tL = np.sum(sum0*ia)
    return -np.log(sum_tL)/lai

def calc_ebal_sunsha(ks, extinc_k, extinc_sum0, lai):
    ko = calc_extinction_coeff(lai, extinc_k, extinc_sum0)

    fsun_ = -1/(ks+ko)*(np.exp(-(ks+ko)*lai)-1)*ko
    fsha_ = -1/(ko)*(np.exp(-(ko)*lai)-1)*ko - (-1/(ks+ko)*(np.exp(-(ks+ko)*lai)-1))*ko 
    fsun  = fsun_/(fsun_+fsha_)
    fsha  = fsha_/(fsun_+fsha_)
    
    return [0.3, 0.7, ko]#[fsun, fsha, ko]

def calc_ebal_canopy_pars(leaf, wl, atmoMs, Ls, ebal_rtm_pars):
    rho, tau = leaf
    rho, tau = rho.flatten(), tau.flatten()
    
    w = rho + tau   #leaf single scattering albedo
 
    [t1, t3, t4, t5, t12, t16] = atmoMs
    
    [rsd, rdd, rs, A_tot, A_dif, i0, iD] = ebal_rtm_pars
    
    # radiation fluxes, downward and upward (these all have dimenstion [nwl]
    # first calculate hemispherical reflectances rsd and rdd according to SAIL
    # these are assumed for the reflectance of the surroundings
    # rdo is computed with SAIL as well
    # assume Fd of surroundings = 0 for the momemnt
    # initial guess of temperature of surroundings from Ta;
    Fd    = np.zeros(wl.shape)
    Esun_ = np.pi*t1*t4
    Esun_[Esun_ < 1E-6] = 1E-6
    Esky_ = np.pi/(1-t3*rdd)*(t1*(t5+t12*rsd)+Fd+(1-rdd)*Ls*t3+t16)
    Esky_[Esky_ < 1E-6] = 1E-6
    #Esun_ = max(1E-6,np.pi*t1*t4)
    #Esky_ = max(1E-6,np.pi/(1-t3*rdd)*(t1*(t5+t12*rsd*1.5)+Fd+(1-rdd)*Ls*t3+t16))

    fEsuno,fEskyo,fEsunt,fEskyt = 0*Esun_,0*Esun_,0*Esun_,0*Esun_
    
    epsc = 1 - w
    epss = 1 - rs
    
    A_sun_sun = i0*epsc
    A_sha_sun = (A_tot - i0*epsc)
    A_sha_sha = A_dif
    
    return [Esun_, Esky_, fEsuno, fEskyo, fEsunt, fEskyt, A_sun_sun, A_sha_sun, A_sha_sha, epss]

def ephoton(lambdas):
    #E = phot2e(lambda) calculates the energy content (J) of 1 photon of
    #wavelength lambda (m)

    h       = 6.6262E-34           # [J s]         Planck's constant
    c       = 299792458            # [m s-1]       speed of light
    E       = h*c/lambdas          # [J]           energy of 1 photon
    return E 

def e2phot(lambdas, E):
    #molphotons = e2phot(lambda,E) calculates the number of moles of photons
    #corresponding to E Joules of energy of wavelength lambda (m)
    A           = 6.02214E+23 #Constant of Avogadro
    e           = ephoton(lambdas)
    photons     = E/e
    molphotons  = photons/A
    return molphotons

def calc_netrad_pars(wl, lai, SW, L, ebal_sunsha_pars, ebal_canopy_pars, fdir, fdif):
    [fsun, fsha, _] = ebal_sunsha_pars 
    [Esun_, Esky_, fEsuno, fEskyo, fEsunt, fEskyt, A_sun_sun, A_sha_sun, A_sha_sha, epss] = ebal_canopy_pars
    
    """
    shortwave radiantion 
    """
    Esunto          = 0.001 * integrate.simpson(Esun_[0:2006],wl[0:2006])
    Eskyto          = 0.001 * integrate.simpson(Esky_[0:2006],wl[0:2006])
    Etoto           = Esunto + Eskyto                   #Calculate total fluxes
    fEsuno[0:2006]  = Esun_[0:2006]/Etoto
    fEskyo[0:2006]  = Esky_[0:2006]/Etoto

    fEsuno[0:2006]  = Esun_[0:2006]/Esunto
    fEskyo[0:2006]  = Esky_[0:2006]/Eskyto
    
    #J_o             = wl<3000;                          #find optical spectrum
    #Esunto          = 0.001 * Sint(Esun_(J_o),wl(J_o)); #Calculate optical sun fluxes (by Integration), including conversion mW >> W
    #Eskyto          = 0.001 * Sint(Esky_(J_o),wl(J_o)); #Calculate optical sun fluxes (by Integration)
    #Etoto           = Esunto + Eskyto;                  #Calculate total fluxes
    #fEsuno(J_o)     = Esun_(J_o)/Etoto;                 #fraction of contribution of Sun fluxes to total light
    #fEskyo(J_o)     = Esky_(J_o)/Etoto;                 #fraction of contribution of Sky fluxes to total light
    
    Esun_[0:2006] = fEsuno[0:2006]*SW*fdir
    Esky_[0:2006] = fEskyo[0:2006]*SW*fdif

    """
    Calculate APAR
    """
    Ipar  = 301
    wlPAR = wl[0:Ipar]
    
    Pnsun  = 0.001 * integrate.simpson(e2phot(wlPAR*1E-9,Esun_[0:Ipar]*A_sun_sun[0:Ipar]),wlPAR)
    Pndir  = Pnsun * 1E6 # 
    Pnsky  = 0.001 * integrate.simpson(e2phot(wlPAR*1E-9,(Esky_*A_sha_sha+Esun_*A_sha_sun)[0:Ipar]),wlPAR)
    Pndif  = Pnsky * 1E6 

    APARu = Pndir + Pndif*fsun
    APARh = Pndif*fsha
        
    """
    longwave radiantion 
    """        
    Esuntt         = 0.001 * integrate.simpson(Esun_[2006:],wl[2006:])
    Eskytt         = 0.001 * integrate.simpson(Esky_[2006:],wl[2006:])
    Etott          = Eskytt + Esuntt
    fEsunt[2006:]  = Esun_[2006:]/Etott
    fEskyt[2006:]  = Esky_[2006:]/Etott    
    #J_t             = wl>=3000;                         %find thermal spectrum
    #Esuntt          = 0.001 * Sint(Esun_(J_t),wl(J_t)); %Themal solar fluxes
    #Eskytt          = 0.001 * Sint(Esky_(J_t),wl(J_t)); %Thermal Sky fluxes
    #Etott           = Eskytt + Esuntt;                  %Total
    #fEsunt(J_t)     = Esun_(J_t)/Etott;                 %fraction from Esun
    #fEskyt(J_t)     = Esky_(J_t)/Etott;                 %fraction from Esky
  
    Esun_[2006:] = fEsunt[2006:]*L
    Esky_[2006:] = fEskyt[2006:]*L
    
    Rndir = 0.001 * integrate.simpson(Esun_*A_sun_sun,wl)
    Rndif = 0.001 * integrate.simpson((Esky_*A_sha_sha+Esun_*A_sha_sun),wl)   # Full spectrum net diffuse flux

    ERnuc = Rndir + Rndif*fsun
    ERnhc = Rndif*fsha

    #soil layer, direct and diffuse radiation
    Rsdir  = 0.001 * integrate.simpson(Esun_*epss,wl) # Full spectrum net diffuse flux
    Rsdif_ = (Esky_*(1-A_sha_sha)+Esun_*(1-A_sun_sun-A_sha_sun))*epss
    Rsdif  = 0.001 * integrate.simpson(Rsdif_, wl) # Full spectrum net diffuse flux

    ERnus = Rsdir + Rsdif        # [1] Absorbed solar flux by the soil
    ERnhs = Rsdif                # [1] Absorbed diffuse downward flux by the soil (W m-2)
 
    return [ERnuc, ERnhc, ERnus, ERnhs, APARu, APARh], [Esun_, Esky_]
 
def cal_reflectance(SW, ci, lai, fdif):
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
    
    tts = np.array([0.0])
    psi = np.array([180.0]) 
    tto = np.array([45.0])
    leaf = sip_leaf(Cab, Car, Cbrown, Cw, Cm, Ant, Alpha, fLMA_k, gLMA_k, gLMA_b)
    soil = soil_spectra()        

    lidfa = 63.6 
    lidfb = 0
    hspot = 0.05
    ci_flag = 1 #Clumping Index is a constant 
    ci_thres = ci
        
    lidf  = cal_lidf(lidfa, lidfb)
    extinc_k, extinc_sum0 = calc_extinc_coeff_pars(ci_flag, ci_thres, lidf)
    _, _, ks, ko, _, sob, sof = weighted_sum_over_lidf_vec(lidf, tts, tto, psi)

    ebal_rtm_pars, brf_pars = rtm_o(leaf, soil, tts, tto, psi, lai, lidfa, ci_flag, ci_thres)
    [nirv, nirv_dif, ndvi, ndvi_dif] = brf_pars
    
    Ta = np.array([25.0])
    ea = 40.0
    ecu = ea 
    ech = ea 
    
    o = 209.0
    p = 970.0    
    
    Ccu = 390.0
    Cch = 390.0
    
    Tcu = Ta[0]
    Tch = Ta[0]
    
    wl, atmoMs = atmoE()
    Ls = Planck(wl, Ta+273.15)
    L  = calc_longwave_irradiance(ea, Ta+273.15)
    ebal_sunsha_pars  = calc_ebal_sunsha(ks, extinc_k, extinc_sum0, lai)
    ebal_canopy_pars  = calc_ebal_canopy_pars(leaf, wl, atmoMs, Ls, ebal_rtm_pars)    
    net_rads, Esolars = calc_netrad_pars(wl, lai, SW, L, ebal_sunsha_pars, ebal_canopy_pars, 1-fdif, fdif)
    
    [_, _, _, _, _, _, A_sun_sun, A_sha_sun, A_sha_sha, _] = ebal_canopy_pars
    [fsun, fsha, _] = ebal_sunsha_pars 
    [_, _, _, _, APARu, APARh] = net_rads
    
    meteo_u = [APARu, Ccu, Tcu, ecu, o, p]
    meteo_h = [APARh, Cch, Tch, ech, o, p]

    bcu_rcw, bcu_Ci, bcu_An = PhotoSynth(meteo_u)
    bch_rcw, bch_Ci, bch_An = PhotoSynth(meteo_h)

    return bcu_An*fsun*lai, bch_An*fsha*lai
           
sw_arr = np.linspace(0.0, 550.0, num=550)
aPARu_list, aPARh_list,  aPAR_list = [], [], []
ci = 1.0
lai = 4.0
for fdif in [0.0, 0.25, 0.5, 0.75, 1.0]:
    aPARu_ci, aPARh_ci, aPAR_ci = [], [], []

    for sw in sw_arr:  
        APARu, APARh = cal_reflectance(sw, ci, lai, fdif)
        
        aPARu_ci.append(APARu) 
        aPARh_ci.append(APARh)  
        aPAR_ci.append(APARu+APARh)
        
    aPARu_list.append(aPARu_ci) 
    aPARh_list.append(aPARh_ci)  
    aPAR_list.append(aPAR_ci)  

def_fontsize = 30
def_linewidth = 3.5

linewidth = 1.8 #边框线宽度
ftsize = 20 #字体大小
axlength = 5.0 #轴刻度长度
axwidth = 3.0 #轴刻度宽度

fig, ax = plt.subplots(1, 3, figsize=(19.5,8))

ax[0].set_title("(a) Sunlit canopy", fontsize=def_fontsize/1.1)
ax[0].plot(sw_arr, aPARu_list[0],  linewidth=def_linewidth, label="DRF = 0.0", color="black")
ax[0].plot(sw_arr, aPARu_list[1],  linewidth=def_linewidth, label="DRF = 0.25", color="red")
ax[0].plot(sw_arr, aPARu_list[2],  linewidth=def_linewidth, label="DRF = 0.5", color="blue")
ax[0].plot(sw_arr, aPARu_list[3],  linewidth=def_linewidth, label="DRF = 0.75", color="darkorange")
ax[0].plot(sw_arr, aPARu_list[4],  linewidth=def_linewidth, label="DRF = 1.0", color="green")
ax[0].set_xlabel('incident radiation (W m\u207B\u00B2)'.format(chr(956)), fontsize=def_fontsize)
ax[0].set_ylabel('Photosynthesis ({0}mol CO₂m\u207B\u00B2s\u207B\u00B9)'.format(chr(956)), fontsize=def_fontsize)
#ax[0].legend(loc='lower left', fancybox = False, shadow = False,frameon = False, ncol = 1, fontsize=def_fontsize/1.5)  

ax[0].spines['left'].set_linewidth(linewidth)
ax[0].spines['right'].set_linewidth(linewidth)
ax[0].spines['top'].set_linewidth(linewidth)
ax[0].spines['bottom'].set_linewidth(linewidth)
ax[0].tick_params(direction = 'in', axis='both', length = axlength, width = axwidth, labelsize = ftsize)        
  

ax[1].set_title("(b) Shaded canopy", fontsize=def_fontsize/1.1)
ax[1].plot(sw_arr, aPARh_list[0],  linewidth=def_linewidth, label="DRF = 0.0", color="black")
ax[1].plot(sw_arr, aPARh_list[1],  linewidth=def_linewidth, label="DRF = 0.25", color="red")
ax[1].plot(sw_arr, aPARh_list[2],  linewidth=def_linewidth, label="DRF = 0.5", color="blue")
ax[1].plot(sw_arr, aPARh_list[3],  linewidth=def_linewidth, label="DRF = 0.75", color="darkorange")
ax[1].plot(sw_arr, aPARh_list[4],  linewidth=def_linewidth, label="DRF = 1.0", color="green")
ax[1].set_xlabel('incident radiation (W m\u207B\u00B2)'.format(chr(956)), fontsize=def_fontsize)
ax[1].set_ylabel('Photosynthesis ({0}mol CO₂m\u207B\u00B2s\u207B\u00B9)'.format(chr(956)), fontsize=def_fontsize)
#ax[1].legend(loc='lower right', fancybox = False, shadow = False,frameon = False, ncol = 1, fontsize=def_fontsize/1.5)  


ax[1].spines['left'].set_linewidth(linewidth)
ax[1].spines['right'].set_linewidth(linewidth)
ax[1].spines['top'].set_linewidth(linewidth)
ax[1].spines['bottom'].set_linewidth(linewidth)
ax[1].tick_params(direction = 'in', axis='both', length = axlength, width = axwidth, labelsize = ftsize)        
  
ax[2].set_title("(c) Total canopy", fontsize=def_fontsize/1.1)
p1, =ax[2].plot(sw_arr, aPAR_list[0],  linewidth=def_linewidth, label="DRF = 0.0", color="black")
p2, =ax[2].plot(sw_arr, aPAR_list[1],  linewidth=def_linewidth, label="DRF = 0.25", color="red")
p3, =ax[2].plot(sw_arr, aPAR_list[2],  linewidth=def_linewidth, label="DRF = 0.5", color="blue")
p4, =ax[2].plot(sw_arr, aPAR_list[3],  linewidth=def_linewidth, label="DRF = 0.75", color="darkorange")
p5, =ax[2].plot(sw_arr, aPAR_list[4],  linewidth=def_linewidth, label="DRF = 1.0", color="green")
ax[2].set_xlabel('incident radiation (W m\u207B\u00B2)'.format(chr(956)), fontsize=def_fontsize)
ax[2].set_ylabel('Photosynthesis ({0}mol CO₂m\u207B\u00B2s\u207B\u00B9)'.format(chr(956)), fontsize=def_fontsize)
#ax[2].legend(loc='lower right', fancybox = False, shadow = False,frameon = False, ncol = 1, fontsize=def_fontsize/1.5)  

ax[2].spines['left'].set_linewidth(linewidth)
ax[2].spines['right'].set_linewidth(linewidth)
ax[2].spines['top'].set_linewidth(linewidth)
ax[2].spines['bottom'].set_linewidth(linewidth)
ax[2].tick_params(direction = 'in', axis='both', length = axlength, width = axwidth, labelsize = ftsize)        
  

handles = [p1, p2, p3, p4, p5]
labels = ['DRF = 0.0', 'DRF = 0.25', "DRF = 0.5", "DRF = 0.75", "DRF = 1.0"]  


fig.tight_layout()         
fig.subplots_adjust(bottom=0.24)

fig.legend(handles, labels, loc ='lower center', bbox_to_anchor=(0.5, 0.02),
          fancybox = False, shadow = False,frameon = False, ncol = 5, 
          handletextpad = 0.2, columnspacing = 1.2, prop={'family':"Calibri", 'size':ftsize*1.5})  

             
plot_path = "../../figs/fpar/fpar_rad.jpg"
fig.savefig(plot_path, dpi=600, bbox_inches = 'tight')    






        