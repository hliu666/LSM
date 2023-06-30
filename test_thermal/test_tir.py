# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 09:08:46 2022

@author: hliu

"""
import sys 
sys.path.append(r"C:\Users\liuha\Desktop\dalecv5.0\src\model")
from TIR import rtm_t

import four_sail
import numpy as np
SB = 5.670373e-8  # Stephan Boltzmann constant (W m-2 K-4)
from RTM_initial import cal_lidf, weighted_sum_over_lidf
from RTM_Optical import define_geometric_constant, hotspot_calculations, i_hemi
from Ebal import calc_longwave_irradiance

def CalcStephanBoltzmann(T_K):
    '''Calculates the total energy radiated by a blackbody.
    
    Parameters
    ----------
    T_K : float
        body temperature (Kelvin).
    
    Returns
    -------
    M : float
        Emitted radiance (W m-2).'''

    M = SB * T_K ** 4
    return np.asarray(M)

def run_TIR(emisVeg, emisSoil, T_Veg, T_Soil, LAI, hot_spot, solar_zenith, solar_azimuth, view_zenith, view_azimuth,
            lidf, T_VegSunlit=None, T_SoilSunlit=None, T_atm=0):
    ''' Estimates the broadband at-sensor thermal radiance using 4SAIL model.
    
    Parameters
    ----------
    emisVeg : float
        Leaf hemispherical emissivity.
    emisSoil : float
        Soil hemispherical emissivity.
    T_Veg : float
        Leaf temperature (Kelvin).
    T_Soil : float
        Soil temperature (Kelvin).
    LAI : float
        Leaf Area Index.
    hot_spot : float
        Hotspot parameter.
    solar_zenith : float
        Sun Zenith Angle (degrees).
    solar_azimuth : float
        Sun Azimuth Angle (degrees).
    view_zenith : float
        View(sensor) Zenith Angle (degrees).
    view_azimuth : float
        View(sensor) Zenith Angle (degrees).
    LIDF : float or tuple(float,float)
        Leaf Inclination Distribution Function parameter.
        
            * if float, mean leaf angle for the Cambpell Spherical LIDF.
            * if tuple, (a,b) parameters of the Verhoef's bimodal LIDF |LIDF[0]| + |LIDF[1]|<=1.
    T_VegSunlit : float, optional
        Sunlit leaf temperature accounting for the thermal hotspot effect,
        default T_VegSunlit=T_Veg.
    T_SoilSunlit : float, optional
        Sunlit soil temperature accounting for the thermal hotspot effect
        default T_SoilSunlit=T_Soil.
    T_atm : float, optional
        Apparent sky brightness temperature (Kelvin), 
        default T_atm =0K (no downwellig radiance).
    
    Returns
    -------
    Lw : float
        At sensor broadband radiance (W m-2).
    TB_obs : float
        At sensor brightness temperature (Kelvin).
    emiss : float
        Surface directional emissivity.

    References
    ----------
    .. [Verhoef2007] Verhoef, W.; Jia, Li; Qing Xiao; Su, Z., (2007) Unified Optical-Thermal
        Four-Stream Radiative Transfer Theory for Homogeneous Vegetation Canopies,
        IEEE Transactions on Geoscience and Remote Sensing, vol.45, no.6, pp.1808-1822,
        http://dx.doi.org/10.1109/TGRS.2007.895844.
    '''
    T_Veg += 273.15
    T_Soil += 273.15
    T_atm += 273.15
    
    T_VegSunlit += 273.15
    T_SoilSunlit += 273.15   
    # Apply Kirchoff's law to get the soil and leaf bihemispherical reflectances
    rsoil = 1 - emisSoil
    rho_leaf = 1 - emisVeg
    tau_leaf = 0

    # Get the relative sun-view azimth angle
    psi = abs(solar_azimuth - view_azimuth)
    # 4SAIL for canopy reflectance and transmittance factors       
    [tss, too, tsstoo, rdd, tdd, rsd, tsd, rdo, tdo, rso, rsos, rsod, rddt, rsdt, rdot,
     rsodt, rsost, rsot, gammasdf, gammasdb,
     gammaso] = four_sail.foursail(LAI, hot_spot,
                                   lidf, solar_zenith, view_zenith, psi, rho_leaf, tau_leaf, rsoil)

    tso = tss * too + tss * (tdo + rsoil * rdd * too) / (1. - rsoil * rdd)
    gammad = 1 - rdd - tdd
    gammao = 1 - rdo - tdo - too
    ttot = (too + tdo) / (1. - rsoil * rdd)
    gammaot = gammao + ttot * rsoil * gammad
    gammasot = gammaso + ttot * rsoil * gammasdf

    aeev = gammaot
    aees = ttot * emisSoil

    # Get the different canopy broadband emssion components
    Hvc = CalcStephanBoltzmann(T_Veg)
    Hgc = CalcStephanBoltzmann(T_Soil)
    Hsky = CalcStephanBoltzmann(T_atm)

    if T_VegSunlit:  # Accout for different suntlit shaded temperatures
        Hvh = CalcStephanBoltzmann(T_VegSunlit)
    else:
        Hvh = Hvc
    if T_SoilSunlit:  # Accout for different suntlit shaded temperatures
        Hgh = CalcStephanBoltzmann(T_SoilSunlit)
    else:
        Hgh = Hgc

    # Calculate the blackbody emission temperature
    Lw = (rdot * Hsky + (
                aeev * Hvc + 
                gammasot * emisVeg * (Hvh - Hvc) + 
                aees * Hgc + 
                tso * emisSoil * (Hgh - Hgc))) / np.pi
    TB_obs = (np.pi * Lw / SB) ** (0.25)
    print(round(rdot * Hsky,2), 
          round(aeev * Hvc ,2), 
          round(gammasot * emisVeg * (Hvh - Hvc),2), 
          round(aees * Hgc, 2),
          round(tso * emisSoil * (Hgh - Hgc),2))
    # Estimate the apparent surface directional emissivity
    emiss = 1 - rdot
    return Lw, TB_obs-273.15, emiss

def rtm_t(lai, L, i0, iD, Tcu, Tch, Tsu, Tsh, k_pars):
    """
    The top-of-canopy TIR radiance (TIR) at viewing angle 

    Returns
    -------
    None.

    """
    rho         = 0.01                    # [1]               Leaf/needle reflection
    tau         = 0.01                    # [1]               Leaf/needle transmission
    rs          = 0.06                    # [1]               Soil reflectance
    emisVeg     = 1-rho-tau               # [nwl]             Emissivity vegetation
    emisGrd     = 1-rs                    # [nwl]             Emissivity soil 
    w = rho + tau                         # [1]               leaf single scattering albedo

    [kc, kg] = k_pars
    
    L0   = L*(iD*(1-emisVeg)*rho + (1-iD)*(1-emisGrd)*rs)

    """
    sip based thermal radiative transfer model
    """
    i = max(1 - np.exp(-kc * lai * CIs), 0.00001)
    Fc, Fs = i/abs(np.log(1-i)), 1-i
    
    ed, eu = i/(2*lai), i/(2*lai)
    p  = 1 - i/lai 
    
    rc1 = w*ed/(1-w*p)
    rc2 = w*eu/(1-w*p)
    
    Aup  = i*emisVeg/(1-p*(1-emisVeg))
    Rdnc = (1-emisGrd)*i/(1-rc2*(1-emisGrd)*i)
    e1  = i*emisVeg/(1-p*(1-emisVeg))
    e2  = (1-i)*Rdnc*Aup
    e3  = i*rc1*Rdnc*Aup
    
    Rdns = emisGrd/(1-(1-emisGrd)*i*rc2)
    e4   = (1-i)*Rdns
    e5   = i*rc1*Rdns
    
    alphav = (e1 + e2 + e3)
    alphas = (e4 + e5)

    #print(round(Tcu,2), round(Tch,2), round(Tsu,2), round(Tsh,2))

    Hcu = CalcStephanBoltzmann(Tcu+273.15)
    Hch = CalcStephanBoltzmann(Tch+273.15)
    Hsu = CalcStephanBoltzmann(Tsu+273.15)
    Hsh = CalcStephanBoltzmann(Tsh+273.15)

    TIRv = Fc*Hcu*alphav + (1-Fc)*Hch*alphav
    TIRs = Fs*Hsu*alphas + (1-Fs)*Hsh*alphas 
    #TIRv = Fc*Planck(wl,Tcu)*alphav + (1-Fc)*Planck(wl,Tch)*alphav
    #TIRs = Fs*Planck(wl,Tsu)*alphas + (1-Fs)*Planck(wl,Tsh)*alphas

    TIRt = TIRv + TIRs + L0
    print(round(TIRv,2), 
          round(TIRs,2), 
          round(L0,2))
    
    emis = alphav + alphas
    LST  = (TIRt/SB)**0.25 - 273.15
    
    Ts  = Fs*Tsu + (1-Fs)*Tsh
    Tc  = Fc*Tcu + (1-Fc)*Tch
    
    #print(Ts, Tc, LST) 
    if np.isnan(LST):
        print("Double Check LST!")

    return LST  

rho         = 0.01                    # [1]               Leaf/needle reflection
tau         = 0.01                    # [1]               Leaf/needle transmission
rs          = 0.06                    # [1]               Soil reflectance
emisVeg     = 1-rho-tau               # [nwl]             Emissivity vegetation
emisSoil    = 1-rs                    # [nwl]             Emissivity soil 
hot_spot    = 0.05

lidfa = 30    # float Leaf Inclination Distribution at regular angle steps. 
lidfb = np.inf # float Leaf Inclination Distribution at regular angle steps. 
lidf  = cal_lidf(lidfa, lidfb)

solar_zenith, solar_azimuth, view_zenith, view_azimuth = 30, 0, 45, 0
lai = 0.1
T_Veg, T_Soil = 1.3, 4.0
T_atm = 1.0

#lai = 4.6
#T_Veg, T_Soil = 32, 40
#T_atm = 30

T_VegSunlit, T_SoilSunlit = T_Veg, T_Soil
ea = 40

Lw, TB_sail, emiss = run_TIR(emisVeg, emisSoil, T_Veg, T_Soil, lai, hot_spot, solar_zenith, solar_azimuth, view_zenith, view_azimuth,
        lidf, T_VegSunlit, T_SoilSunlit, T_atm)
#print(Lw, TB_sail, emiss)

Tcu,Tch = T_Veg, T_Veg
Tsu,Tsh = T_Soil,T_Soil
CIs, CIo = 1.0, 1.0
CI_thres = 1.0
CI_flag = 1
tts, tto, psi = solar_zenith, view_zenith, abs(solar_azimuth-view_azimuth)

_, _, ks, ko, _, sob, sof = weighted_sum_over_lidf(lidf, tts, tto, psi)
dso = define_geometric_constant(tts, tto, psi)
k_pars = hotspot_calculations(lai, ko, ks, CIo, CIs, dso)

i0 = max(1 - np.exp(-ks * lai * CIs), 0.00001)
iD = i_hemi(CI_flag, lai, lidf, CI_thres) 

L = calc_longwave_irradiance(ea, T_atm+273.15)
TB_sip = rtm_t(lai, L, i0, iD, Tcu, Tch, Tsu, Tsh, k_pars)
print(TB_sail, TB_sip)