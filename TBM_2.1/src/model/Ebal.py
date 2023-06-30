# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 11:15:12 2022

@author: Haoran 

Energy Balance Model 
"""
import numpy as np
from resistance import resistance 
from PhotoSynth_Jen import PhotoSynth_Jen
from TIR import rtm_t, calc_netrad
from TIR import calc_ebal_sunsha, calc_ebal_canopy_pars, calc_netrad_pars
from TIR import calc_lambda, calc_longwave_irradiance
from TIR import Planck
from Ebal_pars import *
from hydraulics import calc_sf, calc_fwet, calc_beta_e

import warnings
warnings.filterwarnings('ignore')


def Monin_Obukhov(ustar, T_A_C, rho, c_p, H, LE):
    '''Calculates the Monin-Obukhov length.

    Parameters
    ----------
    ustar : float
        friction velocity (m s-1).
    T_A_K : float
        air temperature (Kelvin).
    rho : float
        air density (kg m-3).
    c_p : float
        Heat capacity of air at constant pressure (J kg-1 K-1).
    H : float
        sensible heat flux (W m-2).
    LE : float
        latent heat flux (W m-2).

    Returns
    -------
    L : float
        Obukhov stability length (m).

    References
    ----------
    .. [Brutsaert2005] Brutsaert, W. (2005). Hydrology: an introduction (Vol. 61, No. 8).
        Cambridge: Cambridge University Press.'''

    T_A_K = T_A_C + 273.15
    
    # first convert latent heat into rate of surface evaporation (kg m-2 s-1)
    Lambda = calc_lambda(T_A_K)  # in J kg-1
    
    E = LE / Lambda
    del LE, Lambda
    
    # Virtual sensible heat flux
    Hv = H + (0.61 * T_A_K * c_p * E)
    del H, E

    L_const = KARMAN * GRAVITY / T_A_K
    L = -ustar**3 / (L_const * (Hv / (rho * c_p)))
    return L

def calc_veg_fluxes(rb,rs,fwet,Tc,Ta,Ci,ea,c_p,rho): 
    """
    # this function calculates latent and sensible heat flux in vegetation canopy
    #
    # input:
    #   rb          leaf boundary layer resistance          s m-1
    #   rs          stomatal resistance                     s m-1
    #   fwet        wetted fraction of the canopy                 
    #   Tc          leaf temperature                        oC
    #   Ta          air temperature above canopy            oC
    #   ea          vapour pressure above canopy            hPa
    #   e_to_q      conv. from vapour pressure to abs hum   hPa-1
    #   Ca          ambient CO2 concentration               umol m-3
    #   Ci          intercellular CO2 concentration         umol m-3
    #
    # output:
    #   lEc         latent heat flux of a leaf              W m-2
    #   Hc          sensible heat flux of a leaf            W m-2
    #   ec          vapour pressure at the leaf surface     hPa
    #   Cc          CO2 concentration at the leaf surface   umol m-3
    """

    Lambda = calc_lambda(Tc+T2K)   # [J kg-1] Evapor. heat (J kg-1)
    ei = calc_vapor_pressure(Tc)
    s  = calc_delta_vapor_pressure(Tc)

    qi = ei * e_to_q
    qa = ea * e_to_q
    
    lE = rho*(fwet/rb+(1-fwet)/(rb+rs))*Lambda*(qi-qa)     # [W m-2] Latent heat flux
    H  = (rho*c_p)/rb*(Tc-Ta)           # [W m-2] Sensible heat flux
    ec = ea + (ei-ea)*rb/(rb+rs)        # [W m-2] vapour pressure at the leaf surface
    Cc = Ca - (Ca-Ci)*rb/(rb+rs)        # [umol m-2 s-1] CO2 concentration at the leaf surface
    
    return [lE, H, ec, Cc, Lambda, s]

def calc_soil_fluxes(ra,rs,Tg,Ta,ea,c_p,rho): 
    """
    # this function calculates latent and sensible heat flux
    #
    # input:
    #   ra          aerodynamic resistance between ground(z0) and d+z0       s m-1
    #   rs          surface resistance                      s m-1
    #   Tg          ground temperature                        oC
    #   ea          vapour pressure above canopy            hPa
    #   Ta          air temperature above canopy            oC
    #   e_to_q      conv. from vapour pressure to abs hum   hPa-1
    #   Ca          ambient CO2 concentration               umol m-3
    #
    # output:
    #   lEc         latent heat flux of a leaf              W m-2
    #   Hc          sensible heat flux of a leaf            W m-2
    """

    Lambda = calc_lambda(Tg+T2K)   # [J kg-1] Evapor. heat (J kg-1)
    ei = calc_vapor_pressure(Tg)
    s  = calc_delta_vapor_pressure(Tg)

    qi = ei * e_to_q
    qa = ea * e_to_q
    
    lE = rho/(ra+rs)*Lambda*(qi-qa)     # [W m-2] Latent heat flux
    H  = (rho*c_p)/ra*(Tg-Ta)           # [W m-2] Sensible heat flux
    
    return [lE, H, Lambda, s]

def Ebal(dC, x, lai, ebal_rtm_pars, k_pars):    
    
    """
    # 1. initialisations and other preparations for the iteration loop
    # parameters for the closure loop
    """
    counter     = 0                # iteration counter of ebal
    maxit       = 50               # maximum number of iterations
    maxEBer     = 0.1              # maximum energy balance error (any leaf) [Wm-2]
    Wc          = 1                # update step (1 is nominal, [0,1] possible)
    CONT        = 1                # boolean indicating whether iteration continues
    
    # meteo
    Ta          = dC.t_mean[x]
    ea          = calc_vapor_pressure(Ta)
    rho         = calc_rho(p, ea, Ta+T2K)
    c_p         = calc_c_p(p, ea)
    L           = calc_longwave_irradiance(ea, Ta+T2K)
    SW          = dC.sw[x]
    
    # initial values for the loop
    ecu         = ea                      # Leaf boundary vapour pressure (sunlit leaves)
    ech         = ea                      # Leaf boundary vapour pressure (shaded leaves)
   
    Ccu         = Ca                      # Leaf boundary CO2 (sunlit leaves)   
    Cch         = Ca                      # Leaf boundary CO2 (shaded leaves)      
    
    Tsu         = (Ta + 3.0)              # soil temperature (+3 for a head start of the iteration) 
    Tsh         = (Ta + 3.0)              # soil temperature (+3 for a head start of the iteration) 
    Tcu         = (Ta + 0.3)              # leaf tempeFrature (sunlit leaves)
    Tch         = (Ta + 0.1)              # leaf temperature (shaded leaves)    
    
    l_mo        = -1E6                    # Monin-Obukhov length

    T_Pars      = [Ta, Tcu, Tch, Tsu, Tsh]    
    
    ET          = 0                       # Initial evapotranspiration
    Ev          = 0                       # Initial evaporation of canopy

    """
    ## 2.1 Energy balance iteration loop
    #Energy balance loop (Energy balance and radiative transfer)
    """
    wl = dC.wl
    Ls = Planck(wl, Ta+273.15)
    ebal_sunsha_pars  = calc_ebal_sunsha(dC, x, lai)
    ebal_canopy_pars  = calc_ebal_canopy_pars(dC, x, Ls, ebal_rtm_pars)    
    net_rads, Esolars = calc_netrad_pars(dC, x, lai, SW, L, ebal_sunsha_pars, ebal_canopy_pars)
    while CONT:                          # while energy balance does not close
        
        #----------------------canopy intercepted wator---------------------
        _, fwet, through_fall = calc_fwet(dC.w_can[x], dC.precip[x], lai, Ev)
        
        #----------------------soil moisture factor---------------------
        _, sf = calc_sf(dC.Soil, dC.sm_top[x], through_fall, ET)

        #----------------------soil moisture factor---------------------        
        beta_e = calc_beta_e(dC.Soil, dC.sm_top[x])
        
        rad_Rnuc, rad_Rnhc, APARu, APARh, PAR, rad_Rnus, rad_Rnhs, Fc, Fs, i0, iD, Rns = calc_netrad(dC, x, lai, L, T_Pars, net_rads, ebal_rtm_pars, ebal_sunsha_pars)
        
        if (dC.tts[x] < 75) and (lai > 0.5): 
            APARu = max(APARu, 1e-16) 
            APARh = max(APARh, 1e-16)  

            APARu_leaf, APARh_leaf = APARu/(lai*Fc), APARh/(lai*(1-Fc))
            meteo_u = [APARu_leaf, Ccu, Tcu, ecu, sf]
            meteo_h = [APARh_leaf, Cch, Tch, ech, sf]
            
            cu_rcw, cu_Ci, _, _, _ = PhotoSynth_Jen(meteo_u)
            ch_rcw, ch_Ci, _, _, _ = PhotoSynth_Jen(meteo_h)

        else:
            cu_rcw, cu_Ci = 4160, Ca
            ch_rcw, ch_Ci = 4160, Ca
            
        # Aerodynamic roughness
        # calculate friction velocity [m s-1] and aerodynamic resistance [s m-1]  
        u_star, R_x, R_s = resistance(lai, l_mo, dC.wds[x])
        rac     = (lai+1)*R_x
        ras     = (lai+1)*R_s

        rss     = ras*(1-beta_e)/beta_e
        
        #print(round(beta_e,3), round(rss,1))
        
        # calculate the latent and sensible heat
        [lEcu,Hcu,ecu,Ccu,lambdau,su] = calc_veg_fluxes(rac,cu_rcw,fwet,Tcu,Ta,cu_Ci,ea,c_p,rho)
        [lEch,Hch,ech,Cch,lambdah,sh] = calc_veg_fluxes(rac,ch_rcw,fwet,Tch,Ta,ch_Ci,ea,c_p,rho)
        [lEsu,Hsu,lambdasu,ssu] = calc_soil_fluxes(ras,rss,Tsu,Ta,ea,c_p,rho)
        [lEsh,Hsh,lambdash,ssh] = calc_soil_fluxes(ras,rss,Tsh,Ta,ea,c_p,rho)
        
        # integration over the layers and sunlit and shaded fractions
        Hctot   = Fc*Hcu + (1-Fc)*Hch
        Hstot   = Fs*Hsu + (1-Fs)*Hsh
        Htot    = Hstot + Hctot*lai

        lEctot   = Fc*lEcu + (1-Fc)*lEch
        lEstot   = Fs*lEsu + (1-Fs)*lEsh
        lEtot    = lEstot + lEctot*lai

        l_mo = Monin_Obukhov(u_star, Ta, rho, c_p, Htot, lEtot)
        # ground heat flux
        soil_rs_thermal = 0.06 #broadband soil reflectance in the thermal range 1 - emissivity
        
        Gu  = 0.35*rad_Rnus
        Gh  = 0.35*rad_Rnhs
        
        dGu = 4*(1-soil_rs_thermal)*sigmaSB*(Tsu+273.15)**3*0.35
        dGh = 4*(1-soil_rs_thermal)*sigmaSB*(Tsh+273.15)**3*0.35

        # energy balance errors, continue criterion and iteration counter
        EBercu  = rad_Rnuc/(lai*Fc)     -lEcu -Hcu
        EBerch  = rad_Rnhc/(lai*(1-Fc)) -lEch -Hch
        EBersu  = rad_Rnus -lEsu -Hsu - Gu
        EBersh  = rad_Rnhs -lEsh -Hsh - Gh
   
        # evapotranspiration and iteration counter      
        ETv = lai*(Fc*lEcu/lambdau + (1-Fc)*lEch/lambdah)
        ETs = Fs*lEsu/lambdasu + (1-Fs)*lEsh/lambdash 
        ET = ETv + ETs  
        
        counter     = counter + 1                   #Number of iterations

        maxEBercu   = abs(EBercu)
        maxEBerch   = abs(EBerch)
        maxEBers    = max(abs(EBersu), abs(EBersh))

        CONT        = (((maxEBercu >   maxEBer)   or
                        (maxEBerch >   maxEBer)   or
                        (maxEBers  >   maxEBer))  and
                        (counter   <   maxit+1))
        if counter==10:
            Wc = 0.8
        if counter==20:
            Wc = 0.6
            
        #l.append([rad_Rnuc, rad_Rnhc, ERnuc, ERnhc, ELnuc, ELnhc, lEcu, lEch, Hcu, Hch, Tcu, Tch])
        # if counter>99, plot(EBercu(:)), hold on, end
        # 2.7. New estimates of soil (s) and leaf (c) temperatures, shaded (h) and sunlit (1)
        leafbio_emis = 0.98
        Tcu         = Tcu + Wc*EBercu/((rho*c_p)/rac + rho*lambdau *e_to_q*su/(rac+cu_rcw) + 4*leafbio_emis       *sigmaSB*(Tcu+273.15)**3)        
        Tch         = Tch + Wc*EBerch/((rho*c_p)/rac + rho*lambdah *e_to_q*sh/(rac+ch_rcw) + 4*leafbio_emis       *sigmaSB*(Tch+273.15)**3)
        Tsu         = Tsu + Wc*EBersu/((rho*c_p)/ras + rho*lambdasu*e_to_q*ssu/(ras+rss)   + 4*(1-soil_rs_thermal)*sigmaSB*(Tsu+273.15)**3 + dGu)        
        Tsh         = Tsh + Wc*EBersh/((rho*c_p)/ras + rho*lambdash*e_to_q*ssh/(ras+rss)   + 4*(1-soil_rs_thermal)*sigmaSB*(Tsh+273.15)**3 + dGh)
        
    if abs(Tcu-Ta) > 10 or np.isnan(Tcu):
        if x > 0:
            Tcu = Ta + (dC.t_mean[x]-dC.t_mean[x-1])  
        else:
            Tcu = Ta + 0.3
    if abs(Tch-Ta) > 10 or np.isnan(Tch):
        if x > 0:
            Tch = Ta + (dC.t_mean[x]-dC.t_mean[x-1])  
        else:
            Tch = Ta + 0.1
    if abs(Tsu-Ta) > 10 or np.isnan(Tsu):
        if x > 0:
            Tsu = Ta + (dC.t_mean[x]-dC.t_mean[x-1])  
        else:
            Tsu = Ta + 3.0
    if abs(Tsh-Ta) > 10 or np.isnan(Tsh):
        if x > 0:
            Tsh = Ta + (dC.t_mean[x]-dC.t_mean[x-1])  
        else:
            Tsh = Ta + 3.0
            
    T_Pars = [round(Ta,2), round(Tcu,2), round(Tch,2), round(Tsu,2), round(Tsh,2)] 

    LST = rtm_t(lai, L, i0, iD, wl, Tcu,Tch,Tsu,Tsh, dC, x, k_pars)

    [lEcu,Hcu,ecu,Ccu,lambdau,su] = calc_veg_fluxes(rac,cu_rcw,fwet,Tcu,Ta,cu_Ci,ea,c_p,rho)
    [lEch,Hch,ech,Cch,lambdah,sh] = calc_veg_fluxes(rac,ch_rcw,fwet,Tch,Ta,ch_Ci,ea,c_p,rho)
    [lEsu,Hsu,lambdasu,ssu] = calc_soil_fluxes(ras,rss,Tsu,Ta,ea,c_p,rho)
    [lEsh,Hsh,lambdash,ssh] = calc_soil_fluxes(ras,rss,Tsh,Ta,ea,c_p,rho)
      
    Evu = rho*(fwet/rac)*((calc_vapor_pressure(Tcu)-ea)*e_to_q)
    Evh = rho*(fwet/rac)*((calc_vapor_pressure(Tch)-ea)*e_to_q)
    Ev = lai*(Fc*Evu + (1-Fc)*Evh)
    if np.isnan(Ev):
        Ev = 0

    ETv = lai*(Fc*lEcu/lambdau + (1-Fc)*lEch/lambdah)
    ETs = Fs*lEsu/lambdasu + (1-Fs)*lEsh/lambdash 
    ET = ETv + ETs
     
    return Ccu, Cch, Tcu, Tch, Tsu, Tsh, ecu, ech, APARu, APARh, PAR, Esolars, LST, Fc, Ev, ET, Rns
