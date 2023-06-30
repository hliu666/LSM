# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 11:15:12 2022

@author: Haoran 

Energy Balance Model 
"""
import numpy as np
from resistances import resistances 
from PhotoSynth import PhotoSynth
from TIR import rtm_t, calc_netrad
from TIR import calc_ebal_sunsha, calc_ebal_canopy_pars, calc_netrad_pars
from TIR import CalcStephanBoltzmann, Planck

import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# List of constants used in Meteorological computations
# ==============================================================================
# Stephan Boltzmann constant (W m-2 K-4)
sb = 5.670373e-8
# heat capacity of dry air at constant pressure (J kg-1 K-1)
c_pd = 1003.5
# heat capacity of water vapour at constant pressure (J kg-1 K-1)
c_pv = 1865
# ratio of the molecular weight of water vapor to dry air
epsilon = 0.622
# Psicrometric Constant kPa K-1
psicr = 0.0658
# gas constant for dry air, J/(kg*degK)
R_d = 287.04
# acceleration of gravity (m s-2)
g = 9.8

# von Karman's constant
KARMAN = 0.41
# acceleration of gravity (m s-2)
GRAVITY = 9.8

# functions for saturated vapour pressure 
def es_fun(T):
    return 6.107*10**(7.5*T/(237.3+T))

def s_fun(es, T):
    return es*2.3026*7.5*237.3/((237.3+T)**2)

def calc_vapor_pressure(T_K):
    """Calculate the saturation water vapour pressure.
    Parameters
    ----------
    T_K : float
        temperature (K).
    Returns
    -------
    ea : float
        saturation water vapour pressure (mb).
    """

    T_C = T_K - 273.15
    ea = 6.112 * np.exp((17.67 * T_C) / (T_C + 243.5))
    return np.asarray(ea)

def calc_lambda(T_A_K):
    '''Calculates the latent heat of vaporization.

    Parameters
    ----------
    T_A_K : float
        Air temperature (Kelvin).

    Returns
    -------
    Lambda : float
        Latent heat of vaporisation (J kg-1).

    References
    ----------
    based on Eq. 3-1 Allen FAO98 '''
    Lambda = 1E6 * (2.501 - (2.361e-3 * (T_A_K - 273.15)))
    return np.asarray(Lambda)

def Monin_Obukhov(ustar, Ta, H):
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

    cp = 1004
    rhoa = 1.2047
    kappa = 0.4
    g = 9.81
    L = -rhoa*cp*ustar**3*(Ta+273.15)/(kappa*g*H)
    return L

def calc_emiss_atm(ea, t_a_k):
    '''Atmospheric emissivity
    Estimates the effective atmospheric emissivity for clear sky.
    Parameters
    ----------
    ea : float
        atmospheric vapour pressure (mb).
    t_a_k : float
        air temperature (Kelvin).
    Returns
    -------
    emiss_air : float
        effective atmospheric emissivity.
    References
    ----------
    .. [Brutsaert1975] Brutsaert, W. (1975) On a derivable formula for long-wave radiation
        from clear skies, Water Resour. Res., 11(5), 742-744,
        htpp://dx.doi.org/10.1029/WR011i005p00742.'''

    emiss_air = 1.24 * (ea / t_a_k)**(1. / 7.)  # Eq. 11 in [Brutsaert1975]_

    return emiss_air

def calc_mixing_ratio(ea, p):
    '''Calculate ratio of mass of water vapour to the mass of dry air (-)
    Parameters
    ----------
    ea : float or numpy array
        water vapor pressure at reference height (mb).
    p : float or numpy array
        total air pressure (dry air + water vapour) at reference height (mb).
    Returns
    -------
    r : float or numpy array
        mixing ratio (-)
    References
    ----------
    http://glossary.ametsoc.org/wiki/Mixing_ratio'''

    r = epsilon * ea / (p - ea)
    return r

def calc_c_p(p, ea):
    ''' Calculates the heat capacity of air at constant pressure.
    Parameters
    ----------
    p : float
        total air pressure (dry air + water vapour) (mb).
    ea : float
        water vapor pressure at reference height above canopy (mb).
    Returns
    -------
    c_p : heat capacity of (moist) air at constant pressure (J kg-1 K-1).
    References
    ----------
    based on equation (6.1) from Maarten Ambaum (2010):
    Thermal Physics of the Atmosphere (pp 109).'''

    # first calculate specific humidity, rearanged eq (5.22) from Maarten
    # Ambaum (2010), (pp 100)
    q = epsilon * ea / (p + (epsilon - 1.0) * ea)
    # then the heat capacity of (moist) air
    c_p = (1.0 - q) * c_pd + q * c_pv
    return np.asarray(c_p)

def calc_lapse_rate_moist(T_A_K, ea, p):
    '''Calculate moist-adiabatic lapse rate (K/m)
    Parameters
    ----------
    T_A_K : float or numpy array
        air temperature at reference height (K).
    ea : float or numpy array
        water vapor pressure at reference height (mb).
    p : float or numpy array
        total air pressure (dry air + water vapour) at reference height (mb).
    Returns
    -------
    Gamma_w : float or numpy array
        moist-adiabatic lapse rate (K/m)
    References
    ----------
    http://glossary.ametsoc.org/wiki/Saturation-adiabatic_lapse_rate'''

    r = calc_mixing_ratio(ea, p)
    c_p = calc_c_p(p, ea)
    lambda_v = calc_lambda(T_A_K)
    Gamma_w = ((g * (R_d * T_A_K**2 + lambda_v * r * T_A_K)
               / (c_p * R_d * T_A_K**2 + lambda_v**2 * r * epsilon)))
    return Gamma_w

def calc_longwave_irradiance(ea, t_a_k, p=1013.25, z_T=2.0, h_C=2.0):
    '''Longwave irradiance
    Estimates longwave atmospheric irradiance from clear sky.
    By default there is no lapse rate correction unless air temperature
    measurement height is considerably different than canopy height, (e.g. when
    using NWP gridded meteo data at blending height)
    Parameters
    ----------
    ea : float
        atmospheric vapour pressure (mb).
    t_a_k : float
        air temperature (K).
    p : float
        air pressure (mb)
    z_T: float
        air temperature measurement height (m), default 2 m.
    h_C: float
        canopy height (m), default 2 m,
    Returns
    -------
    L_dn : float
        Longwave atmospheric irradiance (W m-2) above the canopy
    '''

    lapse_rate = calc_lapse_rate_moist(t_a_k, ea, p)
    t_a_surface = t_a_k - lapse_rate * (h_C - z_T)
    emisAtm = calc_emiss_atm(ea, t_a_surface)
    L_dn = emisAtm * CalcStephanBoltzmann(t_a_surface)
    return L_dn

def heatfluxes(ra,rs,Tc,ea,Ta,e_to_q,Ca,Ci): 
    """
    # this function calculates latent and sensible heat flux
    #
    # input:
    #   ra          aerodynamic resistance for heat         s m-1
    #   rs          stomatal resistance                     s m-1
    #   Tc          leaf temperature                        oC
    #   ea          vapour pressure above canopy            hPa
    #   Ta          air temperature above canopy            oC
    #   e_to_q      conv. from vapour pressure to abs hum   hPa-1
    #   PSI         leaf water potential                    J kg-1
    #   Ca          ambient CO2 concentration               umol m-3
    #   Ci          intercellular CO2 concentration         umol m-3
    #   constants   a structure with physical constants
    #   es_fun      saturated pressure function es(hPa)=f(T(C))
    #   s_fun       slope of the saturated pressure function (s(hPa/C) = f(T(C), es(hPa))
    #
    # output:
    #   lEc         latent heat flux of a leaf              W m-2
    #   Hc          sensible heat flux of a leaf            W m-2
    #   ec          vapour pressure at the leaf surface     hPa
    #   Cc          CO2 concentration at the leaf surface   umol m-3
    """
    rhoa = 1.2047
    cp   = 1004
    
    #Lambda = (2.501-0.002361*Tc)*1E6    # [J kg-1]  Evapor. heat (J kg-1)
    Lambda = calc_lambda(Tc+273.15)   # [J kg-1]  Evapor. heat (J kg-1)
    ei = es_fun(Tc)
    s  = s_fun(ei, Tc)

    qi = ei * e_to_q
    qa = ea * e_to_q
    
    lE = rhoa/(ra+rs)*Lambda*(qi-qa)    # [W m-2]   Latent heat flux
    H  = (rhoa*cp)/ra*(Tc-Ta)           # [W m-2]   Sensible heat flux
    ec = ea + (ei-ea)*ra/(ra+rs)        # [W m-2] vapour pressure at the leaf surface
    Cc = Ca - (Ca-Ci)*ra/(ra+rs)        # [umol m-2 s-1] CO2 concentration at the leaf surface
    
    return [lE, H, ec, Cc, Lambda, s]

def Ebal(dC, x, lai, ebal_rtm_pars, k_pars):    
    
    """
    # 1. initialisations and other preparations for the iteration loop
    # parameters for the closure loop
    """
    counter     = 0                # iteration counter of ebal
    maxit       = 50               # maximum number of iterations
    maxEBer     = 0.1              # maximum energy balance error (any leaf) [Wm-2]
    Wc          = 1                # update step (1 is nominal, [0,1] possible)
    CONT        = 50               # boolean indicating whether iteration continues
    
    # constants
    MH2O        = 18          # [g mol-1]     Molecular mass of water
    Mair        = 28.96       # [g mol-1]     Molecular mass of dry air
    rhoa        = 1.2047      # [kg m-3]      Specific mass of air
    cp          = 1004        # [J kg-1 K-1]  Specific heat of dry air
    sigmaSB     = 5.67E-8     # [W m-2 K-4]   Stefan Boltzman constant  
    
    # input preparation
    rss         = 500.0       # soil resistance for evaporation from the pore space
    
    # meteo
    Ta          = dC.t_mean[x]
    ea          = dC.ea
    #ea         = 4.6491*np.exp(0.062*ta)# atmospheric vapour pressure
    Ca          = dC.ca                   # atmospheric CO2 concentration
    p           = dC.p                    # air pressure
    o           = dC.o                    # atmospheric O2 concentration
    ech         = ea                      # Leaf boundary vapour pressure (shaded/sunlit leaves)
    Cch         = Ca
    ecu         = ea 
    Ccu         = Ca                      # Leaf boundary CO2 (shaded/sunlit leaves)   
    
    # other preparations
    e_to_q      = MH2O/Mair/p             # Conversion of vapour pressure [Pa] to absolute humidity [kg kg-1]

    # initial values for the loop
    Tsu         = (Ta + 3.0)                 # soil temperature (+3 for a head start of the iteration) 
    Tsh         = (Ta + 3.0)                 # soil temperature (+3 for a head start of the iteration) 
    Tcu         = (Ta + 0.3)                 # leaf tempeFrature (sunlit leaves)
    Tch         = (Ta + 0.1)                 # leaf temperature (shaded leaves)    
    l_mo        = -1E6                       # Monin-Obukhov length

    T_Pars = [Ta, Tcu, Tch, Tsu, Tsh]    
    
    L  = calc_longwave_irradiance(ea, Ta+273.15)
    SW = dC.sw[x]
    """
    ## 2.1 Energy balance iteration loop
    #Energy balance loop (Energy balance and radiative transfer)
    """
    wl = dC.wl
    Ls = Planck(wl, Ta+273.15)
    ebal_sunsha_pars  = calc_ebal_sunsha(dC, x, lai)
    ebal_canopy_pars  = calc_ebal_canopy_pars(dC, x, lai, Ls, ebal_rtm_pars)    
    net_rads, Esolars = calc_netrad_pars(dC, x, lai, SW, L, ebal_sunsha_pars, ebal_canopy_pars)

    while CONT:                          # while energy balance does not close
        rad_Rnuc, rad_Rnhc, APARu, APARh, rad_Rnus, rad_Rnhs, Fc, Fs, i0, iD = calc_netrad(dC, x, lai, L, T_Pars, net_rads, ebal_rtm_pars, ebal_sunsha_pars)
        if (dC.tts[x] < 75) and (lai > 0.5): 
            APARu = max(APARu, 0.0) 
            APARh = max(APARh, 0.0) 

            meteo_u = [APARu/lai, Ccu, Tcu, ecu, o, p]
            meteo_h = [APARh/lai, Cch, Tch, ech, o, p]
            
            # Fluxes (latent heat flux (lE), sensible heat flux (H) and soil heat flux G
            # in analogy to Ohm's law, for canopy (c) and soil (s). All in units of [W m-2]
            bcu_rcw, bcu_Ci, bcu_An = PhotoSynth(meteo_u, [dC.Vcmax25, dC.BallBerrySlope])
            bch_rcw, bch_Ci, bch_An = PhotoSynth(meteo_h, [dC.Vcmax25, dC.BallBerrySlope])
        else:
            bcu_rcw, bcu_Ci, bcu_An = 4160, Ccu, 0.0
            bch_rcw, bch_Ci, bch_An = 4160, Cch, 0.0
            
        # Aerodynamic roughness
        # calculate friction velocity [m s-1] and aerodynamic resistances [s m-1]  
        raa, rawc, raws, ustar = resistances(lai, l_mo, dC.wds[x])
        rac     = (lai+1)*(raa+rawc)
        ras     = (lai+1)*(raa+raws)

        [lEcu,Hcu,ecu,Ccu,lambdau,su]     = heatfluxes(rac,bcu_rcw,Tcu,ea,Ta,e_to_q,Ca,bcu_Ci)
        [lEch,Hch,ech,Cch,lambdah,sh]     = heatfluxes(rac,bch_rcw,Tch,ea,Ta,e_to_q,Ca,bch_Ci)
        [lEsu,Hsu,_,_,lambdasu,ssu]       = heatfluxes(ras,rss,    Tsu,ea,Ta,e_to_q,Ca,Ca)
        [lEsh,Hsh,_,_,lambdash,ssh]       = heatfluxes(ras,rss,    Tsh,ea,Ta,e_to_q,Ca,Ca)
        
        # integration over the layers and sunlit and shaded fractions
        Hstot   = Fs*Hsu + (1-Fs)*Hsh
        Hctot   = Fc*Hcu + (1-Fc)*Hch
        Htot    = Hstot + Hctot*lai

        lEstot   = Fs*lEsu + (1-Fs)*lEsh
        lEctot   = Fc*lEcu + (1-Fc)*lEch
        lEtot    = lEstot + lEctot*lai
            
        #rho = 1.2047
        #T_A_K = (Ta + 273.15)
        #H  = Htot
        #LE = lEtot
        #c_p = calc_c_p(1013.25, ea)
        #l_mo = Monin_Obukhov(ustar, T_A_K, rho, c_p, H, LE)

        l_mo = Monin_Obukhov(ustar, Ta, Htot)
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
        Tch         = Tch + Wc*EBerch/((rhoa*cp)/rac + rhoa*lambdah *e_to_q*sh/(rac+bch_rcw)+ 4*leafbio_emis       *sigmaSB*(Tch+273.15)**3)
        Tcu         = Tcu + Wc*EBercu/((rhoa*cp)/rac + rhoa*lambdau *e_to_q*su/(rac+bcu_rcw)+ 4*leafbio_emis       *sigmaSB*(Tcu+273.15)**3)
        Tsh         = Tsh + Wc*EBersh/((rhoa*cp)/ras + rhoa*lambdash*e_to_q*ssh/(ras+rss)   + 4*(1-soil_rs_thermal)*sigmaSB*(Tsh+273.15)**3 + dGh)
        Tsu         = Tsu + Wc*EBersu/((rhoa*cp)/ras + rhoa*lambdasu*e_to_q*ssu/(ras+rss)   + 4*(1-soil_rs_thermal)*sigmaSB*(Tsu+273.15)**3 + dGu)

        if abs(Tch) > 100:
            Tch = Ta
        if abs(Tcu) > 100:
            Tcu = Ta
        
        T_Pars = [round(Ta,2), round(Tcu,2), round(Tch,2), round(Tsu,2), round(Tsh,2)] 
        
    LST, emis = rtm_t(lai, L, i0, iD, wl, Tcu,Tch,Tsu,Tsh, dC, x, k_pars)
    
    Cc = Fc*Ccu + (1-Fc)*Cch
    T  = Fc*Tcu + (1-Fc)*Tch
    ec = Fc*ecu + (1-Fc)*ech

    #return Cc, T, ec, Esolars, LST, Fc
    return Ccu, Cch, Tcu, Tch, ecu, ech, APARu, APARh, Esolars, LST, Fc

def Ebal_single(dC, x, lai, ebal_rtm_pars, k_pars):
    """
    # 1. initialisations and other preparations for the iteration loop
    # parameters for the closure loop
    """
    counter     = 0                # iteration counter of ebal
    maxit       = 50               # maximum number of iterations
    maxEBer     = 0.1              # maximum energy balance error (any leaf) [Wm-2]
    Wc          = 1                # update step (1 is nominal, [0,1] possible)
    CONT        = 50               # boolean indicating whether iteration continues
    
    # constants
    MH2O        = 18          # [g mol-1]     Molecular mass of water
    Mair        = 28.96       # [g mol-1]     Molecular mass of dry air
    rhoa        = 1.2047      # [kg m-3]      Specific mass of air
    cp          = 1004        # [J kg-1 K-1]  Specific heat of dry air
    sigmaSB     = 5.67E-8     # [W m-2 K-4]   Stefan Boltzman constant  
    
    # input preparation
    rss         = 500.0       # soil resistance for evaporation from the pore space
    
    # meteo
    Ta          = dC.t_mean[x]
    ea          = dC.ea
    #ea         = 4.6491*np.exp(0.062*ta)# atmospheric vapour pressure
    Ca          = dC.ca                   # atmospheric CO2 concentration
    p           = dC.p                    # air pressure
    o           = dC.o                    # atmospheric O2 concentration
    ech         = ea                      # Leaf boundary vapour pressure (shaded/sunlit leaves)
    Cch         = Ca
    ecu         = ea 
    Ccu         = Ca                      # Leaf boundary CO2 (shaded/sunlit leaves)   
      
    # other preparations
    e_to_q      = MH2O/Mair/p             # Conversion of vapour pressure [Pa] to absolute humidity [kg kg-1]

    # initial values for the loop
    Tsu         = (Ta + 3.0)                 # soil temperature (+3 for a head start of the iteration) 
    Tsh         = (Ta + 3.0)                 # soil temperature (+3 for a head start of the iteration) 
    Tcu         = (Ta + 0.3)                 # leaf tempeFrature (sunlit leaves)
    Tch         = (Ta + 0.1)                 # leaf temperature (shaded leaves)    
    l_mo        = -1E6                       # Monin-Obukhov length

    T_Pars = [Ta, Tcu, Tch, Tsu, Tsh]    
    
    L = calc_longwave_irradiance(ea, Ta+273.15)
    SW = dC.sw[x]
    """
    ## 2.1 Energy balance iteration loop
    #Energy balance loop (Energy balance and radiative transfer)
    """
    wl = dC.wl
    Ls = Planck(wl, Ta+273.15)
    ebal_sunsha_pars  = calc_ebal_sunsha(dC, x, lai)
    ebal_canopy_pars  = calc_ebal_canopy_pars(dC, x, lai, Ls, ebal_rtm_pars)    
    net_rads, Esolars = calc_netrad_pars(dC, x, lai, SW, L, ebal_sunsha_pars, ebal_canopy_pars)
    
    rad_Rnuc, rad_Rnhc, APARu, APARh, rad_Rnus, rad_Rnhs, Fc, Fs, i0, iD = calc_netrad(dC, x, lai, L, T_Pars, net_rads, ebal_rtm_pars, ebal_sunsha_pars)
    if (dC.tts[x] < 75) and (lai > 0.5): 
        APARu = max(APARu, 0.0) 
        APARh = max(APARh, 0.0) 

        meteo_u = [APARu/lai, Ccu, Tcu, ecu, o, p]
        meteo_h = [APARh/lai, Cch, Tch, ech, o, p]
        
        # Fluxes (latent heat flux (lE), sensible heat flux (H) and soil heat flux G
        # in analogy to Ohm's law, for canopy (c) and soil (s). All in units of [W m-2]
        bcu_rcw, bcu_Ci, bcu_An = PhotoSynth(meteo_u, [dC.Vcmax25, dC.BallBerrySlope])
        bch_rcw, bch_Ci, bch_An = PhotoSynth(meteo_h, [dC.Vcmax25, dC.BallBerrySlope])
    else:
        bcu_rcw, bcu_Ci, bcu_An = 4160, Ccu, 0.0
        bch_rcw, bch_Ci, bch_An = 4160, Cch, 0.0
 
    # Aerodynamic roughness
    # calculate friction velocity [m s-1] and aerodynamic resistances [s m-1]  
    raa, rawc, raws, ustar = resistances(lai, l_mo, dC.wds[x])
    rac     = (lai+1)*(raa+rawc)
    ras     = (lai+1)*(raa+raws)

    [lEcu,Hcu,ecu,Ccu,lambdau,su]     = heatfluxes(rac,bcu_rcw,Tcu,ea,Ta,e_to_q,Ca,bcu_Ci)
    [lEch,Hch,ech,Cch,lambdah,sh]     = heatfluxes(rac,bch_rcw,Tch,ea,Ta,e_to_q,Ca,bch_Ci)
    [lEsu,Hsu,_,_,lambdasu,ssu]       = heatfluxes(ras,rss,    Tsu,ea,Ta,e_to_q,Ca,Ca)
    [lEsh,Hsh,_,_,lambdash,ssh]       = heatfluxes(ras,rss,    Tsh,ea,Ta,e_to_q,Ca,Ca)
    
    # integration over the layers and sunlit and shaded fractions
    Hstot   = Fs*Hsu + (1-Fs)*Hsh
    Hctot   = Fc*Hcu + (1-Fc)*Hch
    Htot    = Hstot + Hctot*lai

    lEstot   = Fs*lEsu + (1-Fs)*lEsh
    lEctot   = Fc*lEcu + (1-Fc)*lEch
    lEtot    = lEstot + lEctot*lai
        
    #rho = 1.2047
    #T_A_K = (Ta + 273.15)
    #H  = Htot
    #LE = lEtot
    #c_p = calc_c_p(1013.25, ea)
    #l_mo = Monin_Obukhov(ustar, T_A_K, rho, c_p, H, LE)

    l_mo = Monin_Obukhov(ustar, Ta, Htot)
    # ground heat flux
    soil_rs_thermal = 0.06 #broadband soil reflectance in the thermal range 1 - emissivity
    
    Gu  = 0.35*rad_Rnus
    Gh  = 0.35*rad_Rnhs
    
    dGu = 4*(1-soil_rs_thermal)*sigmaSB*(Tsu+273.15)**3*0.35
    dGh = 4*(1-soil_rs_thermal)*sigmaSB*(Tsh+273.15)**3*0.35
      
    # energy balance errors, continue criterion and iteration counter
    EBercu  = rad_Rnuc/(lai*Fc)      -lEcu -Hcu
    EBerch  = rad_Rnhc/(lai*(1-Fc))  -lEch -Hch
    EBersu  = rad_Rnus -lEsu -Hsu - Gu
    EBersh  = rad_Rnhs -lEsh -Hsh - Gh
      
    #l.append([rad_Rnuc, rad_Rnhc, ERnuc, ERnhc, ELnuc, ELnhc, lEcu, lEch, Hcu, Hch, Tcu, Tch])
    # if counter>99, plot(EBercu(:)), hold on, end
    # 2.7. New estimates of soil (s) and leaf (c) temperatures, shaded (h) and sunlit (1)
    leafbio_emis = 0.98
    Tch         = Tch + Wc*EBerch/((rhoa*cp)/rac + rhoa*lambdah *e_to_q*sh/(rac+bch_rcw)+ 4*leafbio_emis       *sigmaSB*(Tch+273.15)**3)
    Tcu         = Tcu + Wc*EBercu/((rhoa*cp)/rac + rhoa*lambdau *e_to_q*su/(rac+bcu_rcw)+ 4*leafbio_emis       *sigmaSB*(Tcu+273.15)**3)
    Tsh         = Tsh + Wc*EBersh/((rhoa*cp)/ras + rhoa*lambdash*e_to_q*ssh/(ras+rss)   + 4*(1-soil_rs_thermal)*sigmaSB*(Tsh+273.15)**3 + dGh)
    Tsu         = Tsu + Wc*EBersu/((rhoa*cp)/ras + rhoa*lambdasu*e_to_q*ssu/(ras+rss)   + 4*(1-soil_rs_thermal)*sigmaSB*(Tsu+273.15)**3 + dGu)

    if abs(Tch) > 100:
        Tch = Ta
    if abs(Tcu) > 100:
        Tcu = Ta
    
    T_Pars = [round(Ta,2), round(Tcu,2), round(Tch,2), round(Tsu,2), round(Tsh,2)] 
    
    LST, emis = rtm_t(lai, L, i0, iD, wl, Tcu,Tch,Tsu,Tsh, dC, x, k_pars)
        
    Cc = Fc*Ccu + (1-Fc)*Cch
    T  = Fc*Tcu + (1-Fc)*Tch
    ec = Fc*ecu + (1-Fc)*ech

    #return Cc, T, ec, Esolars, LST, Fc
    return Ccu, Cch, Tcu, Tch, ecu, ech, APARu, APARh, Esolars, LST, Fc
