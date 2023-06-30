# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 11:39:26 2022

@author: hliu
"""
from math import exp, pi, sqrt, log
from photo_pars import *
from scipy import optimize
import numpy as np

def PhotoSynth(meteo):
    Q     = meteo[0] # [umol m-2 s-1] absorbed PAR flux
    Cs    = meteo[1] * ppm2bar
    T     = meteo[2] + T2K
    eb    = meteo[3] 

    # Calculate temp dependancies of Michaelis–Menten constants for CO2, O2
    Km = calc_michaelis_menten_constants(T) 
    
    # Effect of temp on CO2 compensation point
    Gamma_star = arrh(Gamma_star25, Eag, T)

    # Calculations at 25 degrees C or the measurement temperature
    Rd = calc_resp(Rd25, Ear, T)
    
    # Calculate temperature dependancies on Vcmax and Jmax
    Vcmax = peaked_arrh(Vcmax25, Eav, T, deltaSv, Hdv)
    Jmax  = peaked_arrh(Jmax25,  Eaj, T, deltaSj, Hdj)

    # actual rate of electron transport, a function of absorbed PAR
    if Q is not None:
        Je = sel_root(theta_J, -(alpha*Q+Jmax), alpha*Q*Jmax, -1)
    # All measurements are calculated under saturated light!!
    else:
        Je = Jmax
            
    RH = min(1, eb/esat(T))  

    A_Pars  = [Km, Rd, Vcmax, Gamma_star, Je]  
    Ci_Pars = [Cs, RH, minCi, BallBerrySlope, BallBerry0, ppm2bar]  
    
    Ci = optimize.brentq(opt_Ci, -1, 1, (A_Pars, Ci_Pars), 1E-6)
    
    A, Ag = Compute_A(Ci, Km, Rd, Vcmax, Je, Gamma_star)
    gs  = max(0.01, 1.6*A*ppm2bar/(Cs-Ci)) # stomatal conductance
    rcw = (Rhoa/(Mair*1E-3))/gs     # stomatal resistance

    return rcw, Ci/ppm2bar, A

def opt_Ci(x0, A_Pars, Ci_Pars):
    [Km, Rd, Vcmax, Gamma_star, Je] = A_Pars
    A, _ = Compute_A(x0, Km, Rd, Vcmax, Je, Gamma_star)
    
    [Cs, RH, minCi, BallBerrySlope, BallBerry0, ppm2bar] = Ci_Pars
    x1 = BallBerry(Cs, RH, A*ppm2bar, BallBerrySlope, BallBerry0, minCi)
    
    return x0-x1 

def Compute_A(Ci, Km, Rd, Vcmax, Je, Gamma_star):
    """
    Parameters
    ----------
    theta_hyperbol : float
        Curvature of the light response.
        See Peltoniemi et al. 2012 Tree Phys, 32, 510-519
    """
    theta_hyperbol = 0.995    
    
    # Rubisco carboxylation limited rate of photosynthesis
    Ac = Vcmax*(Ci-Gamma_star)/(Km + Ci)   
    # Light-limited rate of photosynthesis allowed by RuBP regeneration
    Aj = Je/5.0*((Ci-Gamma_star)/(2*Gamma_star + Ci))

    An = sel_root(theta_hyperbol, -(Ac+Aj), Ac*Aj, np.sign(-Ac)) 

    A  = An - Rd 
    
    return [A, An] 

def BallBerry(Cs, RH, A, BallBerrySlope, BallBerry0, minCi):
    if BallBerry0 == 0:
        Ci = max(minCi*Cs, Cs*(1-1.6/(BallBerrySlope*RH)))
        
    else:
        gs = max(BallBerry0,  BallBerrySlope*A*RH/(Cs+1E-9) + BallBerry0)
        Ci = max(minCi*Cs, Cs-1.6*A/gs) 
        
    return Ci

def calc_resp(Rd25, Ear, T):
    """ Calculate leaf respiration accounting for temperature dependence.

    Parameters:
    ----------
    Rd25 : float
        Estimate of respiration rate at the reference temperature 25 deg C
        or or 298 K
    Tref : float
        reference temperature
    Q10 : float
        ratio of respiration at a given temperature divided by respiration
        at a temperature 10 degrees lower
    Ear : float
        activation energy for the parameter [J mol-1]
    Returns:
    -------
    Rt : float
        leaf respiration

    References:
    -----------
    Tjoelker et al (2001) GCB, 7, 223-230.
    """
    Rd = arrh(Rd25, Ear, T)

    return Rd
  
def esat(T):
    A_SAT = 613.75
    B_SAT = 17.502
    C_SAT = 240.97
    
    """Saturated vapor pressure (hPa)"""
    return A_SAT*exp((B_SAT*(T - 273.))/(C_SAT + T - 273.))/100.0

def sel_root(a, b, c, dsign):
    """    
    quadratic formula, root of least magnitude
    """
    #  sel_root - select a root based on the fourth arg (dsign = discriminant sign)
    #    for the eqn ax^2 + bx + c,
    #    if dsign is:
    #       -1, 0: choose the smaller root
    #       +1: choose the larger root
    #  NOTE: technically, we should check a, but in biochemical, a is always > 0, dsign is always not equal to 0
    if a == 0:  # note: this works because 'a' is a scalar parameter!
        x = -c/b
    else:
        x = (-b + dsign* np.sqrt(b**2 - 4*a*c))/(2*a)
    
    return x      

def calc_michaelis_menten_constants(Tleaf):
    """ Michaelis-Menten constant for O2/CO2, Arrhenius temp dependancy
    Parameters:
    ----------
    Tleaf : float
        leaf temperature [deg K]

    Returns:
    Km : float

    """
    Kc = arrh(Kc25, Ec, Tleaf)
    Ko = arrh(Ko25, Eo, Tleaf)

    Km = Kc * (1.0 + O_c3 / Ko)

    return Km

def arrh(k25, Ea, Tk):
    """ Temperature dependence of kinetic parameters is described by an
    Arrhenius function.

    Parameters:
    ----------
    k25 : float
        rate parameter value at 25 degC or 298 K
    Ea : float
        activation energy for the parameter [J mol-1]
    Tk : float
        leaf temperature [deg K]

    Returns:
    -------
    kt : float
        temperature dependence on parameter

    References:
    -----------
    * Medlyn et al. 2002, PCE, 25, 1167-1179.
    """
    return k25 * np.exp((Ea * (Tk - 298.15)) / (298.15 * RGAS * Tk))

def peaked_arrh(k25, Ea, Tk, deltaS, Hd):
    """ Temperature dependancy approximated by peaked Arrhenius eqn,
    accounting for the rate of inhibition at higher temperatures.

    Parameters:
    ----------
    k25 : float
        rate parameter value at 25 degC or 298 K
    Ea : float
        activation energy for the parameter [J mol-1]
    Tk : float
        leaf temperature [deg K]
    deltaS : float
        entropy factor [J mol-1 K-1)
    Hd : float
        describes rate of decrease about the optimum temp [J mol-1]

    Returns:
    -------
    kt : float
        temperature dependence on parameter

    References:
    -----------
    * Medlyn et al. 2002, PCE, 25, 1167-1179.

    """
    arg1 = arrh(k25, Ea, Tk)
    arg2 = 1.0 + np.exp((298.15 * deltaS - Hd) / (298.15 * RGAS))
    arg3 = 1.0 + np.exp((Tk * deltaS - Hd) / (Tk * RGAS))

    return arg1 * arg2 / arg3


