# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 11:39:26 2022

@author: hliu
"""
from math import exp, pi, sqrt, log
from photo_pars import *
from scipy import optimize
import numpy as np
from sympy import Symbol, nsolve

def PhotoSynth_Jen(meteo):
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
    
    """
    0. Initial values of parameters and constants
    """
    RUB = 30 # [umol sites m-2] Rubisco density
    kc = 1.0 #[umol CO2 umol sites-1 s-1] Rubisco kcat for CO2
    Rdsc = 0.01 # [] Scalar for mitochondrial (dark) respiration
    
    R = 8.314/1E6 # [J umol-1 K-1] Ideal gas constant
    Tref_K = 25 + T2K # [K] Reference temperature
    
    """
    1. Mitochondrial respiration
    """
    Rd = RUB*kc*Rdsc # [umol CO2 m-2 s-1] Respiration
    Ha = 66000/1E6 # [J umol-1] Activation energy
    Rd = Rd*exp(Ha/R*(1/Tref_K - 1/T)) # [mol CO2 m-2 s-1] Resp.

    # Calculations at 25 degrees C or the measurement temperature
    Rd = calc_resp(Rd25, Ear, T)  
    
    """
    2. Rubisco
    """
    Ha = 58000/1E6  # [J umol-1] Activation energy
    kc = kc*exp(Ha/R*(1/Tref_K - 1/T)) # [s-1] Rubisco kcat for CO2 
    # Vcmax = RUB*kc              # [umol CO2 m-2 s-1] Max Rubisco activity 

    # Calculate temperature dependancies on Vcmax and Jmax
    Vcmax = peaked_arrh(RUB, Eav, T, deltaSv, Hdv)

    #print(Vcmax,Vcmax_1)
    """
    3. define the conductance
    """
    gtc = 0.01 # [] total (stomatal and boundary layer) conductance to CO2
    
    """
    4. Cytochrome b6f complex
    """
    kq = 35 # [umol e-1 umol sites-1 s-1] Cyt b6f kcat for PQH2
    Ha = 37000/1E6 # [J umol-1] Activation energy 
    CB6F = 350/300 #[umol sites m-2] Cyt b6f density

    kq = kq*exp(Ha/R*(1/Tref_K - 1/T))# Cyt b6f kcat for PQH2, s-1
    Vqmax = CB6F*kq # Max Cyt b6f activity, mol e- m-2 s-1
 
    """
    5. Cytochrome b6f-limited rates
    """
    Kp1 = 14.5E9 #[s-1] Rate constant for photochemistry at PSI
    Kf = 0.05E9  #[s-1] Rate constant for fluoresence at PSII and PSI
    Kd = 0.55E9  #[s-1] Rate constant for constitutive heat loss at PSII and PSI

    Abs = 0.85 # [umol absorbed umol-1 incident] Total leaf absorptance to PAR
    a2_frac = 0.52 # [] PS II fraction of mesophyll absorptance to PAR
    a2 = Abs*(a2_frac)  # [] PSII, mol PPFD abs PS2 mol-1 PPFD incident
    a1 = Abs - a2       # [] PSI, mol PPFD abs PS1 mol-1 PPFD incident

    JP700_j = Q*Vqmax/(Q+Vqmax/(a1*(Kp1/(Kp1 + Kd + Kf))))#

    Jmax  = peaked_arrh(Jmax25,  Eaj, T, deltaSj, Hdj)

    nl = 0.75 # [ATP/e-] ATP per e- in linear flow
    nc = 1.00 # [ATP/e-] ATP per e- in cyclic flow

    RH = min(1, eb/esat(T))  

    A_Pars  = [Vcmax, Km, JP700_j, nl, nc, gtc, Gamma_star, Rd]  
    Ci_Pars = [Cs, RH, minCi, BallBerrySlope, BallBerry0, ppm2bar]  

    Ci = optimize.brentq(opt_Ci, -1, 1, (A_Pars, Ci_Pars), 1E-6)

    A, Ac, Aj = Compute_A(Ci, Vcmax, Km, JP700_j, nl, nc, gtc, Gamma_star, Rd)
    #print(A, Ac, Aj, Ci)
    gs  = max(0.01, 1.6*A*ppm2bar/(Cs-Ci)) # stomatal conductance
    rcw = (Rhoa/(Mair*1E-3))/gs     # stomatal resistance

    """
    6. fluorescence
    """
    # Assign values for photochemical constants
    Kn1 = 14.5E9 #[s-1] Rate constant for regulated heat loss at PSI    
    Kp2 = 4.5E9 #[s-1] Rate constant for photochemistry at PSII
    Ku2 = 0E9 #[s-1] Rate constant for exciton sharing at PSII

    C = Ci - A/gtc
    # Rubisco carboxylation limited rate of photosynthesis
    eta = (1-(nl/nc)+(3+7*Gamma_star/C)/((4+8*Gamma_star/C)*nc)) # PS I/II ETR

    JP680_c = Ac*4*(1+2*Gamma_star/C)/(1-Gamma_star/C) 
    JP700_c = JP680_c*eta

    JP680_j = JP700_j/eta

    theta_hyperbol = 1.0
    JP700_a = sel_root(theta_hyperbol, -(JP700_c+JP700_j), JP700_c*JP700_j, np.sign(-JP700_c)) 
    JP680_a = sel_root(theta_hyperbol, -(JP680_c+JP680_j), JP680_c*JP680_j, np.sign(-JP680_c)) 

    phi1P_max = Kp1/(Kp1 + Kd + Kf) # Maximum photochemical yield PS I
    
    CB6F_a = JP700_j/kq       # Eqns. 21, 30a, 34
    phi1P_a = JP700_a/(Q*a1)  # Eqn. 20
    q1_a = phi1P_a/phi1P_max  # Eqn. 19a
    phi2P_a = JP680_a/(Q*a2)  # Eqn. 26
    q2_a = 1 - CB6F_a/CB6F    # Eqns. 28 and 34

    # N.B., rearrange Eqn. 25a to solve for Kn2_a
    Kn2_a = ((Kp2**2.*phi2P_a**2 - 2*Kp2**2*phi2P_a*q2_a + \
       Kp2**2*q2_a**2 - 4*Kp2*Ku2*phi2P_a**2*q2_a + \
       2*Kp2*Ku2*phi2P_a**2 + 2*Kp2*Ku2*phi2P_a*q2_a + \
       Ku2**2*phi2P_a**2)**0.5 - Kp2*phi2P_a + Ku2*phi2P_a + \
       Kp2*q2_a)/(2*phi2P_a) - Kf - Ku2 - Kd

    # Photosystem II (Eqns. 23a-23e and 25a-25d)
    phi2p_a = (q2_a)*Kp2/(Kp2 + Kn2_a + Kd + Kf + Ku2)
    phi2f_a = (q2_a)*Kf/(Kp2 + Kn2_a + Kd + Kf + Ku2) + (1 - q2_a)*Kf/(Kn2_a + Kd + Kf + Ku2)
    phi2u_a = (q2_a)*Ku2/(Kp2 + Kn2_a + Kd + Kf + Ku2) + (1 - q2_a)*Ku2/(Kn2_a + Kd + Kf + Ku2)
    
    phi2P_a = phi2p_a/(1-phi2u_a)
    phi2F_a = phi2f_a/(1-phi2u_a)
    
    phi2F_a_light = Kf/(Kn2_a + Kd + Kf + Ku2)/(1 - phi2u_a) * (1-phi2P_a)

    phi2f_a_dark = Kf/(Kn2_a + Kd + Kf + Ku2)
    phi2u_a_dark = Ku2/(Kn2_a + Kd + Kf + Ku2)
    phi2F_a_dark = phi2f_a_dark/(1-phi2u_a_dark)
    
    # For Photosystem I (Eqns. 19a-19d)
    phi1P_a = (q1_a)*Kp1/(Kp1 + Kd + Kf)
    phi1F_a = (q1_a)*Kf/(Kp1 + Kd + Kf) + (1 - q1_a)*Kf/(Kn1 + Kd + Kf)

    phi1F_a_light = Kf/(Kn1 + Kd + Kf) * (1-phi1P_a)
    phi1F_a_dark = Kf/(Kn1 + Kd + Kf)

    eta2 = phi2F_a_light/phi2F_a_dark
    eta1 = phi1F_a_light/phi1F_a_dark
    
    return rcw, Ci/ppm2bar, A, eta2, eta1

from sympy import Symbol, nsolve

def opt_Ci(x0, A_Pars, Ci_Pars):
    [Vcmax, Km, JP700_j, nl, nc, gtc, Gamma_star, Rd]  = A_Pars
    A,_,_ = Compute_A(x0, Vcmax, Km, JP700_j, nl, nc, gtc, Gamma_star, Rd)

    [Cs, RH, minCi, BallBerrySlope, BallBerry0, ppm2bar] = Ci_Pars
    x1 = BallBerry(Cs, RH, A*ppm2bar, BallBerrySlope, BallBerry0, minCi)
    
    return x0-x1 

def opt_Cm_Ac(Ci, Vcmax, Gamma_star, Km, Rd):
    Cm = Symbol('Cm')
    
    Ac = Vcmax*(Cm - Gamma_star)/(Km + Cm) + Rd
    Ag = gtc*(Ci - Cm)
    
    return nsolve(Ac-Ag, Cm, 0.00039)

def opt_Cm_Aj(Ci, Je, nl, nc, Gamma_star, Rd):
    Cm = Symbol('Cm')
    
    Je = Je/(1-nl/nc+(3*Ci+7*Gamma_star)/(4*nc*(2*Gamma_star+Ci)))
    Aj = Je/5.0*((Ci-Gamma_star)/(2*Gamma_star + Ci)) + Rd
    Ag = gtc*(Ci - Cm)
    
    return nsolve(Aj-Ag, Cm, 0.00039) 

def Compute_A(Ci, Vcmax, Km, Je, nl, nc, gtc, Gamma_star, Rd):
    """
    Parameters
    ----------
    theta_hyperbol : float
        Curvature of the light response.
        See Peltoniemi et al. 2012 Tree Phys, 32, 510-519
    """
    theta_hyperbol = 0.995    

    # Rubisco carboxylation limited rate of photosynthesis
    #a_x2 = 1/gtc
    #b_x2 = -(Km+Ci+(Rd/gtc)-(Vcmax/gtc))
    #c_x2 = Rd*(Km+Ci)-Vcmax*Ci
    #Ac = sel_root(a_x2, b_x2, c_x2, -1)
    #Ac = optimize.brentq(quadratic, -10, 40, (a_x2, b_x2, c_x2), 1E-6)
    #Cm = optimize.brentq(opt_Cm_Ac, -1, 1, (Ci, Vcmax, Gamma_star, Km, Rd), 1E-6)
    Cm = opt_Cm_Ac(Ci, Vcmax, Gamma_star, Km, Rd)
    Ac = float(Vcmax*(Cm - Gamma_star)/(Km + Cm))
    
    # Light-limited rate of photosynthesis allowed by RuBP regeneration
    #a, b, c, d, e, f, g = Je, nc, nl, 1.0/gtc, Ci, Gamma_star, Rd
    #a_x2 = -(4*d*b + 4*d*c + 3*d)
    #b_x2 = 4*b*e + 4*c*e + 8*f*b + 8*f*c + 3*e + 7*f - 4*d*b*g - 4*d*c*g - 3*d*g + a*b*d
    #c_x2 = 4*b*e*g + 4*c*e*g + 8*f*b*g + 8*f*c*g + 3*e*g + 7*f*g - a*d*e + a*b*f
    #Aj = sel_root(a_x2, b_x2, c_x2, -1)
    #Aj = optimize.brentq(quadratic, -10, 40, (a_x2, b_x2, c_x2), 1E-6)
    #Cm = optimize.brentq(opt_Cm_Aj, -1, 1, (Ci, Je, nl, nc, Gamma_star, Rd), 1E-6)
    Cm = opt_Cm_Aj(Ci, Je, nl, nc, Gamma_star, Rd)
    Aj = float(Je/5.0*((Cm-Gamma_star)/(2*Gamma_star + Cm)))

    A = sel_root(theta_hyperbol, -(Ac+Aj), Ac*Aj, np.sign(-Ac)) 
    
    return A, Ac, Aj

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


from PhotoSynth import PhotoSynth

# Q [umol m-2 s-1] absorbed PAR flux
Cs    = 390 
eb    = 40 

#meteo = [1000, Cs, T, eb]
#r1, c1, a1 = PhotoSynth_Jen(meteo)    
#r2, c2, a2 = PhotoSynth(meteo)   

    
A1, A2 = [], []
for Q in range(0, 2500, 50):
   for T in range(0, 30, 3): 
        meteo = [Q, Cs, T, eb]
        r2, c2, p2 = PhotoSynth(meteo)   
        r1, c1, p1, eta2, eta1 = PhotoSynth_Jen(meteo)    
        #print(c2, c1)
        A1.append(p1)
        A2.append(p2)

import matplotlib.pyplot as plt

plt.plot(A1, label='Jen photosynthesis')
plt.plot(A2, label='Faquhar photosynthesis')

#plt.scatter(range(0, 2000, 50), A1, label='Jen photosynthesis')
#plt.scatter(range(0, 2000, 50), A2, label='Faquhar photosynthesis')

plt.ylabel('Photosynthesis')
plt.xlabel('Q [umol m-2 s-1] absorbed PAR flux')
plt.legend()

plt.show()
