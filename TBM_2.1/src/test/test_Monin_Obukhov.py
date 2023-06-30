# -*- coding: utf-8 -*-
"""
Created on Sat Dec 31 14:20:49 2022

@author: hliu
"""
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
# von Karman's constant
KARMAN = 0.41
# acceleration of gravity (m s-2)
GRAVITY = 9.8

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
    return c_p

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

    Lambda = 1e6 * (2.501 - (2.361e-3 * (T_A_K - 273.15)))
    return Lambda

def calc_rho(p, ea, T_A_K):
    '''Calculates the density of air.

    Parameters
    ----------
    p : float
        total air pressure (dry air + water vapour) (mb).
    ea : float
        water vapor pressure at reference height above canopy (mb).
    T_A_K : float
        air temperature at reference height (Kelvin).

    Returns
    -------
    rho : float
        density of air (kg m-3).

    References
    ----------
    based on equation (2.6) from Brutsaert (2005): Hydrology - An Introduction (pp 25).'''

    # p is multiplied by 100 to convert from mb to Pascals
    rho = ((p * 100.0) / (R_d * T_A_K)) * (1.0 - (1.0 - epsilon) * ea / p)
    return rho


def Monin_Obukhov_test(ustar, T_A_C, rho, c_p, H, LE):
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

import numpy as np
p = 309
ea = 40.0
ustar = 120
H = 100
LE = 10
T_A_C_arr = np.linspace(0.5, 35.0, num=100)

L_0_list, L_1_list = [], []
for T_A_C in T_A_C_arr:
    T_A_K = T_A_C + 273.15
    
    c_p = calc_c_p(p, ea)
    rho = calc_rho(p, ea, T_A_K)
    
    L_0 = Monin_Obukhov(ustar, T_A_C, H)
    L_1 = Monin_Obukhov_test(ustar, T_A_C, rho, c_p, H, LE)
    
    L_0_list.append(L_0)
    L_1_list.append(L_1)
    
import matplotlib.pyplot as plt
fig, axs = plt.subplots(2, 2, figsize=(12, 10)) 
axs[0,0].plot(T_A_C_arr, L_0_list, 'black', linewidth=2, label='L0')  
axs[0,1].plot(T_A_C_arr, L_1_list, 'black', linewidth=2, label='L1')  
axs[1,0].plot(T_A_C_arr, L_0_list, 'black', linewidth=2, label='L0')  
axs[1,1].plot(T_A_C_arr, L_1_list, 'black', linewidth=2, label='L1') 

axs[0,0].legend() 
axs[0,1].legend() 
axs[1,0].legend() 
axs[1,1].legend()     