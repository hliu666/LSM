# -*- coding: utf-8 -*-
"""
Created on Sat Dec 31 14:59:26 2022

@author: hliu
"""
import numpy as np

def calc_vapor_pressure(T_C):
    """Calculate the saturation water vapour pressure.

    Parameters
    ----------
    T_C : float
        temperature (C).

    Returns
    -------
    ea : float
        saturation water vapour pressure (mb).
    """

    ea = 6.112 * np.exp((17.67 * T_C) / (T_C + 243.5))
    return ea


def calc_delta_vapor_pressure(T_C):
    """Calculate the slope of saturation water vapour pressure.

    Parameters
    ----------
    T_C : float
        temperature (C).

    Returns
    -------
    s : float
        slope of the saturation water vapour pressure (kPa K-1)
    """

    s = 4098.0 * (0.6108 * np.exp(17.27 * T_C / (T_C + 237.3))) / ((T_C + 237.3)**2)
    return s*10 #convert Kpa to hpa

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

# ==============================================================================
# List of constants used in Meteorological computations
# ==============================================================================
# convert temperatures to K
T2K   = 273.15             
# Stephan Boltzmann constant (W m-2 K-4)
sigmaSB = 5.670373e-8
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
# Molecular mass of water [g mol-1]     
MH2O = 18   
#Molecular mass of dry air [g mol-1]       
Mair = 28.96 
# Air pressure
p = 970  
# Atmospheric O2 concentration       
o = 209   
# atmospheric CO2 concentration
ca = 390    
# Conversion of vapour pressure [Pa] to absolute humidity [kg kg-1]
e_to_q = MH2O/Mair/p   
# soil resistance for evaporation from the pore space
rss = 500.0       
               
