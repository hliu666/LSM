# -*- coding: utf-8 -*-
"""
Created on Sun Jan  1 14:15:26 2023

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

# functions for saturated vapour pressure 
def es_fun(T):
    return 6.107*10**(7.5*T/(237.3+T))

def s_fun(T):
    return 6.107*10**(7.5*T/(237.3+T))*2.3026*7.5*237.3/((237.3+T)**2)

print(s_fun(25))
print(calc_delta_vapor_pressure(25))

print(es_fun(25))
print(calc_vapor_pressure(25))