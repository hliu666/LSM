# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 15:40:17 2022

@author: hliu
"""
def calc_z_0M(h_C):
    """ Aerodynamic roughness lenght.
    Estimates the aerodynamic roughness length for momentum trasport
    as a ratio of canopy height.
    Parameters
    ----------
    h_C : float
        Canopy height (m).
    Returns
    -------
    z_0M : float
        aerodynamic roughness length for momentum transport (m).
    """

    z_0M = h_C * 0.125
    return z_0M

def calc_d_0(h_C):
    ''' Zero-plane displacement height
    Calculates the zero-plane displacement height based on a
    fixed ratio of canopy height.
    Parameters
    ----------
    h_C : float
        canopy height (m).
    Returns
    -------
    d_0 : float
        zero-plane displacement height (m).'''

    d_0 = h_C * 0.65

    return d_0


# von Karman's constant
KARMAN  = 0.41
# acceleration of gravity (m s-2)
GRAVITY = 9.8
U_FRICTION_MIN = 0.01
U_C_MIN = 0.01

leaf_width = 0.1 #efective leaf width size (m)

h_C     =  10.0 # vegetation height
d_0     =  calc_d_0(h_C)  #displacement height
z_0M    =  calc_z_0M(h_C) #roughness length for momentum of the canopy
zm      =  10.0 # Measurement height of meteorological data
z_u     =  10.0 # Height of measurement of windspeed (m).

CM_a    =  0.01 # Choudhury and Monteith 1988 leaf drag coefficient
 