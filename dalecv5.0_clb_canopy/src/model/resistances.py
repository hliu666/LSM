# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 13:36:30 2022

@author: 16072
"""
from numpy import log, arctan, pi, exp, sinh
import numpy as np

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
    return np.asarray(z_0M)

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

    return np.asarray(d_0)

# subfunction pm for stability correction (eg. Paulson, 1970)
def psim(z,L,unst,st,x):
    pm = 0
    if unst:
        pm = 2*log((1+x)/2)+log((1+x**2)/2)-2*arctan(x)+pi/2   #   unstable
    elif st:
        pm = -5*z/L                                            #   stable
    return pm
    
# subfunction ph for stability correction (eg. Paulson, 1970)
def psih(z,L,unst,st,x):
    ph = 0
    if unst:
        ph = 2*log((1+x**2)/2);                                #   unstable
    elif st:
        ph = -5*z/L                                            #   stable
    return ph

# subfunction ph for stability correction (eg. Paulson, 1970)
def phstar(z,zR,d,L,st,unst,x):
    phs = 0
    if unst:
        phs     = (z-d)/(zR-d)*(x**2-1)/(x**2+1)
    elif st:
        phs     = -5*z/L
    return phs

def resistances(lai, L, wds):
    # parameters
    #global constants
    kappa   =  0.4  #Von Karman constant
    Cd      =  0.5  #leaf drag coefficient
    LAI     =  lai
    rwc     =  0.5  #within canopy layer resistance
    h       =  10.0 #vegetation height
    d       =  calc_d_0(h) #1.7604 #displacement height
    z0m     =  calc_z_0M(h)#0.0782 #roughness length for momentum of the canopy

    z       =  10.0 #measurement height of meteorological data
    u       =  wds #wind speed at height z
    L       =  L#1E-6
    rbs     =  20.0#soil boundary layer resistance (from Aerodynamic)
    
    # derived parameters
    #zr: top of roughness sublayer, bottom of intertial sublayer
    zr		= 2.5*h                  #                            [m]			
    #n: dimensionless wind extinction coefficient                       W&V Eq 33
    n		= Cd*LAI/(2*kappa**2)     #                            [] 
    
    # stability correction for non-neutral conditions
    unst        = (L < 0 and L >-500)
    st          = (L > 0 and L < 500)  
    x       	= abs(1-16*z/L)**(1/4) # only used for unstable
    
    # stability correction functions, friction velocity and Kh=Km=Kv
    pm_z    	= psim(z -d,L,unst,st,x)
    ph_z    	= psih(z -d,L,unst,st,x)
    pm_h        = psim(h -d,L,unst,st,x)
    #ph_h       = psih(h -d,L,unst,st)
    if z >= zr:
        ph_zr = psih(zr-d,L,unst,st,x)
    else:
        ph_zr = ph_z
    phs_zr      = phstar(zr,zr,d,L,st,unst,x);
    phs_h		= phstar(h ,zr,d,L,st,unst,x);
    
    ustar   	= max(.001,kappa*u/(log((z-d)/z0m) - pm_z))  #          W&V Eq 30
    Kh          = kappa*ustar*(zr-d)                         #          W&V Eq 35
    
    if unst:
        resist_out_Kh	= Kh*(1-16*(h-d)/L)**.5 # W&V Eq 35
    elif st:
        resist_out_Kh   = Kh*(1+ 5*(h-d)/L)**-1 # W&V Eq 35
    else:
        resist_out_Kh = Kh
    
    # wind speed at height h and z0m
    uh			= max(ustar/kappa * (log((h-d)/z0m) - pm_h),.01)
    uz0 		= uh*exp(n*((z0m+d)/h-1))                     #       W&V Eq 32
    
    # resistances
    
    if z > zr:
        rai = 1.0/(kappa*ustar)*(log((z-d) /(zr-d))  - ph_z   + ph_zr) 
    else:
        rai = 0.0
    rar = 1.0/(kappa*ustar)*((zr-h)/(zr-d)) - phs_zr + phs_h # W&V Eq 39
    rac = h*sinh(n)/(n*Kh)*(log((exp(n)-1)/(exp(n)+1))-log((exp(n*(z0m+d)/h)-1)/(exp(n*(z0m+d)/h)+1))) # W&V Eq 42
    rws = h*sinh(n)/(n*Kh)*(log((exp(n*(z0m+d)/h)-1)/(exp(n*(z0m+d)/h)+1))-log((exp(n*(.01)/h)-1)/(exp(n*(.01)/h)+1))) # W&V Eq 43
    #rbc = 70/LAI * sqrt(w./uz0);						%		W&V Eq 31, but slightly different
    R_S = calc_R_S_Choudhury(ustar, h, z0m, d, z)

    raa  = rai + rar + rac # aerodynamic resistance above the canopy
    rawc = rwc # + rbc;    # aerodynamic resistance within the canopy (canopy)
    raws = rws + rbs       # aerodynamic resistance within the canopy (soil)
    
    return raa, rawc, raws, ustar

def calc_R_S_Choudhury(u_star, h_C, z_0M, d_0, zm, z0_soil=0.01, alpha_k=2.0):
    ''' Aerodynamic resistance at the  soil boundary layer.
    Estimates the aerodynamic resistance at the  soil boundary layer based on the
    K-Theory model of [Choudhury1988]_.
    Parameters
    ----------
    u_star : float
        friction velocity (m s-1).
    h_C : float
        canopy height (m).
    z_0M : float
        aerodynamic roughness length for momentum trasport (m).
    d_0 : float
        zero-plane displacement height (m).
    zm : float
        height on measurement of wind speed (m).
    z0_soil : float, optional
        roughness length of the soil layer, use z0_soil=0.01.
    alpha_k : float, optional
        Heat diffusion coefficient, default=2.
    Returns
    -------
    R_S : float
        Aerodynamic resistance at the  soil boundary layer (s m-1).
    References
    ----------
    .. [Choudhury1988] Choudhury, B. J., & Monteith, J. L. (1988). A four-layer model
        for the heat budget of homogeneous land surfaces.
        Royal Meteorological Society, Quarterly Journal, 114(480), 373-398.
        http://dx/doi.org/10.1002/qj.49711448006.
    '''
    # von Karman's constant
    KARMAN = 0.41

    # Soil resistance eqs. 24 & 25 [Choudhury1988]_
    K_h = KARMAN * u_star * (h_C - d_0)
    del u_star
    R_S = ((h_C * np.exp(alpha_k) / (alpha_k * K_h))
           * (np.exp(-alpha_k * z0_soil / h_C) - np.exp(-alpha_k * (d_0 + z_0M) / h_C)))

    return np.asarray(R_S)