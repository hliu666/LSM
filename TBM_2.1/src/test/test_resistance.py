# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 13:36:30 2022

@author: 16072
"""
from numpy import log, arctan, pi, exp, sinh
import numpy as np
from resistance_pars import *

def calc_Psi_M(zol):
    if zol > 0:
        # for stable and netural (zoL = 0 -> Psi_M = 0) conditions
        a = 6.1
        b = 2.5
        psi_m = -a * np.log(zol + (1.0 + zol**b)**(1.0 / b))
    else:
        # for unstable conditions
        y = -zol
        a = 0.33
        b = 0.41
        x = (y / a)**0.333333
    
        psi_0 = -np.log(a) + 3**0.5 * b * a**0.333333 * np.pi / 6.0
        y = np.minimum(y, b**-3)
        psi_m = (np.log(a + y) - 3.0 * b * y**0.333333 +
                    (b * a**0.333333) / 2.0 * np.log((1.0 + x)**2 / (1.0 - x + x**2)) +
                    3.0**0.5 * b * a**0.333333 * np.arctan((2.0 * x - 1.0) / 3**0.5) +
                    psi_0)

    return psi_m

def calc_u_star(u, z_u, L, d_0, z_0M):
    '''Friction velocity.

    Parameters
    ----------
    u : float
        wind speed above the surface (m s-1).
    z_u : float
        wind speed measurement height (m).
    L : float
        Monin Obukhov stability length (m).
    d_0 : float
        zero-plane displacement height (m).
    z_0M : float
        aerodynamic roughness length for momentum transport (m).

    References
    ----------
    .. [Brutsaert2005] Brutsaert, W. (2005). Hydrology: an introduction (Vol. 61, No. 8).
        Cambridge: Cambridge University Press.
    '''

    # calculate correction factors in other conditions
    L = max(L, 1e-36)
    Psi_M = calc_Psi_M((z_u - d_0) / L)
    Psi_M0 = calc_Psi_M(z_0M / L)
    del L
    u_star = u * KARMAN / (np.log((z_u - d_0) / z_0M) - Psi_M + Psi_M0)
    return max(u_star, U_FRICTION_MIN)

def calc_u_C_star(u_friction, h_C, d_0, z_0M, L=float('inf')):
    ''' MOST wind speed at the canopy

    Parameters
    ----------
    u_friction : float
        friction velocity (m s-1).
    h_C : float
        canopy height (m).
    d_0 : float
        zero-plane displacement height.
    z_0M : float
        aerodynamic roughness length for momentum transport (m).
    L : float, optional
        Monin-Obukhov length (m).

    Returns
    -------
    u_C : float
        wind speed at the canop interface (m s-1).
    '''

    Psi_M = calc_Psi_M((h_C - d_0) / L)
    Psi_M0 = calc_Psi_M(z_0M / L)

    # calcualte u_C, wind speed at the top of (or above) the canopy
    u_C = (u_friction * (np.log((h_C - d_0) / z_0M) - Psi_M + Psi_M0)) / KARMAN
    return u_C

def calc_R_x_Choudhury(u_C, F, leaf_width, alpha_prime=3.0):
    ''' Estimates aerodynamic resistance at the canopy boundary layer.

    Estimates the aerodynamic resistance at the canopy boundary layer based on the
    K-Theory model of [Choudhury1988]_.

    Parameters
    ----------
    u_C : float
        wind speed at the canopy interface (m s-1).
    F : float
        local Leaf Area Index.
    leaf_width : float
        efective leaf width size (m).
    alpha_prime : float, optional
        Wind exctinction coefficient, default=3.

    Returns
    -------
    R_x : float
        Aerodynamic resistance at the canopy boundary layer (s m-1).

    References
    ----------
    .. [Choudhury1988] Choudhury, B. J., & Monteith, J. L. (1988). A four-layer model
        for the heat budget of homogeneous land surfaces.
        Royal Meteorological Society, Quarterly Journal, 114(480), 373-398.
        http://dx/doi.org/10.1002/qj.49711448006.
    '''

    # Eqs. 29 & 30 [Choudhury1988]_
    R_x = (1.0 / (F * (2.0 * CM_a / alpha_prime)
           * np.sqrt(u_C / leaf_width) * (1.0 - np.exp(-alpha_prime / 2.0))))
    # R_x=(alpha_u*(sqrt(leaf_width/U_C)))/(2.0*alpha_0*LAI*(1.-exp(-alpha_u/2.0)))
    return R_x

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

    # Soil resistance eqs. 24 & 25 [Choudhury1988]_
    K_h = KARMAN * u_star * (h_C - d_0)
    del u_star
    R_S = ((h_C * np.exp(alpha_k) / (alpha_k * K_h))
           * (np.exp(-alpha_k * z0_soil / h_C) - np.exp(-alpha_k * (d_0 + z_0M) / h_C)))

    return R_S

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

def resistances(lai, L, u):
		
    #n: dimensionless wind extinction coefficient                       W&V Eq 33
    n		= Cd*lai/(2*KARMAN**2)     #                            [] 
    
    # stability correction for non-neutral conditions
    unst        = (L < 0 and L >-500)
    st          = (L > 0 and L < 500)  
    x       	= abs(1-16*zm/L)**(1/4) # only used for unstable
    
    # stability correction functions, friction velocity and Kh=Km=Kv
    pm_z    	= psim(zm -d_0,L,unst,st,x)
    ph_z    	= psih(zm -d_0,L,unst,st,x)
    pm_h        = psim(h_C -d_0,L,unst,st,x)
    #ph_h       = psih(h -d_0,L,unst,st)
    if zm >= zr:
        ph_zr = psih(zr-d_0,L,unst,st,x)
    else:
        ph_zr = ph_z
    phs_zr      = phstar(zr,zr,d_0,L,st,unst,x);
    phs_h		= phstar(h_C ,zr,d_0,L,st,unst,x);
    
    ustar   	= max(.001,KARMAN*u/(log((zm-d_0)/z_0M) - pm_z))  #          W&V Eq 30
    Kh          = KARMAN*ustar*(zr-d_0)                         #          W&V Eq 35

    # wind speed at height h and z_0M
    uh			= max(ustar/KARMAN * (log((h_C-d_0)/z_0M) - pm_h),.01)
    uz0 		= uh*exp(n*((z_0M+d_0)/h_C-1))                     #       W&V Eq 32
    
    # resistances
    
    if zm > zr:
        rai = 1.0/(KARMAN*ustar)*(log((zm-d_0) /(zr-d_0))  - ph_z   + ph_zr) 
    else:
        rai = 0.0
    rar = 1.0/(KARMAN*ustar)*((zr-h_C)/(zr-d_0)) - phs_zr + phs_h # W&V Eq 39
    rac = h_C*sinh(n)/(n*Kh)*(log((exp(n)-1)/(exp(n)+1))-log((exp(n*(z_0M+d_0)/h_C)-1)/(exp(n*(z_0M+d_0)/h_C)+1))) # W&V Eq 42
    rws = h_C*sinh(n)/(n*Kh)*(log((exp(n*(z_0M+d_0)/h_C)-1)/(exp(n*(z_0M+d_0)/h_C)+1))-log((exp(n*(.01)/h_C)-1)/(exp(n*(.01)/h_C)+1))) # W&V Eq 43
    #rbc = 70/LAI * sqrt(w./uz0)						%		W&V Eq 31, but slightly different
    
    u_star = calc_u_star(u, z_u, L, d_0, z_0M)
    u_C = calc_u_C_star(u_star, h_C, d_0, z_0M)
    R_x = calc_R_x_Choudhury(u_C, lai, leaf_width)
    R_s = calc_R_S_Choudhury(u_star, h_C, z_0M, d_0, zm)
    
    raa  = rai + rar + rac # aerodynamic resistance above the canopy
    rawc = rwc # + rbc     # aerodynamic resistance within the canopy (canopy)
    raws = rws + rbs       # aerodynamic resistance within the canopy (soil)
    
    return raa, rawc, raws, ustar, R_x, R_s

lai = 6
l_mo = 50
wds_arr = np.linspace(0.5, 4.0, num=50)
raa_list, rawc_list, raws_list, ustar_list = [], [], [], []
for wds in wds_arr:
    raa, rawc, raws, ustar, R_x, R_s = resistances(lai, l_mo, wds)
    raa_list.append(raa+rawc)
    rawc_list.append(R_x) 
    raws_list.append(raa+raws) 
    ustar_list.append(R_s)

import matplotlib.pyplot as plt
fig, axs = plt.subplots(2, 2, figsize=(12, 10)) 
axs[0,0].plot(wds_arr, raa_list, 'black', linewidth=2, label='raa')  
axs[0,1].plot(wds_arr, rawc_list, 'black', linewidth=2, label='rawc')  
axs[1,0].plot(wds_arr, raws_list, 'black', linewidth=2, label='raws')  
axs[1,1].plot(wds_arr, ustar_list, 'black', linewidth=2, label='ustar') 

axs[0,0].legend() 
axs[0,1].legend() 
axs[1,0].legend() 
axs[1,1].legend() 