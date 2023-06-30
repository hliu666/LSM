# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 11:28:33 2022

@author: hliu
"""
import numpy as np

def weighted_sum_over_lidf(lidf, tts, tto, psi):
    ks = 0.0
    ko = 0.0
    bf = 0.0
    sob = 0.0
    sof = 0.0
    cts = np.cos(np.radians(tts))
    cto = np.cos(np.radians(tto))
    ctscto = cts * cto
    
    n_angles = len(lidf)
    angle_step = float(90.0 / n_angles)
    litab = np.arange(n_angles) * angle_step + (angle_step * 0.5)
    #litab = np.array( [5.,15.,25.,35.,45.,55.,65.,75.,81.,83.,85.,87.,89.])
    for i, ili in enumerate(litab):
        ttl = 1.0 * ili
        cttl = np.cos(np.radians(ttl))
        # SAIL volume scattering phase function gives interception and portions to be multiplied by rho and tau
        [chi_s, chi_o, frho, ftau] = volscatt(tts, tto, psi, ttl)
        #print(chi_s, chi_o)
        # Extinction coefficients
        ksli = chi_s / cts
        koli = chi_o / cto
        # Area scattering coefficient fractions
        sobli = frho * np.pi / ctscto
        sofli = ftau * np.pi / ctscto
        bfli = cttl ** 2.0
        ks += ksli * float(lidf[i])
        ko += koli * float(lidf[i])
        bf += bfli * float(lidf[i])
        sob += sobli * float(lidf[i])
        sof += sofli * float(lidf[i])

    Gs = ks * cts
    Go = ko * cto   
     
    return Gs, Go, ks, ko, bf, sob, sof 

def volscatt(tts, tto, psi, ttl):
    """Compute volume scattering functions and interception coefficients
    for given solar zenith, viewing zenith, azimuth and leaf inclination angle.
    Parameters
    ----------
    tts : float
        Solar Zenith Angle (degrees).
    tto : float
        View Zenight Angle (degrees).
    psi : float
        View-Sun reliative azimuth angle (degrees).
    ttl : float
        leaf inclination angle (degrees).
    Returns
    -------
    chi_s : float
        Interception function  in the solar path.
    chi_o : float
        Interception function  in the view path.
    frho : float
        Function to be multiplied by leaf reflectance to obtain the volume scattering.
    ftau : float
        Function to be multiplied by leaf transmittance to obtain the volume scattering.
    References
    ----------
    Wout Verhoef, april 2001, for CROMA.
    """

    cts = np.cos(np.radians(tts))
    cto = np.cos(np.radians(tto))
    sts = np.sin(np.radians(tts))
    sto = np.sin(np.radians(tto))
    cospsi = np.cos(np.radians(psi))
    psir = np.radians(psi)
    cttl = np.cos(np.radians(ttl))
    sttl = np.sin(np.radians(ttl))
    cs = cttl * cts
    co = cttl * cto
    ss = sttl * sts
    so = sttl * sto
    cosbts = 5.
    if abs(ss) > 1e-6: cosbts = -cs / ss
    cosbto = 5.
    if abs(so) > 1e-6: cosbto = -co / so
    if abs(cosbts) < 1.0:
        bts = np.arccos(cosbts)
        ds = ss
    else:
        bts = np.pi
        ds = cs
    chi_s = 2. / np.pi * ((bts - np.pi * 0.5) * cs + np.sin(bts) * ss)
    if abs(cosbto) < 1.0:
        bto = np.arccos(cosbto)
        do_ = so
    else:
        if tto < 90.:
            bto = np.pi
            do_ = co
        else:
            bto = 0.0
            do_ = -co
    #print(tto, bto)
    chi_o = 2.0 / np.pi * ((bto - np.pi * 0.5) * co + np.sin(bto) * so)
    btran1 = abs(bts - bto)
    btran2 = np.pi - abs(bts + bto - np.pi)
    if psir <= btran1:
        bt1 = psir
        bt2 = btran1
        bt3 = btran2
    else:
        bt1 = btran1
        if psir <= btran2:
            bt2 = psir
            bt3 = btran2
        else:
            bt2 = btran2
            bt3 = psir
    t1 = 2. * cs * co + ss * so * cospsi
    t2 = 0.
    if bt2 > 0.: t2 = np.sin(bt2) * (2. * ds * do_ + ss * so * np.cos(bt1) * np.cos(bt3))
    denom = 2. * np.pi ** 2
    frho = ((np.pi - bt2) * t1 + t2) / denom
    ftau = (-bt2 * t1 + t2) / denom
    if frho < 0.: frho = 0.
    if ftau < 0.: ftau = 0.

    return [chi_s, chi_o, frho, ftau]

def hotspot_calculations(lai, ko, ks, CIo, CIs, dso):
    ko *= CIo
    ks *= CIs
    
    # Treatment of the hotspot-effect
    alf = 1e36

    hotspot = 0.05

    tss = np.exp(-ks * lai)
    
    # Apply correction 2/(K+k) suggested by F.-M. Breon
    if hotspot > 0.:
        alf = (dso / hotspot) * 2. / (ks + ko)
    if alf == 0.:
        # The pure hotspot
        tsstoo = tss
        sumint = (1. - tss) / (ks * lai)
    else:
        # Outside the hotspot
        alf = (dso / hotspot) * 2. / (ks + ko)
        fhot = lai * np.sqrt(ko * ks)
        # Integrate by exponential Simpson method in 20 steps the steps are arranged according to equal partitioning of the slope of the joint probability function
        x1 = 0.
        y1 = 0.
        f1 = 1.
        fint = (1. - np.exp(-alf)) * .05
        sumint = 0.
        for istep in range(1, 21):
            if istep < 20:
                x2 = -np.log(1. - istep * fint) / alf
            else:
                x2 = 1.
            y2 = -(ko + ks) * lai * x2 + fhot * (1. - np.exp(-alf * x2)) / alf
            f2 = np.exp(y2)
            sumint = sumint + (f2 - f1) * (x2 - x1) / (y2 - y1)
            x1 = x2
            y1 = y2
            f1 = f2

        tsstoo = f1
        if np.isnan(sumint):
            sumint = 0.
            
    gammasos = ko * lai * sumint
    return gammasos, tsstoo #kc, kg 

def define_geometric_constant(tts, tto, psi):
    tants = np.tan(np.radians(tts))
    tanto = np.tan(np.radians(tto))
    cospsi = np.cos(np.radians(psi))
    dso = np.sqrt(tants ** 2. + tanto ** 2. - 2. * tants * tanto * cospsi)
    return dso