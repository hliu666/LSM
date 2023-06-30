# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 14:13:43 2022

@author: hliu
"""
from numpy import exp, radians, cos, sin, pi
import numpy as np
import itertools
#%% 1) Initialize Leaf Properties based on SIP

#def sip_leaf(Cab, Car, Cbrown, Cw, Cm, Ant, Alpha, fLMA_k=2765.0, gLMA_k=-631.5441990240528, gLMA_b=0.006443637536132927):
def sip_leaf(Cab, Car, Cbrown, Cw, Cm, Ant, Alpha, fLMA_k, gLMA_k, gLMA_b):

    '''SIP D Plant leaf reflectance and transmittance modeled
    from 400 nm to 2500 nm (1 nm step).
    Parameters
    ----------    
    Cab : float
        chlorophyll a+b content (mug cm-2).
    Car : float
        carotenoids content (mug cm-2).
    Cbrown : float
        brown pigments concentration (unitless).
    Cw : float
        equivalent water thickness (g cm-2 or cm).
    Cm : float
        dry matter content (g cm-2).
    Ant : float
        Anthocianins concentration (mug cm-2).
    Alpha: float
        Constant for the the optimal size of the leaf scattering element 
    Returns
    -------
    l : array_like
        wavelenght (nm).
    rho : array_like
        leaf reflectance .
    tau : array_like
        leaf transmittance .
    '''
    prospectpro = np.loadtxt("../../data/parameters/dataSpec_PDB.txt")
    
    lambdas   = prospectpro[:,0].reshape(-1,1)
    nr        = prospectpro[:,1].reshape(-1,1)
    Cab_k     = prospectpro[:,2].reshape(-1,1)
    Car_k     = prospectpro[:,3].reshape(-1,1)
    Ant_k     = prospectpro[:,4].reshape(-1,1)    
    Cbrown_k  = prospectpro[:,5].reshape(-1,1)
    Cw_k      = prospectpro[:,6].reshape(-1,1)    
    Cm_k      = prospectpro[:,7].reshape(-1,1)

    kall    = (Cab*Cab_k + Car*Car_k + Ant*Ant_k + Cbrown*Cbrown_k + Cw*Cw_k + Cm*Cm_k)/(Cm*Alpha)
    w0      = np.exp(-kall)
    
    # spectral invariant parameters
    fLMA = fLMA_k*Cm
    gLMA = gLMA_k*(Cm - gLMA_b)
    
    p = 1-(1 - np.exp(-fLMA))/fLMA
    q = 2/(1+ np.exp(gLMA)) - 1
    qabs = np.sqrt(q**2)
    
    # leaf single scattering albedo
    w = w0*(1-p)/(1-p*w0)
    
    # leaf reflectance and leaf transmittance
    refl  = w*(1/2+q/2*(1-p*w0)/(1-qabs*p*w0))
    tran  = w*(1/2-q/2*(1-p*w0)/(1-qabs*p*w0))

    if isinstance(refl[0], float):
        time = 1
    else:
        time = len(refl[0])
    thermal  = np.full((2162-2100, time), 0.01)
    refl = np.vstack((refl[:2100], thermal))  
    tran = np.vstack((tran[:2100], thermal))  

    #refl = np.vstack((refl[:2100], thermal))  
    #tran = np.vstack((tran[:2100], thermal))  
    return [refl, tran]

#%% 2) Initialize Soil reflectance
def soil_spectra():
    """
    #Read soil and prospect parameters
    soil = np.loadtxt("../../data/soil_reflectance.txt") 
    soil_spectrum1 = soil[:,0]
    soil_spectrum2 = soil[:,1]
    
    rsoil = 1.0  #土壤标量1（亮度）
    psoil = 0.5  #土壤标量2（水分）
    
    rg = rsoil * (
        psoil * soil_spectrum1 + (1.0 - psoil) * soil_spectrum2
    ) #soil reflectance
    """
    rsoil = 0.5 #土壤标量1（亮度）
    
    #soil = np.genfromtxt("../../data/HARV_soil_reflectance.csv")
    #soil[soil < 0] = 0 
    #rg   = rsoil*soil
    soil = np.genfromtxt("../../data/parameters/soil_reflectance.txt")
    rg   = rsoil*(1-soil)
    
    rg_t = np.array([0.06]*(2162-2000))
    rg_spc = np.concatenate([rg[:2000],rg_t])    
    
    return rg_spc

#%% 3) Calculation of incoming light Esun_ and Esky_
def atmoE():
    TOCirr = np.loadtxt("../../data/parameters/atmo.txt", skiprows=1)
    wl  = TOCirr[:,0]
    t1  = TOCirr[:,1]
    t3  = TOCirr[:,2]
    t4  = TOCirr[:,3]
    t5  = TOCirr[:,4]
    t12 = TOCirr[:,5]
    t16 = TOCirr[:,6]
    return wl, [t1, t3, t4, t5, t12, t16]

#%% 4) Bidirectional gap fraction function
def leafangles(a,b):  
    """                                   
    % Subroutine FluorSail_dladgen
    % Version 2.3 
    % For more information look to page 128 of "theory of radiative transfer models applied in optical remote sensing of
    % vegetation canopies"
    %
    % FluorSail for Matlab
    % FluorSail is created by Wout Verhoef, 
    % National Aerospace Laboratory (NLR)
    % Present e-mail: w.verhoef@utwente.nl
    %
    % This code was created by Joris Timmermans, 
    % International institute for Geo-Information Science and Earth Observation. (ITC)
    % Email: j.timmermans@utwente.nl
    %
    %% main function
    """
    F = np.zeros(13)
    for i in range(0,8):                                                               
        theta   =   (i+1)*10              # theta_l =  10:80
        F[i]    =   dcum(a,b,theta)       # F(theta)
    
    for i in range(8,12):                                                              
        theta   =   80 + (i-7)*2;                         # theta_l = 82:88
        F[i]    =   dcum(a,b,theta);                      #  F(theta)
    
    for i in range(12,13):                              #  theta_l = 90:90
        F[i] =   1                                      #  F(theta)
    
    lidf = np.zeros(13)
    for i in np.arange(12,0,-1):                                                           
        lidf[i] = F[i]-F[i-1]                     
    lidf[0] =   F[0]                                   #  Boundary condition
    return lidf

def dcum(a, b, theta):
    rd  =   pi/180                                     #   Geometrical constant
    if a > 1:
        F    =   1 - cos(theta*rd)
    else:
        eps     =   1e-8
        delx    =   1
        
        x       =   2*rd *theta
        theta2  =   x
                                                                                
        while (delx > eps):
            y    =   a*sin(x) + 0.5*b*sin(2*x)
            dx   =   0.5*(y - x + theta2)
            x    =   x + dx
            delx =   abs(dx)
        F    =   (2*y + theta2)/pi                     #   Cumulative leaf inclination density function
    return F

def campbell(alpha, n_elements=18):
    """Calculate the Leaf Inclination Distribution Function based on the
    mean angle of [Campbell1990] ellipsoidal LIDF distribution.

    Parameters
    ----------
    alpha : float
        Mean leaf angle (degrees) use 57 for a spherical LIDF.
    n_elements : int
        Total number of equally spaced inclination angles .

    Returns
    -------
    lidf : list
        Leaf Inclination Distribution Function for 18 equally spaced angles.

    References
    ----------
    .. [Campbell1986] G.S. Campbell, Extinction coefficients for radiation in
        plant canopies calculated using an ellipsoidal inclination angle distribution,
        Agricultural and Forest Meteorology, Volume 36, Issue 4, 1986, Pages 317-321,
        ISSN 0168-1923, http://dx.doi.org/10.1016/0168-1923(86)90010-9.
    .. [Campbell1990] G.S Campbell, Derivation of an angle density function for
        canopies with ellipsoidal leaf angle distributions,
        Agricultural and Forest Meteorology, Volume 49, Issue 3, 1990, Pages 173-176,
        ISSN 0168-1923, http://dx.doi.org/10.1016/0168-1923(90)90030-A.
    """

    alpha = float(alpha)
    excent = np.exp(-1.6184e-5 * alpha ** 3. + 2.1145e-3 * alpha ** 2. - 1.2390e-1 * alpha + 3.2491)
    sum0 = 0.
    freq = []
    step = 90.0 / n_elements
    for i in range(n_elements):
        tl1 = np.radians(i * step)
        tl2 = np.radians((i + 1.) * step)
        x1 = excent / (np.sqrt(1. + excent ** 2. * np.tan(tl1) ** 2.))
        x2 = excent / (np.sqrt(1. + excent ** 2. * np.tan(tl2) ** 2.))
        if excent == 1.:
            freq.append(abs(np.cos(tl1) - np.cos(tl2)))
        else:
            alph = excent / np.sqrt(abs(1. - excent ** 2.))
            alph2 = alph ** 2.
            x12 = x1 ** 2.
            x22 = x2 ** 2.
            if excent > 1.:
                alpx1 = np.sqrt(alph2 + x12)
                alpx2 = np.sqrt(alph2 + x22)
                dum = x1 * alpx1 + alph2 * np.log(x1 + alpx1)
                freq.append(abs(dum - (x2 * alpx2 + alph2 * np.log(x2 + alpx2))))
            else:
                almx1 = np.sqrt(alph2 - x12)
                almx2 = np.sqrt(alph2 - x22)
                dum = x1 * almx1 + alph2 * np.arcsin(x1 / alph)
                freq.append(abs(dum - (x2 * almx2 + alph2 * np.arcsin(x2 / alph))))
    sum0 = sum(freq)
    lidf = []
    for i in range(n_elements):
        lidf.append(float(freq[i]) / sum0)

    return lidf

def verhoef_bimodal(a,b,n_elements=18):
    """Calculate the Leaf Inclination Distribution Function based on the
    Verhoef's bimodal LIDF distribution.

    Parameters
    ----------
    a : float
        controls the average leaf slope.
    b : float
        controls the distribution's bimodality.

            * LIDF type     [a,b].
            * Planophile    [1,0].
            * Erectophile   [-1,0].
            * Plagiophile   [0,-1].
            * Extremophile  [0,1].
            * Spherical     [-0.35,-0.15].
            * Uniform       [0,0].
            * requirement: |LIDFa| + |LIDFb| < 1.
    n_elements : int
        Total number of equally spaced inclination angles.

    Returns
    -------
    lidf : list
        Leaf Inclination Distribution Function at equally spaced angles.

    References
    ----------
    .. [Verhoef1998] Verhoef, Wout. Theory of radiative transfer models applied
        in optical remote sensing of vegetation canopies.
        Nationaal Lucht en Ruimtevaartlaboratorium, 1998.
        http://library.wur.nl/WebQuery/clc/945481.
        """

    freq = 1.0
    step = 90.0 / n_elements
    lidf = []
    angles = [i * step for i in reversed(range(n_elements))]
    for angle in angles:
        tl1 = np.radians(angle)
        if a > 1.0:
            f = 1.0 - np.cos(tl1)
        else:
            eps = 1e-8
            delx = 1.0
            x = 2.0 * tl1
            p = float(x)
            while delx >= eps:
                y = a * np.sin(x) + .5 * b * np.sin(2. * x)
                dx = .5 * (y - x + p)
                x = x + dx
                delx = abs(dx)
            f = (2. * y + p) / np.pi
        freq = freq - f
        lidf.append(freq)
        freq = float(f)
    lidf = list(reversed(lidf))
    return lidf

def cal_lidf(lidfa, lidfb):
    #lidfa = 30 # float Leaf Inclination Distribution at regular angle steps. 
    #lidfb = -0.15 # float Leaf Inclination Distribution at regular angle steps. 
    lidftype = 2 # float Leaf Inclination Distribution at regular angle steps.
    # Calcualte leaf angle distribution
    if lidftype == 1:
        #lidf = leafangles(lidfa, lidfb)
        lidf = verhoef_bimodal(lidfa, lidfb, n_elements=13)
    elif lidftype == 2:
        lidf = campbell(lidfa, n_elements=13)
    else:
        raise ValueError(
            "lidftype can only be 1 (Campbell) or 2 (ellipsoidal)"
        )
    
    return lidf

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

def weighted_sum_over_lidf_vec(lidf, tts, tto, psi):
    ks = 0
    ko = 0.
    bf = 0.
    sob = 0.
    sof = 0.
    cts = np.cos(np.radians(tts))
    cto = np.cos(np.radians(tto))
    ctscto = cts * cto

    n_angles = len(lidf)
    angle_step = 90.0 / n_angles
    litab = np.arange(n_angles) * angle_step + (angle_step * 0.5)

    for i, ili in enumerate(litab):
        ttl = 1. * ili
        cttl = np.cos(np.radians(ttl))
        # SAIL volume scattering phase function gives interception and portions to be multiplied by rho and tau
        [chi_s, chi_o, frho, ftau] = volscatt_vec(tts, tto, psi, ttl)
        # Extinction coefficients
        ksli = chi_s / cts
        koli = chi_o / cto
        # Area scattering coefficient fractions
        sobli = frho * np.pi / ctscto
        sofli = ftau * np.pi / ctscto
        bfli = cttl ** 2.
        ks += ksli * lidf[i]
        ko += koli * lidf[i]
        bf += bfli * lidf[i]
        sob += sobli * lidf[i]
        sof += sofli * lidf[i]

    Gs = ks * cts
    Go = ko * cto 
    
    #ks = np.clip(ks, 1e-16, 0.99999)
    #ko = np.clip(ko, 1e-16, 0.99999)

    #sob = np.clip(sob, 1e-16, 0.99999)
    #sof = np.clip(sof, 1e-16, 0.99999)
    
    return Gs, Go, ks, ko, bf, sob, sof 

def weighted_sum_over_lidf_solar(tts, lidf):  
    
    #litab   = np.array([5.,15.,25.,35.,45.,55.,65.,75.,81.,83.,85.,87.,89.]).T  
    n_angles = len(lidf)
    angle_step = 90.0 / n_angles
    litab = np.arange(n_angles) * angle_step + (angle_step * 0.5)
    cos_tts = np.cos(np.radians(tts))    
    chi_s   = volscatt_solar(tts,litab)    
    ksli    = chi_s/cos_tts
    k       = np.dot(lidf,ksli)
    Gs      = k*cos_tts 
    return Gs, k

def weighted_sum_over_lidf_solar_vec(tts, lidf):  
    
    #litab   = np.array([5.,15.,25.,35.,45.,55.,65.,75.,81.,83.,85.,87.,89.]).T  
    n_angles = len(lidf)
    angle_step = 90.0 / n_angles
    litab = np.arange(n_angles) * angle_step + (angle_step * 0.5)
    cos_tts = np.cos(np.radians(tts))    
    chi_s   = volscatt_solar_vec(tts,litab)    
    ksli    = chi_s/cos_tts
    k       = np.dot(lidf,ksli)
    Gs      = k*cos_tts 
    return Gs, k

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

def volscatt_vec(tts, tto, psi, ttl):
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
    cosbts = np.ones(cs.shape) * 5.
    cosbto = np.ones(co.shape) * 5.
    cosbts[np.abs(ss) > 1e-6] = -cs[np.abs(ss) > 1e-6] / ss[np.abs(ss) > 1e-6]
    cosbto[np.abs(so) > 1e-6] = -co[np.abs(so) > 1e-6] / so[np.abs(so) > 1e-6]

    bts = np.ones(cosbts.shape) * np.pi
    ds = np.array(cs)
    bts[np.abs(cosbts) < 1.0] = np.arccos(cosbts[np.abs(cosbts) < 1.0])
    ds[np.abs(cosbts) < 1.0] = ss[np.abs(cosbts) < 1.0]
    chi_s = 2. / np.pi * ((bts - np.pi * 0.5) * cs + np.sin(bts) * ss)

    bto = np.zeros(cosbto.shape)
    do_ = np.zeros(cosbto.shape)
    bto[np.abs(cosbto) < 1.0] = np.arccos(cosbto[np.abs(cosbto) < 1.0])
    do_[np.abs(cosbto) < 1.0] = so[np.abs(cosbto) < 1.0]
    bto[np.logical_and(np.abs(cosbto) >= 1.0, tto < 90.)] = np.pi
    do_[np.logical_and(np.abs(cosbto) >= 1.0, tto < 90.)] = co[np.logical_and(np.abs(cosbto) >= 1.0, tto < 90.)]
    bto[np.logical_and(np.abs(cosbto) >= 1.0, tto > 90.)] = 0
    do_[np.logical_and(np.abs(cosbto) >= 1.0, tto > 90.)] = -co[np.logical_and(np.abs(cosbto) >= 1.0, tto > 90.)]

    #print(tto[0],bto[0])
    chi_o = 2.0 / np.pi * ((bto - np.pi * 0.5) * co + np.sin(bto) * so)
    btran1 = np.abs(bts - bto)
    btran2 = np.pi - np.abs(bts + bto - np.pi)
    bt1 = np.array(psir)
    bt2 = np.array(btran1)
    bt3 = np.array(btran2)
    bt1[psir > btran1] = btran1[psir > btran1]
    bt2[np.logical_and(psir > btran1, psir <= btran2)] = psir[np.logical_and(psir > btran1, psir <= btran2)]
    bt3[np.logical_and(psir > btran1, psir <= btran2)] = btran2[np.logical_and(psir > btran1, psir <= btran2)]
    bt2[np.logical_and(psir > btran1, psir > btran2)] = btran2[np.logical_and(psir > btran1, psir > btran2)]
    bt3[np.logical_and(psir > btran1, psir > btran2)] = psir[np.logical_and(psir > btran1, psir > btran2)]

    t1 = 2. * cs * co + ss * so * cospsi
    t2 = np.zeros(t1.shape)
    t2[bt2 > 0.] = np.sin(bt2[bt2 > 0.]) * (
                2. * ds[bt2 > 0.] * do_[bt2 > 0.] + ss[bt2 > 0.] * so[bt2 > 0.] * np.cos(bt1[bt2 > 0.]) * np.cos(
            bt3[bt2 > 0.]))
    denom = 2. * np.pi ** 2
    frho = ((np.pi - bt2) * t1 + t2) / denom
    ftau = (-bt2 * t1 + t2) / denom
    frho[frho < 0.] = 0.
    ftau[ftau < 0.] = 0.

    return [chi_s, chi_o, frho, ftau]

def volscatt_solar_vec(tts, ttl):
    #tts    [1]         Sun            zenith angle in degrees
    #tto    [1]         Observation    zenith angle in degrees
    #psi    [1]         Difference of  azimuth angle between solar and viewing position
    #ttl    [ttl]      leaf inclination array
    cts = np.cos(np.radians(tts))                 #   cosine of sun zenith angle
    sts = np.sin(np.radians(tts))                 #   sine   of sun zenith angle
   
    cttl = np.cos(np.radians(ttl))                
    sttl = np.sin(np.radians(ttl))    

    cs = np.dot(cttl.reshape(-1,1), cts.reshape(1,-1))              
    ss = np.dot(sttl.reshape(-1,1), sts.reshape(1,-1))

    As = np.max(np.stack((cs, ss)), axis=0)
    bts = np.arccos(-cs/As)             
     
    chi_s = 2/pi*((bts-pi/2)*cs + np.sin(bts)*ss) 
    return chi_s 

def volscatt_solar(tts, ttl):
    #tts    [1]         Sun            zenith angle in degrees
    #tto    [1]         Observation    zenith angle in degrees
    #psi    [1]         Difference of  azimuth angle between solar and viewing position
    #ttl    [ttl]      leaf inclination array
    cts = np.cos(np.radians(tts))                 #   cosine of sun zenith angle
    sts = np.sin(np.radians(tts))                 #   sine   of sun zenith angle
   
    cttl = np.cos(np.radians(ttl))                
    sttl = np.sin(np.radians(ttl))    

    cs = cttl*cts          
    ss = sttl*sts

    As = np.max(np.hstack((cs.reshape(-1,1), ss.reshape(-1,1))), axis=1)
    bts = np.arccos(-cs/As)             
     
    chi_s = 2/pi*((bts-pi/2)*cs + np.sin(bts)*ss) 
    return chi_s 

def CIxy(flag, tts, CI_thres):
    #CI varied with the zenith angle
    if flag == 0:
        u = 1-np.cos(np.radians(tts))
        CI = 0.1829*u+0.6744
        return CI
    
    #CI as a constant 
    elif flag == 1:
        CI = CI_thres
        if isinstance(tts, float):
            return CI
        else:
            return np.full(len(tts), CI)   
        
    #Without considering CI effect      
    elif flag == 2:
        CI = 1.0
        if isinstance(tts, float):
            return CI
        else:
            return np.full(len(tts), CI)  
    
def hemi_initial(flag, tts, lidf, CI_thres):
    xx=np.array([0.9602898565, -0.9602898565, 0.7966664774, -0.7966664774, 0.5255324099, -0.5255324099, 0.1834346425, -0.1834346425])
    
    ww=np.array([0.1012285363,  0.1012285363, 0.2223810345,  0.2223810345, 0.3137066459,  0.3137066459, 0.3626837834,  0.3626837834])   
    
    # * define limits of integration and the convertion factors for integration
    # * over thetaL (note the tL suffix!)
    upperlimit_tL = np.pi/2.0
    lowerlimit_tL = 0.0
    conv1_tL = (upperlimit_tL-lowerlimit_tL)/2.0
    conv2_tL = (upperlimit_tL+lowerlimit_tL)/2.0    
    
    # * define limits of integration and the convertion factors for integration
    # * over phiL (note the pL suffix!)
    upperlimit_pL = 2.0*pi
    lowerlimit_pL = 0.0
    conv1_pL = (upperlimit_pL-lowerlimit_pL)/2.0
    conv2_pL = (upperlimit_pL+lowerlimit_pL)/2.0  
    
    """
    pars = []
    for i in range(0,8): 
        neword_tL = conv1_tL*xx[i] + conv2_tL  
        for j in range(0,8): 
            neword_pL  = conv1_pL*xx[j] + conv2_pL
            if isinstance(tts, float):
                tto  = neword_tL*180/pi              # observer zenith angle
                psi  = neword_pL*180/pi              # relative azimuth angle 
                
                [Gs, Go, ks, ko, bf, sob, sof] = weighted_sum_over_lidf(lidf, tts, tto, psi)
                CIs = CIxy(flag, tts, CI_thres)
                CIo = CIxy(flag, tto, CI_thres) 
                [Ps_arr, Po_arr, int_res_arr, nl] = dir_gap_initial(tts, tto, psi, ks, ko, CIs, CIo)
            else:
                tto = np.tile(neword_tL*180/pi, len(tts))
                psi = np.tile(neword_pL*180/pi, len(tts))

                [Gs, Go, ks, ko, bf, sob, sof] = weighted_sum_over_lidf_vec(lidf, tts, tto, psi)
                CIs = CIxy(flag, tts, CI_thres)
                CIo = CIxy(flag, tto, CI_thres)
                [Ps_arr, Po_arr, int_res_arr, nl] = dir_gap_initial_vec(tts, tto, psi, ks, ko, CIs, CIo)

            pars.append([tts, tto, psi, ks, ko, sob, sof, CIs, CIo, Ps_arr, Po_arr, int_res_arr, nl])
    """
    """
    Vectorize the parameters calculation
    """
    tL = list((conv1_tL*xx + conv2_tL)*180/pi) 
    pL = list((conv1_pL*xx + conv2_pL)*180/pi) 
    
    angle_arr = np.array(list(itertools.product(tts, tL, pL)))
    tts, tto, psi = angle_arr[:,0], angle_arr[:,1], angle_arr[:,2]
    [Ga, Go, ks, ko, bf, sob, sof] = weighted_sum_over_lidf_vec(lidf, tts, tto, psi)
    CIs = CIxy(flag, tts, CI_thres)
    CIo = CIxy(flag, tto, CI_thres) 
    
    return [tts, tto, psi, ks, ko, sob, sof, CIs, CIo]

def dif_initial(flag, tto, lidf, CI_thres):
    xx=np.array([0.9602898565, -0.9602898565, 0.7966664774, -0.7966664774, 0.5255324099, -0.5255324099, 0.1834346425, -0.1834346425])
    
    ww=np.array([0.1012285363,  0.1012285363, 0.2223810345,  0.2223810345, 0.3137066459,  0.3137066459, 0.3626837834,  0.3626837834])   
    
    # * define limits of integration and the convertion factors for integration
    # * over thetaL (note the tL suffix!)
    upperlimit_tL = np.pi/2.0
    lowerlimit_tL = 0.0
    conv1_tL = (upperlimit_tL-lowerlimit_tL)/2.0
    conv2_tL = (upperlimit_tL+lowerlimit_tL)/2.0    
    
    # * define limits of integration and the convertion factors for integration
    # * over phiL (note the pL suffix!)
    upperlimit_pL = 2.0*pi
    lowerlimit_pL = 0.0
    conv1_pL = (upperlimit_pL-lowerlimit_pL)/2.0
    conv2_pL = (upperlimit_pL+lowerlimit_pL)/2.0  

    """
    pars = []
    for i in range(0,8): 
        neword_tL = conv1_tL*xx[i] + conv2_tL  
        for j in range(0,8): 
            neword_pL  = conv1_pL*xx[j] + conv2_pL
            if isinstance(tto, float):
                tta  = neword_tL*180/pi              # observer zenith angle
                psi  = neword_pL*180/pi              # relative azimuth angle    
                
                [Gs, Go, ks, ko, bf, sob, sof] = weighted_sum_over_lidf(lidf, tta, tto, psi)
                CIo = CIxy(flag, tto, CI_thres)
                CIs = CIxy(flag, tta, CI_thres)
                [Ps_arr, Po_arr, int_res_arr, nl] = dir_gap_initial(tta, tto, psi, ks, ko, CIs, CIo)
            else:
                tta = np.tile(neword_tL*180/pi, len(tto))
                psi = np.tile(neword_pL*180/pi, len(tto))

                [Gs, Go, ks, ko, bf, sob, sof] = weighted_sum_over_lidf_vec(lidf, tta, tto, psi)
                CIo = CIxy(flag, tto, CI_thres)
                CIs = CIxy(flag, tta, CI_thres)
                [Ps_arr, Po_arr, int_res_arr, nl] = dir_gap_initial_vec(tta, tto, psi, ks, ko, CIs, CIo)
               
            pars.append([tta, tto, psi, ks, ko, sob, sof, CIs, CIo, Ps_arr, Po_arr, int_res_arr, nl])
    """
    """
    Vectorize the parameters calculation
    """
    tL = list((conv1_tL*xx + conv2_tL)*180/pi) 
    pL = list((conv1_pL*xx + conv2_pL)*180/pi) 
    
    angle_arr = np.array(list(itertools.product(tL, tto, pL)))
    tta, tto, psi = angle_arr[:,0], angle_arr[:,1], angle_arr[:,2]
    [Ga, Go, ks, ko, bf, sob, sof] = weighted_sum_over_lidf_vec(lidf, tta, tto, psi)
    CIs = CIxy(flag, tta, CI_thres)
    CIo = CIxy(flag, tto, CI_thres) 
    
    return [tta, tto, psi, ks, ko, sob, sof, CIs, CIo] 

def hemi_dif_initial(flag, lidf, CI_thres):    
    xx=np.array([0.9602898565, -0.9602898565, 0.7966664774, -0.7966664774, 0.5255324099, -0.5255324099, 0.1834346425, -0.1834346425])
    
    ww=np.array([0.1012285363,  0.1012285363, 0.2223810345,  0.2223810345, 0.3137066459,  0.3137066459, 0.3626837834,  0.3626837834])   

    # * define limits of integration and the convertion factors for integration
    # * over thetaL (note the tL suffix!)
    upperlimit_mL = pi/2.0
    lowerlimit_mL = 0.0
    conv1_mL = (upperlimit_mL-lowerlimit_mL)/2.0
    conv2_mL = (upperlimit_mL+lowerlimit_mL)/2.0
        
    #   * define limits of integration and the convertion factors for integration
    # * over phiL (note the pL suffix!)
    upperlimit_nL = 2.0*pi
    lowerlimit_nL = 0.0
    conv1_nL = (upperlimit_nL-lowerlimit_nL)/2.0
    conv2_nL = (upperlimit_nL+lowerlimit_nL)/2.0
    
    # * define limits of integration and the convertion factors for integration
    # * over thetaL (note the tL suffix!)
    upperlimit_tL = np.pi/2.0
    lowerlimit_tL = 0.0
    conv1_tL = (upperlimit_tL-lowerlimit_tL)/2.0
    conv2_tL = (upperlimit_tL+lowerlimit_tL)/2.0    
    
    # * define limits of integration and the convertion factors for integration
    # * over phiL (note the pL suffix!)
    upperlimit_pL = 2.0*pi
    lowerlimit_pL = 0.0
    conv1_pL = (upperlimit_pL-lowerlimit_pL)/2.0
    conv2_pL = (upperlimit_pL+lowerlimit_pL)/2.0  

    """
    pars = [] 
    for m in range(0,8):
        neword_mL = conv1_mL*xx[m] + conv2_mL 
        for n in range(0,8):     
            neword_nL = conv1_nL*xx[n] + conv2_nL 
            for i in range(0,8): 
                neword_tL = conv1_tL*xx[i] + conv2_tL  
                for j in range(0,8): 
                    neword_pL  = conv1_pL*xx[j] + conv2_pL
                    
                    tts = neword_mL*180/pi
                    tto = neword_tL*180/pi
                    psi = abs(neword_nL*180/pi-neword_pL*180/pi) 

                    [Ga, Go, ks, ko, bf, sob, sof] = weighted_sum_over_lidf(lidf, tts, tto, psi)
                    #CIs = CIxy(CIy1, CIy2, tts[i], CI_thres)
                    #CIo = CIxy(CIy1, CIy2, tto[j], CI_thres)  
                    CIs = CIxy(flag, tts, CI_thres)
                    CIo = CIxy(flag, tto, CI_thres) 
                    [Ps_arr, Po_arr, int_res_arr, nl] = dir_gap_initial(tts, tto, psi, ks, ko, CIs, CIo)
                    pars.append([tts, tto, psi, ks, ko, sob, sof, CIs, CIo, Ps_arr, Po_arr, int_res_arr, nl])  
                    #print([tts, tto, psi, ks, ko, CIs, CIo, sob, sof])
                    #pars.append([tts, tto, psi, ks, ko, CIs, CIo, sob, sof])  
    """
     
    """
    Vectorize the parameters calculation
    """
    mL = list((conv1_mL*xx + conv2_mL)*180/pi)
    nL = list((conv1_nL*xx + conv2_nL)*180/pi)
    tL = list((conv1_tL*xx + conv2_tL)*180/pi) 
    pL = list((conv1_pL*xx + conv2_pL)*180/pi) 
    
    angle_arr = np.array(list(itertools.product(mL, nL, tL, pL)))
    tts, tto, psi = angle_arr[:,0], angle_arr[:,2], abs(angle_arr[:,1]-angle_arr[:,3])
    [Ga, Go, ks, ko, bf, sob, sof] = weighted_sum_over_lidf_vec(lidf, tts, tto, psi)
    CIs = CIxy(flag, tts, CI_thres)
    CIo = CIxy(flag, tto, CI_thres) 
        
    return [tts, tto, psi, ks, ko, sob, sof, CIs, CIo]
 
#%% 5) Calculation of SZA, SAA
def calc_sun_angles(lat, lon, stdlon, doy, ftime):
    """Calculates the Sun Zenith and Azimuth Angles (SZA & SAA).
    Parameters
    ----------
    lat : float
        latitude of the site (degrees).
    long : float
        longitude of the site (degrees).
    stdlng : float
        central longitude of the time zone of the site (degrees).
    doy : float
        day of year of measurement (1-366).
    ftime : float
        time of measurement (decimal hours).
    Returns
    -------
    sza : float
        Sun Zenith Angle (degrees).
    saa : float
        Sun Azimuth Angle (degrees).
    """

    lat, lon, stdlon, doy, ftime = map(
        np.asarray, (lat, lon, stdlon, doy, ftime))
    # Calculate declination
    declination = 0.409 * np.sin((2.0 * np.pi * doy / 365.0) - 1.39)
    EOT = 0.258 * np.cos(declination) - 7.416 * np.sin(declination) - \
          3.648 * np.cos(2.0 * declination) - 9.228 * np.sin(2.0 * declination)
    LC = (stdlon - lon) / 15.
    time_corr = (-EOT / 60.) + LC
    solar_time = ftime - time_corr
    # Get the hour angle
    w = np.asarray((solar_time - 12.0) * 15.)
    # Get solar elevation angle
    sin_thetha = np.cos(np.radians(w)) * np.cos(declination) * np.cos(np.radians(lat)) + \
                 np.sin(declination) * np.sin(np.radians(lat))
    sun_elev = np.arcsin(sin_thetha)
    # Get solar zenith angle
    sza = np.pi / 2.0 - sun_elev
    sza = np.asarray(np.degrees(sza))
    # Get solar azimuth angle
    cos_phi = np.asarray(
        (np.sin(declination) * np.cos(np.radians(lat)) -
         np.cos(np.radians(w)) * np.cos(declination) * np.sin(np.radians(lat))) /
        np.cos(sun_elev))
    saa = np.zeros(sza.shape)
    saa[w <= 0.0] = np.degrees(np.arccos(cos_phi[w <= 0.0]))
    saa[w > 0.0] = 360. - np.degrees(np.arccos(cos_phi[w > 0.0]))
    return np.asarray(sza), np.asarray(saa)