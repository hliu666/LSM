import numpy as np
import pandas as pd 
from spectres import spectres

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

    return refl.flatten()

def run_model(pars):      

    Cab, Car, Cbrown, Cw, Ant = pars[0], pars[1], pars[2], pars[3], pars[4]
    
    Alpha = 600
    fLMA_k = 2519.65
    gLMA_k = -631.54
    gLMA_b = 0.0086
    Cm = 0.006518
    
    leaf = sip_leaf(Cab, Car, Cbrown, Cw, Cm, Ant, Alpha, fLMA_k, gLMA_k, gLMA_b)


    return leaf



