# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 16:19:40 2022

@author: 16072
"""
import numpy as np
from scipy import integrate

#==============================================================================
# List of constants used in the netRadiation Module
#==============================================================================
# Stephan Boltzmann constant (W m-2 K-4)
SB = 5.670373e-8

def CalcStephanBoltzmann(T_K):
    '''Calculates the total energy radiated by a blackbody.
    
    Parameters
    ----------
    T_K : float
        body temperature (Kelvin).
    
    Returns
    -------
    M : float
        Emitted radiance (W m-2).'''
        
    M = SB * T_K ** 4
    return M

def Planck(wl,Tb):
    c1 = 1.191066e-22
    c2 = 14388.33
    em = np.ones(Tb.shape)
    Lb = em* c1*(wl*1e-9)**(-5)/(np.exp(c2/(wl*1e-3*Tb))-1)
    return Lb

#%% 1) Parameters of energy balance model    
def calc_extinction_coeff(lai, cik, sum0):
    ia = np.exp(-cik*lai)
    sum_tL = np.sum(sum0*ia)
    return -np.log(sum_tL)/lai

def calc_ebal_sunsha(dC, x, lai):
    ks = dC.ks[x]
    ko = calc_extinction_coeff(lai, dC.extinc_k, dC.extinc_sum0)

    fsun_ = -1/(ks+ko)*(np.exp(-(ks+ko)*lai)-1)*ko
    fsha_ = -1/(ko)*(np.exp(-(ko)*lai)-1)*ko - (-1/(ks+ko)*(np.exp(-(ks+ko)*lai)-1))*ko 
    fsun  = max(fsun_/(fsun_+fsha_), 1e-6)
    fsha  = max(fsha_/(fsun_+fsha_), 1e-6)
    
    return [fsun, fsha, ko]

def calc_ebal_canopy_pars(dC, x, Ls, ebal_rtm_pars):
    rhos, taus = dC.leaf
    rho,  tau  = rhos[:,x%365], taus[:,x%365]
    w = rho + tau   #leaf single scattering albedo
 
    [t1, t3, t4, t5, t12, t16] = dC.atmoMs
    
    [rsd, rdd, rs, A_tot, A_dif, i0, iD] = ebal_rtm_pars
    
    # radiation fluxes, downward and upward (these all have dimenstion [nwl]
    # first calculate hemispherical reflectances rsd and rdd according to SAIL
    # these are assumed for the reflectance of the surroundings
    # rdo is computed with SAIL as well
    # assume Fd of surroundings = 0 for the momemnt
    # initial guess of temperature of surroundings from Ta;
    Fd    = np.zeros(dC.wl.shape)
    Esun_ = np.pi*t1*t4
    Esun_[Esun_ < 1E-6] = 1E-6
    Esky_ = np.pi/(1-t3*rdd)*(t1*(t5+t12*rsd)+Fd+(1-rdd)*Ls*t3+t16)
    Esky_[Esky_ < 1E-6] = 1E-6
    #Esun_ = max(1E-6,np.pi*t1*t4)
    #Esky_ = max(1E-6,np.pi/(1-t3*rdd)*(t1*(t5+t12*rsd*1.5)+Fd+(1-rdd)*Ls*t3+t16))

    fEsuno,fEskyo,fEsunt,fEskyt = 0*Esun_,0*Esun_,0*Esun_,0*Esun_
    
    epsc = 1 - w
    epss = 1 - rs
    
    A_sun_sun = i0*epsc
    A_sha_sun = (A_tot - i0*epsc)
    A_sha_sha = A_dif
    
    return [Esun_, Esky_, fEsuno, fEskyo, fEsunt, fEskyt, A_sun_sun, A_sha_sun, A_sha_sha, epss]
#%% 2) Net radiation    
def ephoton(lambdas):
    #E = phot2e(lambda) calculates the energy content (J) of 1 photon of
    #wavelength lambda (m)

    h       = 6.6262E-34           # [J s]         Planck's constant
    c       = 299792458            # [m s-1]       speed of light
    E       = h*c/lambdas          # [J]           energy of 1 photon
    return E 
    
def e2phot(lambdas, E):
    #molphotons = e2phot(lambda,E) calculates the number of moles of photons
    #corresponding to E Joules of energy of wavelength lambda (m)
    A           = 6.02214E+23 #Constant of Avogadro
    e           = ephoton(lambdas)
    photons     = E/e
    molphotons  = photons/A
    return molphotons
    
def calc_netrad_pars(dC, x, lai, SW, L, ebal_sunsha_pars, ebal_canopy_pars):
    [fsun, fsha, _] = ebal_sunsha_pars 
    [Esun_, Esky_, fEsuno, fEskyo, fEsunt, fEskyt, A_sun_sun, A_sha_sun, A_sha_sha, epss] = ebal_canopy_pars
    wl = dC.wl
    
    """
    shortwave radiantion 
    """
    if (dC.tts[x] < 75) and (lai > 0.5): 
        Esunto          = 0.001 * integrate.simpson(Esun_[0:2006],wl[0:2006])
        Eskyto          = 0.001 * integrate.simpson(Esky_[0:2006],wl[0:2006])
        Etoto           = Esunto + Eskyto                   #Calculate total fluxes
        fEsuno[0:2006]  = Esun_[0:2006]/Etoto
        fEskyo[0:2006]  = Esky_[0:2006]/Etoto
        #J_o             = wl<3000;                          #find optical spectrum
        #Esunto          = 0.001 * Sint(Esun_(J_o),wl(J_o)); #Calculate optical sun fluxes (by Integration), including conversion mW >> W
        #Eskyto          = 0.001 * Sint(Esky_(J_o),wl(J_o)); #Calculate optical sun fluxes (by Integration)
        #Etoto           = Esunto + Eskyto;                  #Calculate total fluxes
        #fEsuno(J_o)     = Esun_(J_o)/Etoto;                 #fraction of contribution of Sun fluxes to total light
        #fEskyo(J_o)     = Esky_(J_o)/Etoto;                 #fraction of contribution of Sky fluxes to total light
        
        Esun_[0:2006] = fEsuno[0:2006]*SW
        Esky_[0:2006] = fEskyo[0:2006]*SW
        
        """
        Calculate APAR
        """
        Ipar  = 301
        wlPAR = wl[0:Ipar]
        
        Pnsun  = 0.001 * integrate.simpson(e2phot(wlPAR*1E-9,Esun_[0:Ipar]*A_sun_sun[0:Ipar]),wlPAR)
        Pndir  = Pnsun * 1E6 # 
        Pnsky  = 0.001 * integrate.simpson(e2phot(wlPAR*1E-9,(Esky_*A_sha_sha+Esun_*A_sha_sun)[0:Ipar]),wlPAR)
        Pndif  = Pnsky * 1E6 
        
        APARu = Pndir + Pndif*fsun
        APARh = Pndif*fsha

    else:
        Esun_[0:2006]  = 0.0
        Esky_[0:2006]  = 0.0

        APARu = 0.0
        APARh = 0.0
        
    """
    longwave radiantion 
    """        
    Esuntt         = 0.001 * integrate.simpson(Esun_[2006:],wl[2006:])
    Eskytt         = 0.001 * integrate.simpson(Esky_[2006:],wl[2006:])
    Etott          = Eskytt + Esuntt
    fEsunt[2006:]  = Esun_[2006:]/Etott
    fEskyt[2006:]  = Esky_[2006:]/Etott    
    #J_t             = wl>=3000;                         %find thermal spectrum
    #Esuntt          = 0.001 * Sint(Esun_(J_t),wl(J_t)); %Themal solar fluxes
    #Eskytt          = 0.001 * Sint(Esky_(J_t),wl(J_t)); %Thermal Sky fluxes
    #Etott           = Eskytt + Esuntt;                  %Total
    #fEsunt(J_t)     = Esun_(J_t)/Etott;                 %fraction from Esun
    #fEskyt(J_t)     = Esky_(J_t)/Etott;                 %fraction from Esky
  
    Esun_[2006:] = fEsunt[2006:]*L
    Esky_[2006:] = fEskyt[2006:]*L
    
    Rndir = 0.001 * integrate.simpson(Esun_*A_sun_sun,wl)
    Rndif = 0.001 * integrate.simpson((Esky_*A_sha_sha+Esun_*A_sha_sun),wl)   # Full spectrum net diffuse flux

    ERnuc = Rndir + Rndif*fsun
    ERnhc = Rndif*fsha

    #soil layer, direct and diffuse radiation
    Rsdir  = 0.001 * integrate.simpson(Esun_*epss,wl) # Full spectrum net diffuse flux
    Rsdif_ = (Esky_*(1-A_sha_sha)+Esun_*(1-A_sun_sun-A_sha_sun))*epss
    Rsdif  = 0.001 * integrate.simpson(Rsdif_, wl) # Full spectrum net diffuse flux

    ERnus = Rsdir + Rsdif        # [1] Absorbed solar flux by the soil
    ERnhs = Rsdif                # [1] Absorbed diffuse downward flux by the soil (W m-2)
 
    return [ERnuc, ERnhc, ERnus, ERnhs, APARu, APARh], [Esun_, Esky_]
 
def calc_netrad(dC, x, lai, L, T_Pars, net_rads, ebal_rtm_pars, ebal_sunsha_pars):
    [Ta, Tcu, Tch, Tsu, Tsh] = T_Pars
    
    ks = dC.ks[x]
    [_, _, ko] = ebal_sunsha_pars 
    
    [rsd, rdd, rs, A_tot, A_dif, i0, iD] = ebal_rtm_pars
    p  = 1 - iD/lai 
    
    ERnuc, ERnhc, ERnus, ERnhs, APARu, APARh = net_rads

    """
    longwave radiantion 
    """
    rho         = 0.01                    # [1]               Leaf/needle reflection
    tau         = 0.01                    # [1]               Leaf/needle transmission
    rs          = 0.06                    # [1]               Soil reflectance
    emisVeg     = 1-rho-tau               # [nwl]             Emissivity vegetation
    emisGrd     = 1-rs                    # [nwl]             Emissivity soil

    w = 1 - emisVeg
    ed, eu = iD/(2*lai), iD/(2*lai)
    
    rc1 = w*ed/(1-w*p)
    rc2 = w*eu/(1-w*p)
    
    Aup  = iD*emisVeg/(1-p*(1-emisVeg))
    Rdnc = (1-emisGrd)*iD/(1-rc2*(1-emisGrd)*iD)
    
    e1  = iD*emisVeg/(1-p*(1-emisVeg))
    e2  = (1-iD)*Rdnc*Aup
    e3  = iD*rc1*Rdnc*Aup
    
    Rdns = emisGrd/(1-(1-emisGrd)*iD*rc2)
    e4   = (1-iD)*Rdns
    e5   = iD*rc1*Rdns
    
    alphav = (e1 + e2 + e3)
    alphas = emisGrd + (e4 + e5)*(1-emisGrd) 

    Hcu = emisVeg*CalcStephanBoltzmann(Tcu+273.15)
    Hch = emisVeg*CalcStephanBoltzmann(Tch+273.15)
    Hsu = emisGrd*CalcStephanBoltzmann(Tsu+273.15)
    Hsh = emisGrd*CalcStephanBoltzmann(Tsh+273.15)

    #Asun   = integrate.simpson(Esun_*epsc,wl)*0.001 # Total absorbed solar radiation
    #Lndir  = 0.001 * integrate.simpson(Esun_[2006:]*i0*epsc[2006:],wl[2006:])
    #Lndif_ = Esky_[2006:]*A_dif[2006:]+Esun_[2006:]*(A[2006:]-i0*epsc[2006:]) #   Net (absorbed) radiation by leaves
    #Lndif  = 0.001 * integrate.simpson(Lndif_,wl[2006:])   # Full spectrum net diffuse flux
    
    #print(Lndir + Lndif*fsun,  fLsun*L)
    #print(Lndif*fsha, fLsha*L)
    #t1  = 0.001 * integrate.simpson(Esun_*i0*epsc,wl)
    #t0 = Esky_*A_dif+Esun_*(A-i0*epsc) #   Net (absorbed) radiation by leaves
    #t2  = 0.001 * integrate.simpson(t0,wl)   # Full spectrum net diffuse flux
    #print(t1, t2)
    #print(t1 + t2*fsun, fLsun*t2)
    
    """
    def updir(x, L, ks, ko):
        return Hcu*np.exp(-ks*x)*(1-np.exp(-ko*x)+1-np.exp(-ko*(L-x)))*ko
    def updif(x, L, ks, ko):
        return Hch*(1-np.exp(-ks*x))*(1-np.exp(-ko*x)+1-np.exp(-ko*(L-x)))*ko   
    print(integrate.quad(updir, 0, lai, args = (L, ks, ko))) 
    print(integrate.quad(updif, 0, lai, args = (L, ks, ko))) 
    print(fLsun*L) 
    print(fLsha*L) 
    

    def insun_nlu(x, ks, ko, nlai, nl):
        return np.exp(-ks*(nlai-x))*(1-np.exp(-ko*x))*ko/nl
    def insha_nlu(x, ks, ko, nlai, nl):
        return (1-np.exp(-ks*(nlai-x)))*(1-np.exp(-ko*x))*ko/nl
    
    def insun_nld(x, ks, ko, nlai, nl):
        return np.exp(-ks*(x+nlai))*(1-np.exp(-ko*x))*ko/nl
    def insha_nld(x, ks, ko, nlai, nl):
        return (1-np.exp(-ks*(x+nlai)))*(1-np.exp(-ko*x))*ko/nl
    
    def insun(x, Hcu, Hch, L, ks, ko):
        Hx  = (Hcu*np.exp(-ks*x)+Hch*(1-np.exp(-ks*x)))
        #return Hx*(integrate.quad(insun_nlu, 0, x,     args = (ks, ko, x, x))[0] +
        #           integrate.quad(insun_nld, 0, (L-x), args = (ks, ko, x, (L-x)))[0])
        return Hx*((1-np.exp(-ko*x))+(1-np.exp(-ko*(L-x))))*ko
    def insha(x, Hcu, Hch, L, ks, ko):
        Hx  = (Hcu*np.exp(-ks*x)+Hch*(1-np.exp(-ks*x)))
        #return Hx*(integrate.quad(insha_nlu, 0, x,     args = (ks, ko, x, x))[0] +
        #           integrate.quad(insha_nld, 0, (L-x), args = (ks, ko, x, (L-x)))[0])
        return Hx*((1-np.exp(-ko*x))+(1-np.exp(-ko*(L-x))))*ko
    """

    #def test(x,  ks):
    #    return (1-np.exp(-ks*x))/ks
    #print(integrate.quad(test, 0, lai, args = (ks)))
    #print((1-np.exp(-ks*lai))/ks)

    #inHcsun,_ = integrate.quad(insun, 0, lai, args = (Hcu, Hch, lai, ks, ko))
    #inHcsha,_ = integrate.quad(insha, 0, lai, args = (Hcu, Hch, lai, ks, ko))

    #fHssun_ = np.exp(-ks*lai)
    #fHssha_ = 1 - fHssun_
    
    """
    Reference:
        A Two-Big-Leaf Model for Canopy Temperature, Photosynthesis, and Stomatal Conductance
    """ 
    fHssun_ = -1/(ko-ks)*(np.exp(-ko*lai)-np.exp(-ks*lai))*ko
    fHssha_ = -1/(ko)*(np.exp(-ko*lai)-1)*ko-(-1/(ko-ks)*(np.exp(-ko*lai)-np.exp(-ks*lai))*ko)
    fHssun  = fHssun_/(fHssun_+fHssha_)
    fHssha  = fHssha_/(fHssun_+fHssha_)
    
    """
    def sun(x, Hcu, L, ks, ko):
        return Hcu*np.exp(-ks*x)*(np.exp(-ko*x)+np.exp(-ko*(L-x)))*ko
    def sha(x, Hch, L, ks, ko):
        return Hch*(1-np.exp(-ks*x))*(np.exp(-ko*x)+np.exp(-ko*(L-x)))*ko 
   
    #print(round((e1 + e2 + e3)*fHssun_/(fHssun_+fHssha_),4),
    #      round(emisVeg*fHssun_,4))
    Hcsun,_ = integrate.quad(sun, 0, lai, args = (Hcu, lai, ks, ko))
    Hcsha,_ = integrate.quad(sha, 0, lai, args = (Hch, lai, ks, ko))
    """
    Hcsun = Hcu*ko*(((1-np.exp(lai*-(ks+ko)))/(ks+ko))-((np.exp(-ks*lai)-np.exp(-ko*lai))/(ks-ko)))
    Hcsha = Hch*ko*(((np.exp(-ks*lai)-np.exp(-ko*lai))/(ks-ko))+((-1+np.exp(lai*-(ks+ko)))/(ks+ko))+(2*(1-np.exp(-ko*lai))/ko))
    
    Hs = Hsu*np.exp(-ks*lai) + Hsh*(1-np.exp(-ks*lai))
    ELnuc = fHssun*Hs*alphav-Hcsun
    ELnhc = fHssha*Hs*alphav-Hcsha
    #ELnuc = fHssun*alphav*(alphas*Hs-2*Hcu)
    #ELnhc = fHssha*alphav*(alphas*Hs-2*Hch)
    
    #Bonan's Method:
    #Hc = CalcStephanBoltzmann((Tcu+Tch)/2.0+273.15)
    #Lc = emisVeg*(L+Hs)*(1-np.exp(-ko*lai))-2*emisVeg*Hc*(1-np.exp(-ko*lai))
    #ELnuc = emisVeg*(L -Hc)*ko/(ko+ks)*(1-np.exp(-(ks+ko)*lai)) +\
    #        emisVeg*(Hs-Hc)*ko/(ko-ks)*(np.exp(-ks*lai)-np.exp(-ko*lai))
    #ELnhc = Lc-ELnuc
    
    """
    def down(x, Hcu, Hch, L, ks, ko):
        return (Hcu*np.exp(-ks*x)+Hch*(1-np.exp(-ks*x)))*np.exp(-ko*(L-x))*ko   
    def up(x, Hcu, Hch, L, ks, ko):
        return (Hcu*np.exp(-ks*x)+Hch*(1-np.exp(-ks*x)))*np.exp(-ko*x)*ko 
    
    Hcdown,_ = integrate.quad(down, 0, lai, args = (Hcu, Hch, lai, ks, ko))
    Hcup  ,_ = integrate.quad(up,   0, lai, args = (Hcu, Hch, lai, ks, ko))
    """
    Hcdown = ko*(ko*(np.exp(-ks*lai)*(Hch-Hcu)+Hcu*np.exp(-ko*lai)-Hch)+ks*Hch*(1-np.exp(-ko*lai)))/(ko*(ks-ko))
    
    ELnus = alphas*(Hcdown-Hsu)
    ELnhs = alphas*(Hcdown-Hsh)
    
    rad_Rnuc = ERnuc+ELnuc
    rad_Rnhc = ERnhc+ELnhc
    rad_Rnus = ERnus+ELnus
    rad_Rnhs = ERnhs+ELnhs
    
    Fc, Fs = max(i0/abs(np.log(1-i0)), 1e-6), max(1-i0, 1e-6)
    #Fc, Fs = fsun, fHssun_    
    return rad_Rnuc, rad_Rnhc, APARu, APARh, rad_Rnus, rad_Rnhs, Fc, Fs, i0, iD
#%% 3) Atmospheric Longwave radiation 
# ==============================================================================
# List of constants used in Atmospheric longwave radiation 
# ==============================================================================
# heat capacity of dry air at constant pressure (J kg-1 K-1)
c_pd = 1003.5
# heat capacity of water vapour at constant pressure (J kg-1 K-1)
c_pv = 1865
# ratio of the molecular weight of water vapor to dry air
epsilon = 0.622
# gas constant for dry air, J/(kg*degK)
R_d = 287.04
# acceleration of gravity (m s-2)
g = 9.8

def calc_emiss_atm(ea, t_a_k):
    '''Atmospheric emissivity
    Estimates the effective atmospheric emissivity for clear sky.
    Parameters
    ----------
    ea : float
        atmospheric vapour pressure (mb).
    t_a_k : float
        air temperature (Kelvin).
    Returns
    -------
    emiss_air : float
        effective atmospheric emissivity.
    References
    ----------
    .. [Brutsaert1975] Brutsaert, W. (1975) On a derivable formula for long-wave radiation
        from clear skies, Water Resour. Res., 11(5), 742-744,
        htpp://dx.doi.org/10.1029/WR011i005p00742.'''

    emiss_air = 1.24 * (ea / t_a_k)**(1. / 7.)  # Eq. 11 in [Brutsaert1975]_

    return emiss_air

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
    Lambda = 1E6 * (2.501 - (2.361e-3 * (T_A_K - 273.15)))
    return Lambda

def calc_mixing_ratio(ea, p):
    '''Calculate ratio of mass of water vapour to the mass of dry air (-)
    Parameters
    ----------
    ea : float or numpy array
        water vapor pressure at reference height (mb).
    p : float or numpy array
        total air pressure (dry air + water vapour) at reference height (mb).
    Returns
    -------
    r : float or numpy array
        mixing ratio (-)
    References
    ----------
    http://glossary.ametsoc.org/wiki/Mixing_ratio'''

    r = epsilon * ea / (p - ea)
    return r

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

def calc_lapse_rate_moist(T_A_K, ea, p):
    '''Calculate moist-adiabatic lapse rate (K/m)
    Parameters
    ----------
    T_A_K : float or numpy array
        air temperature at reference height (K).
    ea : float or numpy array
        water vapor pressure at reference height (mb).
    p : float or numpy array
        total air pressure (dry air + water vapour) at reference height (mb).
    Returns
    -------
    Gamma_w : float or numpy array
        moist-adiabatic lapse rate (K/m)
    References
    ----------
    http://glossary.ametsoc.org/wiki/Saturation-adiabatic_lapse_rate'''

    r = calc_mixing_ratio(ea, p)
    c_p = calc_c_p(p, ea)
    lambda_v = calc_lambda(T_A_K)
    Gamma_w = ((g * (R_d * T_A_K**2 + lambda_v * r * T_A_K)
               / (c_p * R_d * T_A_K**2 + lambda_v**2 * r * epsilon)))
    return Gamma_w

def calc_longwave_irradiance(ea, t_a_k, p=1013.25, z_T=2.0, h_C=2.0):
    '''Longwave irradiance
    Estimates longwave atmospheric irradiance from clear sky.
    By default there is no lapse rate correction unless air temperature
    measurement height is considerably different than canopy height, (e.g. when
    using NWP gridded meteo data at blending height)
    Parameters
    ----------
    ea : float
        atmospheric vapour pressure (mb).
    t_a_k : float
        air temperature (K).
    p : float
        air pressure (mb)
    z_T: float
        air temperature measurement height (m), default 2 m.
    h_C: float
        canopy height (m), default 2 m,
    Returns
    -------
    L_dn : float
        Longwave atmospheric irradiance (W m-2) above the canopy
    '''

    lapse_rate = calc_lapse_rate_moist(t_a_k, ea, p)
    t_a_surface = t_a_k - lapse_rate * (h_C - z_T)
    emisAtm = calc_emiss_atm(ea, t_a_surface)
    L_dn = emisAtm * CalcStephanBoltzmann(t_a_surface)
    return L_dn

#%% 4) Thermal irridiation 
def rtm_t(lai, L, i0, iD, wl, Tcu,Tch,Tsu,Tsh, dC, x, k_pars):
    """
    The top-of-canopy TIR radiance (TIR) at viewing angle 

    Returns
    -------
    None.

    """
    rho         = 0.01                    # [1]               Leaf/needle reflection
    tau         = 0.01                    # [1]               Leaf/needle transmission
    rs          = 0.06                    # [1]               Soil reflectance
    emisVeg     = 1-rho-tau               # [nwl]             Emissivity vegetation
    emisGrd     = 1-rs                    # [nwl]             Emissivity soil 
    w = rho + tau                         # [1]               leaf single scattering albedo
    
    rg = dC.soil
        
    CIs, CIo, sob, sof = dC.CIs[x], dC.CIo[x], dC.sob[x], dC.sof[x] 
    ko = dC.ko[x]  
    [kc, kg] = k_pars
    """
    reflected longwave radiation
    """  
    """
    p  = 1 - iD/lai
    wso  = sob*rho + sof*tau
    iv = 1 - np.exp(-ko * lai * CIo)
    rho2 = iv/2/lai
      
    BRFv = wso*kc/ko + i0*w*w*p*rho2/(1-p*w)      
    BRFs = kg*rg   
    
    L0   = L*(iD*(1-emisVeg)*BRFv + (1-iD)*(1-emisGrd)*BRFs)
    L0   = 0.001*integrate.simpson(L0,wl)
    """
    L0   = L*(iD*(1-emisVeg)*rho + (1-iD)*(1-emisGrd)*rs)
    
    """
    sip based thermal radiative transfer model
    """
    i  = max(1 - np.exp(-kc * lai * CIs), 0.00001)
    Fc, Fs = i/abs(np.log(1-i)), 1-i
    
    ed, eu = i/(2*lai), i/(2*lai)
    p  = 1 - i/lai 
    
    rc1 = w*ed/(1-w*p)
    rc2 = w*eu/(1-w*p)
    
    Aup  = i*emisVeg/(1-p*(1-emisVeg))
    Rdnc = (1-emisGrd)*i/(1-rc2*(1-emisGrd)*i)
    e1  = i*emisVeg/(1-p*(1-emisVeg))
    e2  = (1-i)*Rdnc*Aup
    e3  = i*rc1*Rdnc*Aup
    
    Rdns = emisGrd/(1-(1-emisGrd)*i*rc2)
    e4   = (1-i)*Rdns
    e5   = i*rc1*Rdns
    
    alphav = (e1 + e2 + e3)
    alphas = (e4 + e5)

    #print(round(Tcu,2), round(Tch,2), round(Tsu,2), round(Tsh,2))

    Hcu = CalcStephanBoltzmann(Tcu+273.15)
    Hch = CalcStephanBoltzmann(Tch+273.15)
    Hsu = CalcStephanBoltzmann(Tsu+273.15)
    Hsh = CalcStephanBoltzmann(Tsh+273.15)

    TIRv = Fc*Hcu*alphav + (1-Fc)*Hch*alphav
    TIRs = Fs*Hsu*alphas + (1-Fs)*Hsh*alphas 
    #TIRv = Fc*Planck(wl,Tcu)*alphav + (1-Fc)*Planck(wl,Tch)*alphav
    #TIRs = Fs*Planck(wl,Tsu)*alphas + (1-Fs)*Planck(wl,Tsh)*alphas

    TIRt = TIRv + TIRs + L0

    emis = alphav + alphas
    LST  = (TIRt/SB)**0.25 - 273.15
    
    Ts  = Fs*Tsu + (1-Fs)*Tsh
    Tc  = Fc*Tcu + (1-Fc)*Tch

    return LST

