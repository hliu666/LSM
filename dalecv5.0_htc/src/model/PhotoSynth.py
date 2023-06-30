# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 11:06:47 2022

@author: 16072
"""
import numpy as np
from scipy import optimize

def Respiration(meteo, Photo_Pars):
    R     = 8.314             # [J mol-1K-1]   Molar gas constant
    
    [Vcmax25, BallBerrySlope] = Photo_Pars 
    RdPerVcmax25 = 0.015
    Rd25  = RdPerVcmax25 * Vcmax25
    
    T     = meteo[2] + 273.15 # convert temperatures to K if not already
    Tref  = 25.0     + 273.15 # [K] absolute temperature at 25 oC
    
    # temperature correction for Rd
    delHaR     = 46390  #Unit is  [J K^-1]
    delSR      = 490    #Unit is [J mol^-1 K^-1]
    delHdR     = 150650 #Unit is [J mol^-1]
    fTv         = temperature_functionC3(Tref,R,T,delHaR)
    fHTv        = high_temp_inhibtionC3(Tref,R,T,delSR,delHdR)
    f_Rd        = fTv * fHTv
     
    stressfactor  = 1
    Rd    = Rd25   * f_Rd    * stressfactor
    
    return -Rd

def PhotoSynth(meteo, Photo_Pars):
    
    rhoa  = 1.2047            # [kg m-3]      Specific mass of air   
    Mair  = 28.96             # [g mol-1]     Molecular mass of dry air
    R     = 8.314             # [J mol-1K-1]   Molar gas constant
    
    Q     = meteo[0]          # [umol m-2 s-1] absorbed PAR flux
    Cs    = meteo[1]
    T     = meteo[2] + 273.15 # convert temperatures to K if not already
    eb    = meteo[3] 
    O     = meteo[4] 
    p     = meteo[5] 

    #biochemical
    Type            = "C3"
    [Vcmax25, BallBerrySlope] = Photo_Pars 
    #Vcmax25         = 128.0 # [umol m-2 s-1] maximum carboxylation capacity (at optimum temperature of 25C, former Vcmo) 
    #BallBerrySlope  = 13.1  # slope of Ball-Berry stomatal conductance model (former m)
    RdPerVcmax25    = 0.015
    BallBerry0      = 0.01  # intercept of Ball-Berry stomatal conductance model
    Tref            = 25 + 273.15    # [K]           absolute temperature at 25 oC

    Kc25            = 405    # [umol mol-1]
    Ko25            = 279    # [mmol mol-1]
    spfy25          = 2444   # specificity (Computed from Bernacchhi et al 2001 paper)

    # convert all to bar: CO2 was supplied in ppm, O2 in permil, and pressure in mBar
    ppm2bar     = 1E-6 * (p *1E-3)
    Cs          = Cs *ppm2bar
    if Type == "C3":
        O       = (O * 1e-3) * (p *1E-3)    
    elif Type == "C4":
        O       = 0  # force O to be zero for C4 vegetation (this is a trick to prevent oxygenase)
    
    Kc25         = Kc25 * 1e-6 
    Ko25         = Ko25 * 1e-3 
    Gamma_star25 = 0.5  * O/spfy25    # [ppm] compensation point in absence of Rd
    Rd25         = RdPerVcmax25 * Vcmax25
    if Type == "C3":
        effcon =  1/5
    elif Type == "C4":
        effcon = 1/6
    atheta     = 0.8 

    # Mesophyll conductance: by default we ignore its effect
    #  so Cc = Ci - A/gm = Ci
    g_m = float("inf")
    #if isfield(leafbio, 'g_m')
    #    g_m = leafbio.g_m * 1e6 % convert from mol to umol
    stressfactor  = 1
    
    # fluorescence
    leafbio_Kn0, leafbio_Knalpha, leafbio_Knbeta = 2.48, 2.83, 0.114
    Knparams    = [leafbio_Kn0, leafbio_Knalpha, leafbio_Knbeta]
    Kf          = 0.05    # []            rate constant for fluorescence
    Kd          = max(0.8738,  0.0301*(T-273.15)+ 0.0773)
    Kp          = 4.0     # []            rate constant for photochemisty

    if Type == "C4":
        # RdPerVcmax25 = 0.025  % Rd25 for C4 is different than C3
        #   Rd25 = RdPerVcmax25 * Vcmax25
        # Constant parameters for temperature correction of Vcmax
        print("Pending.......")
        """
        Q10 = leafbio.TDP.Q10                           # Unit is  []
        s1  = leafbio.TDP.s1                            # Unit is [K]
        s2  = leafbio.TDP.s2                            # Unit is [K^-1]
        s3  = leafbio.TDP.s3                            # Unit is [K]
        s4  = leafbio.TDP.s4                            # Unit is [K^-1]
        
        # Constant parameters for temperature correction of Rd
        s5  = leafbio.TDP.s5                            # Unit is [K]
        s6  = leafbio.TDP.s6                            # Unit is [K^-1]
        
        fHTv = 1 + np.exp(s1*(T - s2))
        fLTv = 1 + np.exp(s3*(s4 - T))
        Vcmax = (Vcmax25 * Q10**(0.1*(T-Tref)))/(fHTv * fLTv)#Temp Corrected Vcmax
        
        # Temperature correction of Rd
        
        fHTv = 1 + np.exp(s5*(T - s6))
        Rd = (Rd25 * Q10**(0.1*(T-Tref)))/fHTv # Temp Corrected Rd
        # Temperature correction of Ke
        Ke25 = 20000 * Vcmax25 # Unit is  []
        Ke = (Ke25 * Q10**(0.1*(T-Tref)))# Temp Corrected Ke  
        """
    elif Type == "C3":

        # temperature correction of Vcmax
        delHaV     = 65330  #Unit is  [J K^-1]
        delSV      = 485    #Unit is [J mol^-1 K^-1]
        delHdV     = 149250 #Unit is [J mol^-1]
        fTv         = temperature_functionC3(Tref,R,T,delHaV)
        fHTv        = high_temp_inhibtionC3(Tref,R,T,delSV,delHdV)
        f_Vcmax     = fTv * fHTv
        
        # temperature correction for Rd
        delHaR     = 46390  #Unit is  [J K^-1]
        delSR      = 490    #Unit is [J mol^-1 K^-1]
        delHdR     = 150650 #Unit is [J mol^-1]
        fTv         = temperature_functionC3(Tref,R,T,delHaR)
        fHTv        = high_temp_inhibtionC3(Tref,R,T,delSR,delHdR)
        f_Rd        = fTv * fHTv
        
        # temperature correction for Kc
        delHaKc     = 79430		# Unit is  [J K^-1]
        fTv         = temperature_functionC3(Tref,R,T,delHaKc)
        f_Kc        = fTv
        
        # temperature correction for Ko
        delHaKo     = 36380     # Unit is  [J K^-1]
        fTv         = temperature_functionC3(Tref,R,T,delHaKo)
        f_Ko        = fTv
        
        # temperature correction for Gamma_star
        delHaT      = 37830     # Unit is  [J K^-1]
        fTv         = temperature_functionC3(Tref,R,T,delHaT)
        f_Gamma_star= fTv
        
        Ke          = 1 # dummy value (only needed for C4)
    
    if Type == "C3":
        Vcmax       = Vcmax25* f_Vcmax * stressfactor
        Rd          = Rd25   * f_Rd    * stressfactor
        Kc          = Kc25   * f_Kc
        Ko          = Ko25   * f_Ko
    
    Gamma_star   = Gamma_star25 * f_Gamma_star
    # calculation of potential electron transport rate
    po0         = Kp/(Kf+Kd+Kp)    # maximum dark photochemistry fraction, i.e. Kn = 0 (Genty et al., 1989)
    Je          = 0.5*po0*Q        # potential electron transport rate (JAK: add fPAR)
    
    if Type == "C3":
        MM_consts = (Kc*(1+O/Ko))# Michaelis-Menten constants
        Vs_C3 = (Vcmax/2)
        #  minimum Ci (as fraction of Cs) for BallBerry Ci. (If Ci_input is present we need this only as a placeholder for the function call)
        minCi = 0.3
    elif Type == "C4":
        MM_consts = 0# just for formality, so MM_consts is initialized
        Vs_C3 = 0    #  the same
        minCi = 0.1  # C4
        
    """
    % compute Ci using iteration (JAK)
    % it would be nice to use a built-in root-seeking function but fzero requires scalar inputs and outputs,
    % Here I use a fully vectorized method based on Brent's method (like fzero) with some optimizations.
    """
    RH = min(1, eb/satvap(T-273.15)) #jak: don't allow "supersaturated" air! (esp. on T curves)    
    if BallBerry0==0:
        Ci = max(minCi * Cs,  Cs*(1-1.6/(BallBerrySlope * RH))) 
    else:
        #a = 1*1E-4
        #b = 10*1E-4
        #fa = opt_Ci(a, Cs, RH, BallBerrySlope, BallBerry0, minCi, Vcmaxt, Gammast, MM_consts, effcon, Jk, Rdt)
        #fb = opt_Ci(b, Cs, RH, BallBerrySlope, BallBerry0, minCi, Vcmaxt, Gammast, MM_consts, effcon, Jk, Rdt)
        #if fa*fb < 0:
        #    Ci = optimize.brentq(opt_Ci, 1*1E-4, 10*1E-4, (Cs, RH, BallBerrySlope, BallBerry0, minCi, Vcmaxt, Gammast, MM_consts, effcon, Jk, Rdt), 1E-7)
        #else:
        #    Ci = max(minCi * Cs,  Cs*(1-1.6/(BallBerrySlope * RH)))
        A_Pars  = [Type, g_m, Vs_C3, MM_consts, Rd, Vcmax, Gamma_star, Je, effcon, atheta, Ke]  
        Ci_Pars = [Cs, RH, minCi, BallBerrySlope, BallBerry0, ppm2bar]  
        
        Ci = optimize.brentq(opt_Ci, 0, 1, (A_Pars, Ci_Pars), 1E-6)

    [A, biochem_out] = Compute_A(Ci, Type, g_m, Vs_C3, MM_consts, Rd, Vcmax, Gamma_star, Je, effcon, atheta, Ke)
    #A = max(A, 0.0)
    [Ag, Vc, Vs, Ve, CO2_per_electron] = biochem_out
    gs  = max(0.01, 1.6*A*ppm2bar/(Cs-Ci)) # stomatal conductance
    Ja  = Ag/CO2_per_electron       # actual electron transport rate
    rcw = (rhoa/(Mair*1E-3))/gs     # stomatal resistance

    Ci = Ci/ppm2bar
    return rcw, Ci, A

def opt_Ci(x, A_Pars, Ci_Pars):
    [Type, g_m, Vs_C3, MM_consts, Rd, Vcmax, Gamma_star, Je, effcon, atheta, kpepcase] = A_Pars
    [Cs, RH, minCi, BallBerrySlope, BallBerry0, ppm2bar] = Ci_Pars
    A = Compute_A(x, Type, g_m, Vs_C3, MM_consts, Rd, Vcmax, Gamma_star, Je, effcon, atheta, kpepcase)[0]
    return Ci_next(x, A, Cs, RH, minCi, BallBerrySlope, BallBerry0, ppm2bar) 

## Test-function for iteration
#   (note that it assigns A in the function's context.)
#   As with the next section, this code can be read as if the function body executed at this point.
#    (if iteration was used). In other words, A is assigned at this point in the file (when iterating).
def Ci_next(Ci_in, A, Cs, RH, minCi, BallBerrySlope, BallBerry0, ppm2bar):
    # compute the difference between "guessed" Ci (Ci_in) and Ci computed using BB after computing A
    A_bar = A * ppm2bar
    Ci_out = BallBerry(Cs, RH, A_bar, BallBerrySlope, BallBerry0, minCi)#[Ci_out, gs]
    err = Ci_out - Ci_in 
    return err

def Compute_A(Ci, Type, g_m, Vs_C3, MM_consts, Rd, Vcmax, Gamma_star, Je, effcon, atheta, kpepcase):
    if Type == "C3":
        Vs = Vs_C3 # = (Vcmax25/2) .* exp(log(1.8).*qt)    % doesn't change on iteration.
        if not np.isinf(g_m):
            Vc = sel_root(1/g_m, -(MM_consts + Ci +(Rd + Vcmax)/g_m), Vcmax*(Ci-Gamma_star+Rd/g_m), -1)
            Ve = sel_root(1/g_m, -(Ci + 2*Gamma_star +(Rd + Je * effcon)/g_m), Je*effcon*(Ci-Gamma_star+Rd/g_m), -1)
            CO2_per_electron = Ve/Je
        else:
            Vc          = Vcmax*(Ci-Gamma_star)/(MM_consts + Ci)# MM_consts = (Kc .* (1+O./Ko)) % doesn't change on iteration.
            CO2_per_electron = ((Ci-Gamma_star)/(Ci+2*Gamma_star))* effcon
            Ve          = Je * CO2_per_electron
    
    elif Type == "C4":
        Vc          = Vcmax
        Vs          = kpepcase*Ci
        CO2_per_electron = effcon # note: (Ci-Gamma_star)./(Ci+2*Gamma_star) = 1 for C4 (since O = 0) this line avoids 0/0 when Ci = 0
        Ve          = Je * CO2_per_electron
    
    V           = sel_root(atheta,-(Vc+Ve),Vc*Ve, np.sign(-Vc))# i.e. sign(Gamma_star - Ci)
    Ag          = sel_root(0.98,  -(V+Vs), V*Vs, -1)
    A           = Ag - Rd 
    
    return [A, [Ag, Vc, Vs, Ve, CO2_per_electron]] 

def BallBerry(Cs, RH, A, BallBerrySlope, BallBerry0, minCi):
    gs = gsFun(Cs, RH, A, BallBerrySlope, BallBerry0)
    Ci = max(minCi * Cs,  Cs - 1.6 * A/gs) 
    return Ci

def gsFun(Cs, RH, A, BallBerrySlope, BallBerry0):
    # add in a bit just to avoid div zero. 1 ppm = 1e-6 (note since A < 0 if Cs ==0, it gives a small gs rather than maximal gs
    gs = max(BallBerry0,  BallBerrySlope* A * RH / (Cs+1E-9)  + BallBerry0)
    return gs
    
def satvap(T):   
    """
    % calculates the saturated vapour pressure at 
    % temperature T (degrees C)
    % and the derivative of es to temperature s (kPa/C)
    % the output is in mbar or hPa. The approximation formula that is used is:
    % es(T) = es(0)*10^(aT/(b+T))
    % where es(0) = 6.107 mb, a = 7.5 and b = 237.3 degrees C
    % and s(T) = es(T)*ln(10)*a*b/(b+T)^2
    """
    # constants
    a           = 7.5
    b           = 237.3         #degrees C
    # calculations
    es          = 6.107*10**(7.5*T/(b+T))
    return es     
    
  
"""    
quadratic formula, root of least magnitude
"""
def sel_root(a,b,c, dsign):
    #  sel_root - select a root based on the fourth arg (dsign = discriminant sign)
    #    for the eqn ax^2 + bx + c,
    #    if dsign is:
    #       -1, 0: choose the smaller root
    #       +1: choose the larger root
    #  NOTE: technically, we should check a, but in biochemical, a is always > 0
    if a == 0:  # note: this works because 'a' is a scalar parameter!
        x = -c/b
    else:
        if dsign:
            dsign = -1 # technically, dsign==0 iff b = c = 0, so this isn't strictly necessary except, possibly for ill-formed cases)
        x = (-b + dsign* np.sqrt(b**2 - 4*a*c))/(2*a)
    
    return x      
    
## Temperature Correction Functions
# The following two functions pertains to C3 photosynthesis
def temperature_functionC3(Tref,R,T,deltaHa):
    # Temperature function
    tempfunc1 = (1 - Tref/T)
    fTv = np.exp(deltaHa/(Tref*R)*tempfunc1)
    return fTv

def high_temp_inhibtionC3(Tref,R,T,deltaS,deltaHd):
    # High Temperature Inhibition Function
    hightempfunc_num = (1+np.exp((Tref*deltaS-deltaHd)/(Tref*R)))
    hightempfunc_deno = (1+np.exp((deltaS*T - deltaHd)/(R*T)))
    fHTv = hightempfunc_num / hightempfunc_deno
    return fHTv    
    
    
    
    
    
    
    