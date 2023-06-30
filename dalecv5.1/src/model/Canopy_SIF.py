# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 11:57:14 2022

@author: hliu
"""
from numpy import exp,  sin, pi, log, sqrt
import numpy as np
import scipy.special as sc
from scipy.interpolate import interp1d
import numpy.matlib
import mat4py
#%% 1) SIF at Leaf level
#PROSPECT calculations
def calctav(alfa,nr):

    rd          = pi/180
    n2          = nr**2
    nP          = n2+1
    nm          = n2-1
    a           = (nr+1)*(nr+1)/2
    k           = -(n2-1)*(n2-1)/4
    sa          = sin(alfa*rd)

    if alfa != 90:
        b1 = sqrt((sa**2-nP/2)*(sa**2-nP/2)+k)
    else:
        b1 = 0
    b2          = sa**2-nP/2.0
    b           = b1-b2
    b3          = b**3
    a3          = a**3
    ts          = (k**2/(6*b3)+k/b-b/2)-(k**2/(6*a3)+k/a-a/2)

    tp1         = -2*n2*(b-a)/(nP**2)
    tp2         = -2*n2*nP*log(b/a)/(nm**2);
    tp3         = n2*(1/b-1/a)/2
    tp4         = 16*n2**2*(n2**2+1)*log((2*nP*b-nm**2)/(2*nP*a-nm**2))/(nP**3*nm**2)
    tp5         = 16*n2**3*(1/(2*nP*b-nm**2)-1/(2*nP*a-nm**2))/(nP**3)
    tp          = tp1+tp2+tp3+tp4+tp5
    tav         = (ts+tp)/(2*sa**2)

    return tav

def sif_leaf(Cab, Kab, Cca, N, nr, Kall):
   
    ndub        = 15 #number of doublings applied
    Int         = 5  #
    fqe         = 0.01
    V2Z         = 0
    
    Kab  = Kab[0:2001]
    nr   = nr[0:2001]
    Kall = Kall[0:2001]
    
    optipar = mat4py.loadmat("Optipar_ProspectPRO_CX.mat")['optipar']
    
    if V2Z == -999:
        # Use old Kca spectrum if this is given as input
        Kca = np.array(optipar['Kca']).flatten()
    else:
        # Otherwise make linear combination based on V2Z
        # For V2Z going from 0 to 1 we go from Viola to Zea
        KcaV = np.array(optipar['KcaV']).flatten()
        KcaZ = np.array(optipar['KcaZ']).flatten()
        Kca  = (1-V2Z) * KcaV + V2Z * KcaZ 
        
    phi = np.array(optipar['phi']).flatten()  
    
    j           = np.where(Kall > 0)[0] # Non-conservative scattering (normal case)
    t1          = (1-Kall)*exp(-Kall)
    t2          = Kall**2*sc.exp1(Kall)
    tau         = np.ones(t1.shape)
    tau[j]      = t1[j] + t2[j]
    kChlrel, kCarrel = np.zeros(t1.shape), np.zeros(t1.shape) 
    kChlrel[j]  = Cab*Kab[j]/(Kall[j]*N)
    kCarrel[j]  = Cca*Kca[j]/(Kall[j]*N)

    talf        = calctav(59,nr)
    ralf        = 1-talf
    t12         = calctav(90,nr)
    r12         = 1-t12
    t21         = t12/(nr**2)
    r21         = 1-t21

    # top surface side
    denom       = 1-r21*r21*tau**2
    Ta          = talf*tau*t21/denom
    Ra          = ralf+r21*tau*Ta

    # bottom surface side
    t           = t12*tau*t21/denom
    r           = r12+r21*tau*t

    # Stokes equations to compute properties of next N-1 layers (N real)
    # Normal case

    D           = sqrt((1+r+t)*(1+r-t)*(1-r+t)*(1-r-t))
    rq          = r**2
    tq          = t**2
    a           = (1+rq-tq+D)/(2*r)
    b           = (1-rq+tq+D)/(2*t)

    bNm1        = b**(N-1)
    bN2         = bNm1**2
    a2          = a**2
    denom       = a2*bN2-1
    Rsub        = a*(bN2-1)/denom
    Tsub        = bNm1*(a2-1)/denom

    #			Case of zero absorption
    j           = np.where(r+t >= 1) 
    Tsub[j]     = t[j]/t[j]+(1-t[j]*(N-1));
    Rsub[j]	    = 1-Tsub[j]

    # Reflectance and transmittance of the leaf: combine top layer with next N-1 layers
    denom       = 1-Rsub*r
    tran        = Ta*Tsub/denom
    refl        = Ra+Ta*Rsub*t/denom

    # From here a new path is taken: The doubling method used to calculate
    # fluoresence is now only applied to the part of the leaf where absorption
    # takes place, that is, the part exclusive of the leaf-air interfaces. The
    # reflectance (rho) and transmittance (tau) of this part of the leaf are
    # now determined by "subtracting" the interfaces

    Rb  = (refl-ralf)/(talf*t21+(refl-ralf)*r21)  # Remove the top interface
    Z   = tran*(1-Rb*r21)/(talf*t21)              # Derive Z from the transmittance

    rho = (Rb-r21*Z**2)/(1-(r21*Z)**2)     # Reflectance and transmittance 
    tau = (1-Rb*r21)/(1-(r21*Z)**2)*Z      # of the leaf mesophyll layer
    t   = tau
    #r   =    max(rho,0)                    
    r   = np.array([0 if rho < 0 else rho for rho in rho])  # Avoid negative r

    # Derive Kubelka-Munk s and k

    I_rt     =   np.where((r+t) < 1)[0] 
    D[I_rt]  =   sqrt((1+r[I_rt]+t[I_rt]) * (1+r[I_rt]-t[I_rt]) * (1-r[I_rt]+t[I_rt]) * (1-r[I_rt]-t[I_rt]))
    a[I_rt]  =   ((1+r[I_rt]**2-t[I_rt]**2+D[I_rt])/(2*r[I_rt]))
    b[I_rt]  =   ((1-r[I_rt]**2+t[I_rt]**2+D[I_rt])/(2*t[I_rt]))

    mask = np.ones(len(r), np.bool)
    mask[I_rt] = 0
    a[mask] =   1
    b[mask] =   1

    s        =   r/t
    I_a      =   np.where(a >1 & ~np.isfinite(a))
    s[I_a]   =   2*a[I_a]/(a[I_a]**2-1)*log(b[I_a])

    k        =   log(b)
    k[I_a]   =   (a[I_a]-1)/(a[I_a]+1)*log(b[I_a])
    kChl     =   kChlrel * k

    # 3)  Fluorescence of the leaf mesophyll layer
    # Fluorescence part is skipped for fqe = 0

    #%light version. The spectral resolution of the irradiance is lowered.
    if fqe > 0:
        wle          = np.arange(400, 751) #';%spectral.wlE';    % excitation wavelengths, transpose to column
        spectral_wlP = np.arange(400, 2401)
        
        
        k_iwle      = interp1d(spectral_wlP,k)(wle)
        s_iwle      = interp1d(spectral_wlP,s)(wle)
        kChl_iwle   = interp1d(spectral_wlP,kChl)(wle)  
        r21_iwle    = interp1d(spectral_wlP,r21)(wle)  
        rho_iwle    = interp1d(spectral_wlP,rho)(wle)  
        tau_iwle    = interp1d(spectral_wlP,tau)(wle)  
        talf_iwle   = interp1d(spectral_wlP,talf)(wle)
        
        wlf         = np.arange(640, 851) #%spectral.wlF';    % fluorescence wavelengths, transpose to column
        wlp         = spectral_wlP  #PROSPECT wavelengths, kept as a row vector

        Iwlf        = np.in1d(wlp, wlf).nonzero()
        eps         = 2**(-ndub)

        # initialisations
        te          = 1-(k_iwle+s_iwle) * eps  
        tf          = 1-(k[Iwlf]+s[Iwlf]) * eps 
        re          = s_iwle * eps
        rf          = s[Iwlf] * eps

        sigmoid     = 1/(1+np.outer(exp(-wlf/10), exp(wle.T/10)))# matrix computed as an outproduct
        
        Mf = Int*fqe * np.multiply(0.5*phi[Iwlf][:,None]*eps, kChl_iwle*sigmoid)
        Mb = Int*fqe * np.multiply(0.5*phi[Iwlf][:,None]*eps, kChl_iwle*sigmoid)
        
        
        Ih          = np.ones((1,len(te)))# row of ones
        Iv          = np.ones((len(tf),1))# column of ones

        # Doubling routine
        for i in range(1, ndub):
            
            xe  = te/(1-re*re)  
            ten = te*xe
            ren = re*(1+ten)  
            
            xf = tf/(1-rf*rf)  
            tfn = tf*xf
            rfn = rf*(1+tfn)
                  
            A11 = xf[:,None]*Ih + Iv*xe[None,:]           
            A12 = (xf[:,None]*xe[None,:])*(rf[:,None]*Ih + Iv*re[None,:])
            A21 = 1+(xf[:,None]*xe[None,:])*(1+rf[:,None]*re[None,:])   
            A22 = (xf[:,None]*rf[:,None])*Ih+Iv*(xe*re[None,:])
            
            Mfn = Mf * A11 + Mb * A12
            Mbn = Mb * A21 + Mf * A22
            
            te = ten  
            re = ren   
            tf = tfn   
            rf = rfn
            
            Mf = Mfn 
            Mb = Mbn
        
        # Here we add the leaf-air interfaces again for obtaining the final 
        # leaf level fluorescences.
        
        g = Mb
        f = Mf
        
        Rb = rho + tau**2*r21/(1-rho*r21)
        Rb_iwle = interp1d(spectral_wlP,Rb)(wle) 
            
        Xe = Iv * (talf_iwle/(1-r21_iwle*Rb_iwle)).T
        Xf = np.outer(t21[Iwlf]/(1-r21[Iwlf]*Rb[Iwlf]),Ih)
        Ye = Iv * (tau_iwle*r21_iwle/(1-rho_iwle*r21_iwle)).T
        Yf = np.outer(tau[Iwlf]*r21[Iwlf]/(1-rho[Iwlf]*r21[Iwlf]), Ih)
        
        A = Xe * (1 + Ye*Yf)* Xf
        B = Xe * (Ye + Yf)  * Xf
        
        gn = A * g + B * f
        fn = A * f + B * g
        
        leafopt_Mb  = gn
        leafopt_Mf  = fn
        
        return leafopt_Mb, leafopt_Mf

#%% 2) SIF at Canopy level
def cal_rta_sif(x, dC, Esolar, Mps, canopy_pars, hemi_pars, dif_pars, hemi_dif_pars):
    
    rho_l, tau_l = dC.leaf
    rho_l, tau_l = rho_l[:,x%365], tau_l[:,x%365]

    rs = dC.soil

    [Qins, Qind] = Esolar
    [Mfp, Mbp]   = Mps
    
    [i0, iD, p, rho, rho2, tv, kc, kg] = canopy_pars
    
    K, sob, sof     = dC.ko[x], dC.sob[x], dC.sof[x] 
    [sob_vsla,          sof_vsla,          kgd]     = hemi_pars
    [sob_vsla_dif,      sof_vsla_dif,      kg_dif]  = dif_pars
    [sob_vsla_hemi_dif, sof_vsla_hemi_dif, kgd_dif] = hemi_dif_pars
    
    #nb = 2162
    #nf = 211
    #angle = 58
    
    iwlfi = np.arange(0, 351)
    iwlfo = np.arange(240, 451)  
    
    t0 = 1-i0
    td = 1-iD
    
    wleaf = rho_l + tau_l
    Mf = Mfp + Mbp
    
    """
    Part 1:
    """    
    Qsig_x  = np.logspace(0, 11, num=12, base=(wleaf*p)[:,None])
    Qsig_y  = Qins*i0
    Qsig    = np.multiply(Qsig_x, Qsig_y[:,None])
    
    MQ = np.dot(Mf, Qsig[iwlfi,:])
    Qsig[iwlfo] += MQ*p
    
    Qapar = np.multiply(Qsig, (1-wleaf)[:,None]) #absorbed photon
    
    Qfdir = np.multiply(Qsig,wleaf[:,None])[...,None]*rho
    Qfdir[iwlfo,:,:] += MQ[...,None]*rho #escaped directional SIF+photon
    Qfdir = np.einsum('ijk->ikj', Qfdir)
    
    Qfhemi = np.multiply(Qsig, (wleaf*rho2)[:,None]) 
    Qfhemi[iwlfo] += MQ*rho2#upper hemi SIF+photon
    
    Qfdir[:,:,0] = np.outer(Qins*rho_l, sob*kc/K)+ np.outer(Qins*tau_l, sof*kc/K)    
    Qfdir[iwlfo,:,0] += np.outer(Mbp@Qins[iwlfi],sob*kc/K) + np.outer(Mfp@Qins[iwlfi],sof*kc/K) #escaped directional SIF+photon
    
    Qfhemi[:,0] = Qins*rho_l*sob_vsla+ Qins*tau_l*sof_vsla
    Qfhemi[iwlfo,0] += Mbp@Qins[iwlfi]*sob_vsla+ Mfp@Qins[iwlfi]*sof_vsla#upper hemi SIF+photon
    
    Qfyld = MQ #yield SIF
    
    Qdown = np.multiply(Qsig, (wleaf*rho2)[:,None]) #lower hemi total
    Qdown[iwlfo,:] = Qdown[iwlfo,:]+MQ*rho2
    
    """
    Part 2:
    """
    Qsig_x  = np.logspace(0, 11, num=12, base=(wleaf*p)[:,None])
    Qsig_y  = Qind*iD
    Qsig_d  = np.multiply(Qsig_x, Qsig_y[:,None])
        
    MQ = np.dot(Mf, Qsig_d[iwlfi,:])
    Qsig_d[iwlfo] += MQ*p
    
    Qapar_d = np.multiply(Qsig_d, (1-wleaf)[:,None]) #absorbed photon
    
    Qfdir_d = np.multiply(Qsig_d,wleaf[:,None])[...,None]*rho
    Qfdir_d[iwlfo,:,:] += MQ[...,None]*rho #escaped directional SIF+photon
    Qfdir_d = np.einsum('ijk->ikj', Qfdir_d)
    
    Qfhemi_d = np.multiply(Qsig_d, (wleaf*rho2)[:,None]) 
    Qfhemi_d[iwlfo] += MQ*rho2#upper hemi SIF+photon
    
    Qfdir_d[:,:,0] = np.outer(Qind*rho_l, sob_vsla_dif)+ np.outer(Qind*tau_l, sof_vsla_dif)    
    Qfdir_d[iwlfo,:,0] += np.outer(Mbp@Qind[iwlfi],sob_vsla_dif) + np.outer(Mfp@Qind[iwlfi],sof_vsla_dif) #escaped directional SIF+photon
    
    Qfhemi_d[:,0] = Qind*rho_l*sob_vsla_hemi_dif+ Qind*tau_l*sof_vsla_hemi_dif
    Qfhemi_d[iwlfo,0] += Mbp@Qind[iwlfi]*sob_vsla_hemi_dif+ Mfp@Qind[iwlfi]*sof_vsla_hemi_dif#upper hemi SIF+photon
    
    Qfyld_d = MQ #yield SIF
    
    Qdown_d = np.multiply(Qsig_d, (wleaf*rho2)[:,None]) #lower hemi total
    Qdown_d[iwlfo] += MQ*rho2
        
    """
    Part 3:
    """
    Qapar_bs = np.sum(Qapar+Qapar_d, axis=1)
    
    Qfdir_bs = np.sum(Qfdir+Qfdir_d, axis=2)
    Qfhemi_bs= np.sum(Qfhemi+Qfhemi_d, axis=1)
    Qfyld_bs = np.sum(Qfyld+Qfyld_d, axis=1)
    
    #The preparation for the S problem
    Qdown_bs = Qins*t0 + Qind*td + np.sum(Qdown+Qdown_d, axis=1)
    Qind_s   = Qdown_bs*rs #include the 0-order transmission
    
    Qdown_bs_hot = Qins*t0
    Qind_s_hot   = Qdown_bs_hot*rs
    
    Qdown_bs_d = Qind*td + np.sum(Qdown+Qdown_d, axis=1)
    Qind_s_d = Qdown_bs_d*rs
    
    for k in range(0,20):
        
        if k==0:
            Qsig_y=Qind_s_hot*iD+Qind_s_d*iD
        else:
            Qsig_y=Qind_s*iD  
        
        Qsig_x  = np.logspace(0, 11, num=12, base=(wleaf*p)[:,None])
        Qsig_s  = np.multiply(Qsig_x, Qsig_y[:,None])
    
        MQ = np.dot(Mf, Qsig_s[iwlfi,:])
        Qsig_s[iwlfo] += MQ*p
    
        Qapar_s = np.multiply(Qsig_s, (1-wleaf)[:,None]) #absorbed photon
        
        Qfdir_s = np.multiply(Qsig_s,wleaf[:,None])[...,None]*rho
        Qfdir_s[iwlfo,:,:] += MQ[...,None]*rho #escaped directional SIF+photon
        Qfdir_s = np.einsum('ijk->ikj', Qfdir_s)
        
        Qfhemi_s = np.multiply(Qsig_s, (wleaf*rho2)[:,None]) 
        Qfhemi_s[iwlfo] += MQ*rho2#upper hemi SIF+photon
        
        Qfyld_s = MQ #yield SIF
        
        Qdown_s = np.multiply(Qsig_s, (wleaf*rho2)[:,None]) #lower hemi total
        Qdown_s[iwlfo] += MQ*rho2
    
        #sum up direct and diffuse radiation in S problem
        # Qadir_ss=(sum(Qadir_s,3)+Qind_s*t0_s)./repmat((Qins+Qind),1,angle);
        # Qalbedo_ss=(sum(Qalbedo_s,2)+Qind_s*t02_s)./(Qins+Qind);    
        if k == 0:
            Qapar_ss  = np.sum(Qapar_s, axis=1)        
            Qfdir_ss  = np.sum(Qfdir_s, axis=2)+np.outer(Qins*rs,kg)+np.outer(Qind*rs,kg_dif)+np.outer(np.sum(Qdown+Qdown_d, axis=1)*rs,tv) #Qind_s_hot*t0_shot ; Qind_s_d*tv
            Qfhemi_ss = np.add(np.sum(Qfhemi_s, axis=1)[:,None], np.outer(Qins*rs,kgd) +np.outer(Qind*rs,kgd_dif)+np.outer(np.sum(Qdown+Qdown_d, axis=1)*rs,td)) #Qind_s_hot*t02_shot; Qind_s_d*td
            Qfyld_ss  = np.sum(Qfyld_s, axis=1)
        else:
            Qapar_ss  += np.sum(Qapar_s, axis=1)
            Qfdir_ss  += np.sum(Qfdir_s, axis=2)+np.outer(Qind_s,tv)
            Qfhemi_ss += np.add(np.sum(Qfhemi_s, axis=1)[:,None], np.outer(Qind_s,td))
            Qfyld_ss  += np.sum(Qfyld_s, axis=1)
        
        Qdown_ss = np.sum(Qdown_s, axis=1)
        Qind_s   = Qdown_ss*rs   #include the 0-order transmission
        
    """
    Part 4: Output
    """
    Qfdir_all=Qfdir_bs+Qfdir_ss
    Qfhemi_all=Qfhemi_bs[:,None]+Qfhemi_ss
    Qfyld_all=Qfyld_bs+Qfyld_ss
    Qapar_all=Qapar_bs+Qapar_ss
    
    Qpdir_bs=Qfdir_bs
    return Qfdir_all, Qfhemi_all, Qfyld_all, Qapar_all, Qpdir_bs

def cal_canopy_sif(x, dC, Esolar, canopy_pars, hemi_pars, dif_pars, hemi_dif_pars):
            
    [Mf, Mb] = dC.Mps
    MfI, MfII = Mf*0.5, Mf*(1-0.5)
    MbI, MbII = Mb*0.5, Mf*(1-0.5)
    [Qins, Qind] = Esolar 
    
    iwlfi = np.arange(0, 351)
    iwlfo = np.arange(240, 451)  
    angle = 1
    
    MpsI     = [MfI,  MbI]    
    MpsII    = [MfII, MbII]    
    Mpsall   = [MfI*0, MbI*0] 
    
    Qfdir_1,   Qfhemi_1,   Qfyld_1,   Qapar_all, _        = cal_rta_sif(x, dC, Esolar, MpsI, canopy_pars, hemi_pars, dif_pars, hemi_dif_pars)
    #Qfdir_2,   Qfhemi_2,   Qfyld_2,   Qapar_all, _        = cal_rta_sif(x, dC, Esolar, MpsII, canopy_pars, hemi_pars, dif_pars, hemi_dif_pars)
    #Qpdir_all, Qphemi_all, Qpyld_all, Qpapar_all, Qpdir_bs = cal_rta_sif(x, dC, Esolar, Mpsall, canopy_pars, hemi_pars, dif_pars, hemi_dif_pars)
    
    """
    directional SIF 680-740 PSI, PSII
    """
    """
    SRTE_Fs_fdir1 = Qfdir_1-Qpdir_all
    SRTE_Fs_fdir1 = SRTE_Fs_fdir1[iwlfo]
    SRTE_Fs_fdir2 = Qfdir_2-Qpdir_all
    SRTE_Fs_fdir2 = SRTE_Fs_fdir2[iwlfo]
    SRTE_Fs_fdir_all = SRTE_Fs_fdir1 + SRTE_Fs_fdir2
    """
    """
    hemishpere SIF 680-740 PSI, PSII
    """
    """
    SRTE_Fs_fhemi1 = Qfhemi_1-Qphemi_all
    SRTE_Fs_fhemi1 = SRTE_Fs_fhemi1[iwlfo]
    SRTE_Fs_fhemi2 = Qfhemi_2-Qphemi_all
    SRTE_Fs_fhemi2 = SRTE_Fs_fhemi2[iwlfo]
    SRTE_Fs_fhemi_all = SRTE_Fs_fhemi1+SRTE_Fs_fhemi2
    """
    """
    Canopy yield SIF PSI, PSII
    """
    """
    SRTE_Fs_fyld1 = Qfyld_1
    SRTE_Fs_fyld2 = Qfyld_2
    SRTE_Fs_fyld_all = Qfyld_1+Qfyld_2
    
    SRTE_RefAll=Qpdir_all/np.matlib.repmat((Qins+Qind),1,angle)
    SRTE_Hemi=Qphemi_all/(Qins+Qind)
    SRTE_Absorb=Qapar_all/(Qins+Qind)
    Qin=Qins+Qind
    SRTE_RefBS=Qpdir_bs/np.matlib.repmat((Qins+Qind),1,angle)
    """
    return np.nanmean(Qfyld_1)