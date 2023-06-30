# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 11:58:01 2022

@author: hliu
"""

"""
General parameters
"""
T2K   = 273.15            # convert temperatures to K 
Tyear = 7.4

"""
Photosynthesis parameters
"""
Rhoa  = 1.2047            # [kg m-3]      Specific mass of air   
Mair  = 28.96             # [g mol-1]     Molecular mass of dry air
RGAS  = 8.314             # [J mol-1K-1]   Molar gas constant

O               = 209.0   # [per mille] atmospheric O2 concentration
p               = 970.0   # [hPa] air pressure

Vcmax25         = 30.0  
Jmax25          = Vcmax25 * 2.68
RdPerVcmax25    = 0.015
BallBerrySlope  = 8.0
BallBerry0      = 0.01           # intercept of Ball-Berry stomatal conductance model
Rd25            = RdPerVcmax25 * Vcmax25

Tref            = 25 + 273.15    # [K] absolute temperature at 25 degrees

Kc25            = 404.9 * 1e-6    # [mol mol-1]
Ko25            = 278.4 * 1e-3    # [mol mol-1]

# temperature correction for Kc
Ec = 79430	   # Unit is  [J K^-1]

# temperature correction for Ko
Eo = 36380     # Unit is  [J K^-1]

spfy25          = 2444          # specificity (Computed from Bernacchhi et al 2001 paper)
ppm2bar         = 1E-6 * (p *1E-3) # convert all to bar: CO2 was supplied in ppm, O2 in permil, and pressure in mBar
O_c3            = (O * 1e-3) * (p *1E-3) 
Gamma_star25    = 0.5 * O_c3/spfy25 # [ppm] compensation point in absence of Rd
   
# temperature correction for Gamma_star
Eag     = 37830     # Unit is  [J K^-1]

# temperature correction for Rd
Ear     = 46390  #Unit is [J K^-1]

# temperature correction of Vcmax
Eav     = 55729                #Unit is [J K^-1]
deltaSv = (-1.07*Tyear+668)    #Unit is [J mol^-1 K^-1]
Hdv     = 200000               #Unit is [J mol^-1]

# temperature correction of Jmax
Eaj     = 40719                #Unit is [J K^-1]
deltaSj = (-0.75*Tyear+660)    #Unit is [J mol^-1 K^-1]
Hdj     = 200000               #Unit is [J mol^-1]

minCi   = 0.3

# electron transport
kf        = 3.0E7   # [s-1]         rate constant for fluorescence
kD        = 1.0E8   # [s-1]         rate constant for thermal deactivation at Fm
kd        = 1.95E8  # [s-1]         rate constant of energy dissipation in closed RCs (for theta=0.7 under un-stressed conditions)  
po0max    = 0.88    # [mol e-/E]    maximum PSII quantum yield, dark-acclimated in the absence of stress (Pfundel 1998)
kPSII     = (kD+kf)*po0max/(1.0-po0max) # [s-1]         rate constant for photochemisty (Genty et al. 1989)
fo0       = kf/(kf+kPSII+kD)            # [E/E]         reference dark-adapted PSII fluorescence yield under un-stressed conditions

qLs       = 1.0
NPQs      = 0.0
kps       = kPSII * qLs   # [s-1]         rate constant for photochemisty under stressed conditions (Porcar-Castell 2011)
kNPQs     = NPQs * (kf+kD)# [s-1]         rate constant of sustained thermal dissipation (Porcar-Castell 2011)
kds       = kd * qLs
kDs       = kD + kNPQs
po0       = kps /(kps+kf+kDs)# [mol e-/E]    maximum PSII quantum yield, dark-acclimated in the presence of stress
theta_J   = (kps-kds)/(kps+kf+kDs)# []            convexity factor in J response to PAR

beta      = 0.507 # [] fraction of photons partitioned to PSII (0.507 for C3, 0.4 for C4; Yin et al. 2006; Yin and Struik 2012)
alpha     = beta*po0
