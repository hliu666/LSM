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

Vcmax25         = 60.0  
Jmax25          = Vcmax25 * 2.68
RdPerVcmax25    = 0.015
BallBerrySlope  = 8.0
BallBerry0      = 0.01           # intercept of Ball-Berry stomatal conductance model
Rd25            = RdPerVcmax25 * Vcmax25

Tref            = 25 + 273.15    # [K] absolute temperature at 25 degrees

Kc25            = 404.9 * 1E-6    # [mol mol-1]
Ko25            = 278.4 * 1E-3    # [mol mol-1]

# temperature correction for Kc
Ec = 79430	   # Unit is  [J K^-1]

# temperature correction for Ko
Eo = 36380     # Unit is  [J K^-1]

spfy25          = 2444          # specificity (Computed from Bernacchhi et al 2001 paper)
ppm2bar         = 1E-6 * (p *1E-3) # convert all to bar: CO2 was supplied in ppm, O2 in permil, and pressure in mBar
O_c3            = (O * 1E-3) * (p *1E-3) 
Gamma_star25    = 0.5 * O_c3/spfy25 # [ppm] compensation point in absence of Rd
   
# temperature correction for Gamma_star
Eag     = 37830  #Unit is [J K^-1]

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

"""
Photosynthesis parameters (Jen)
"""

# Initial values of parameters and constants
RUB = 60     # [umol sites m-2] Rubisco density
Rdsc = 0.01  # [] Scalar for mitochondrial (dark) respiration

R = 8.314    # [J mol-1 K-1] Ideal gas constant

# define the conductance in mesophyll cell 
gm = 0.01 # [] mesophyll conductance to CO2

# Cytochrome b6f complex
kc0 = 1.0 #[umol CO2 umol sites-1 s-1] Rubisco kcat for CO2
Eac = 58000  # [J umol-1] Activation energy

kq0 = 35 # [umol e-1 umol sites-1 s-1] Cyt b6f kcat for PQH2
Eaq = 37000 # [J mol-1] Activation energy 
CB6F = 1.6*350/300 #[umol sites m-2] Cyt b6f density

# Cytochrome b6f-limited rates
Kp1 = 14.5E9 #[s-1] Rate constant for photochemistry at PSI
Kf = 0.05E9  #[s-1] Rate constant for fluoresence at PSII and PSI
Kd = 0.55E9  #[s-1] Rate constant for constitutive heat loss at PSII and PSI

Abs = 0.85 # [umol absorbed umol-1 incident] Total leaf absorptance to PAR
a2_frac = 0.52 # [] PS II fraction of mesophyll absorptance to PAR
a2 = Abs*(a2_frac)  # [] PSII, mol PPFD abs PS2 mol-1 PPFD incident
a1 = Abs - a2       # [] PSI, mol PPFD abs PS1 mol-1 PPFD incident

nl = 0.75 # [ATP/e-] ATP per e- in linear flow
nc = 1.00 # [ATP/e-] ATP per e- in cyclic flow

"""
Fluorescence parameters (Jen)
"""
# Assign values for photochemical constants
Kn1 = 14.5E9 #[s-1] Rate constant for regulated heat loss at PSI    
Kp2 = 4.5E9 #[s-1] Rate constant for photochemistry at PSII
Ku2 = 0E9 #[s-1] Rate constant for exciton sharing at PSII

eps1 = 0.5 # [mol PSI F to detector mol-1 PSI F emitted] PS I transfer function
eps2 = 0.5 # [mol PSII F to detector mol-1 PSII F emitted] PS II transfer function
