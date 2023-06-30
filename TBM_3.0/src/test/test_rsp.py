# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 14:09:39 2022

@author: hliu
"""
import numpy as np
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
    
def Respiration(t):
    R     = 8.314             # [J mol-1K-1]   Molar gas constant
    
    Vcmax25 = 60
    RdPerVcmax25 = 0.015
    Rd25  = RdPerVcmax25 * Vcmax25
    
    T     = t + 273.15 # convert temperatures to K if not already
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
    
    return Rd
import pandas as pd
data1 = pd.read_csv("../../data/driving/HARV.csv", na_values="nan") 
t_mean = data1['TA'][185*24:190*24]
out = []
for t in t_mean:
    out.append(Respiration(t))
    
import matplotlib.pyplot as plt
plt.plot(out)    