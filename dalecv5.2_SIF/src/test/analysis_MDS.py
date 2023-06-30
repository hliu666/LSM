# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 18:16:12 2022

@author: 16072
"""
import sys 
sys.path.append("../model")

import os
import matplotlib.pyplot as plt
import numpy as np
import data_class as dc
import pandas as pd

from Optical_RTM import Opt_Refl_MDS
import prosail

import time
start = time.time()

ci_flag = 0
site = "UNDE"
d = dc.DalecData(2019, 2022, site, ci_flag, 'nee')

"""
CI_flag
    0: CI varied with the zenith angle
    1: CI as a constant 
    2: Without considering CI effect            
"""    
#output_0 = np.load("../../data/output/verify/d_lai0_{0}.npy".format(site))  
#output_1 = np.load("../../data/output/verify/d_lai1_{0}.npy".format(site)) 
#output_2 = np.load("../../data/output/verify/d_lai2_{0}.npy".format(site)) 

import joblib
output_0 = joblib.load("../../data/output/model/lai_ci0_{0}.pkl".format(site))  
output_1 = joblib.load("../../data/output/model/lai_ci1_{0}.pkl".format(site)) 
output_2 = joblib.load("../../data/output/model/lai_ci2_{0}.pkl".format(site)) 

lai_arrs = output_0
#lai_arrs = output_1[:,-1] + output_1[:,-2]
#lai_arrs = output_2[:,-1] + output_2[:,-2]

startrun = 0
endrun = int(len(d.flux_data)/24)
refls = []
#sails = []
for doy in range(endrun-startrun):
    if doy in d.brf_data['index'].values:
        lai = lai_arrs[doy]
        loc = d.brf_data[d.brf_data['index']==doy].index.values[0]
        refl_MDS = Opt_Refl_MDS(d, loc, doy, lai)
        if len(refl_MDS) > 1:
            refls.append(refl_MDS)

        rhos, taus = d.leaf
        rho,  tau  = rhos[:,loc], taus[:,loc]
        
        lidfa = 30
        lidfb = -0.15
        sza, vza, raa = d.tts_MDS[loc], d.tto_MDS[loc], d.psi_MDS[loc]
        hspot = 0.001
                 
        #rho_canopy = prosail.run_sail(rho[0:2101], tau[0:2101], lai, lidfa, hspot, sza, vza, raa, rsoil0=d.soil[0:2101])
        #sails.append(float(np.nanmean(rho_canopy[220:271])))

refl_sim = np.array(refls)
for i in range(0,2):
    plt.figure(figsize=(8,4))
    plt.title('sur_refl_b0'+str(i+1))
    plt.scatter(d.brf_data['doy'], d.brf_data['sur_refl_b0'+str(i+1)], color = 'r', alpha=0.5)
    plt.scatter(d.brf_data['doy'], refl_sim[:,i], color = 'b', alpha=0.5)
    #plt.plot(d.brf_data['doy'], sails, color = 'black', alpha=0.5)

end = time.time()
print(end - start)
