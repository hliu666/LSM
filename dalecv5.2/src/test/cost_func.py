# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 01:22:04 2022

@author: hliu
"""
import os
import numpy as np
import pandas as pd
import joblib

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

path = r"C:\Users\liuha\Desktop\output2"

Y1 = pd.read_csv("../../data/parameters/HARV_pars.csv")
    
v1 = pd.read_csv("../../data/verify/HARV_nee.csv").iloc[0:26280,:]
v2 = pd.read_csv("../../data/verify/HARV_tir.csv")
index = np.array(v2['index']).astype(int)
v3 = pd.read_csv("../../data/verify/HARV_brf.csv")

rmin = 9999
rid = -1

for i in range(len(Y1)):
    if os.path.exists(os.path.join(path, "nee_ci1_HARV_{0}.pkl".format(i))):
        s1 = joblib.load(os.path.join(path, "nee_ci1_HARV_{0}.pkl".format(i)))
        s2 = joblib.load(os.path.join(path, "lst_ci1_HARV_{0}.pkl".format(i)))
        s3 = joblib.load(os.path.join(path, "refl_ci1_HARV_{0}.pkl".format(i)))
        
        v1['nee_sim'], v2['lst_sim'] = s1, s2[index]
        v3['red_sim'], v3['nir_sim'] = s3[:,0], s3[:,1]
        
        v1 = v1[['nee_sim', 'nee']]
        v2 = v2[['lst_sim', 'LST_Day_1km']]
        v3 = v3[['red_sim', 'nir_sim', 'sur_refl_b01', 'sur_refl_b02']]
        
        v1_nan = v1.dropna(axis=0)
        v2_nan = v2.dropna(axis=0)
        v3_nan = v3.dropna(axis=0)

        r1 = rmse(v1_nan['nee_sim'], v1_nan['nee'])
        r2 = rmse(v2_nan['lst_sim'], v2_nan['LST_Day_1km'])
        r3 = rmse(v3_nan['red_sim'], v3_nan['sur_refl_b01'])
        r4 = rmse(v3_nan['nir_sim'], v3_nan['sur_refl_b02'])
        
        r = r1 + r2 + r3 + r4
        if r < rmin:
            rmin = r
            rid = i 
            
print("The optimal parameters are:")
print(Y1.iloc[rid])
print("The min RMSE is {0}".format(rmin))
