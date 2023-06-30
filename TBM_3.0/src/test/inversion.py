# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 16:56:54 2022

@author: 16072
"""
import sys 
sys.path.append("../model")


import numpy as np
import data_class_inversion as dc
import pandas as pd

from Optical_RTM import Opt_Refl_inver

import time
start = time.time()

data1 = pd.read_csv("../../data/driving/UNDE.csv", na_values="nan") 
data2 = pd.read_csv("../../data/verify/UNDE_nee.csv", na_values="nan") 
data3 = pd.read_csv("../../data/verify/UNDE_lai.csv", na_values="nan")
data4 = pd.read_csv("../../data/verify/UNDE_gpp.csv", na_values="nan")
data5 = pd.read_csv("../../data/verify/UNDE_brf.csv", na_values="nan")
data6 = pd.read_csv("../../data/verify/UNDE_tir.csv", na_values="nan")
datas = [data1, data2, data3, data4, data5, data6]

import joblib 
output_0 = joblib.load("../../data/output/model/lai_ci0_UNDE.pkl")  
output_1 = joblib.load("../../data/output/model/sai_ci0_UNDE.pkl")  
lai_arrs  = output_0 + output_1
  
from SALib.sample import fast_sampler
problem = {
    'num_vars': 9,
    'names': ['Cab','Car','Cbrown','Cw','Cm','Alpha', 'a', 'b', 'c'],
    'bounds': [[35, 45],
               [5, 10],
               [0.1, 0.8],
               [0.1, 0.8],
               [0.001, 0.1],
               [100, 200], 
               [0, 3000],
               [-1000, 0],
               [0, 1]]}

param_values = fast_sampler.sample(problem, 400, M=9)
RMSE_opt = 1000
for pars in param_values:
    print(pars)
    d = dc.DalecData(2019, 2022, pars, datas, 'nee')
    startrun = 0
    endrun = int(len(d.flux_data)/24)
    count = 0 
    refls = []
    for doy in range(endrun-startrun):
        if doy in d.brf_data['index'].values:
            lai = lai_arrs[doy]
            loc = d.brf_data[d.brf_data['index']==doy].index.values[0]
            refl_MDS = Opt_Refl_inver(d, loc, lai)
            if len(refl_MDS) > 1:
                refls.append(refl_MDS)
                
    a = np.array(d.brf_data[["sur_refl_b01", "sur_refl_b02"]]).reshape(-1,1)
    b = np.array(refls).reshape(-1,1)
    from sklearn.metrics import mean_squared_error
    RMSE = mean_squared_error(a, b, squared=True)
    if RMSE < RMSE_opt :
        RMSE_opt  = RMSE
        RMSE_list = pars

"""
refl_sim = np.array(refls)
for i in range(0, 7):
    plt.figure(figsize=(8,4))
    plt.title('sur_refl_b0'+str(i+1))
    plt.plot(d.brdf_data['sur_refl_b0'+str(i+1)], color = 'r', alpha=0.5)
    plt.plot(refl_sim[:,i], color = 'b', alpha=0.5)
"""
print(RMSE_list)
end = time.time()
print(end - start)