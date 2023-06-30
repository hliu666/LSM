# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 14:01:14 2022

@author: hliu
"""

import os
import numpy as np
import pandas as pd
import joblib

# this will fail if SALib isn't properly installed
from SALib.sample import saltelli
from SALib.analyze import sobol

N = 256
# Specify the model inputs and their bounds. The default probability
# distribution is a uniform distribution between lower and upper bounds.
#%% Leaf traits parameter sensitivity analysis
problem = {
    "num_vars": 18, 
    "names": ["p0", "p1", "p2", "p3", "p4", "p5", "p6",\
              "p7", "p8", "p9", "p10", "p11", "p12", "p13",\
              "p14", "p15", "Vcmax25", "BallBerrySlope"], 
        
    "bounds": [[1e-5, 1e-2], [0.3, 0.7],   [0.01, 0.5],   [0.01, 0.5], [1.0001, 5], [2.5e-5, 1e-3], [1e-4, 1e-2],\
               [1e-4, 1e-2], [1e-7, 1e-3], [0.018, 0.08], [1, 365],    [0.01, 0.5], [10, 100],      [242, 332],\
               [10, 100],    [50, 100],    [0, 100],      [0, 50]]
}

# generate the input sample
sample = saltelli.sample(problem, N)

path = r"C:\Users\liuha\Desktop\sa"
v0 = pd.read_csv("../../data/verify/HARV.csv").iloc[0:26280,:]
v1 = pd.read_csv("../../data/verify/HARV_daystamp.csv") 
v2 = pd.read_csv("../../data/verify/HARV_brf.csv")

seasons = ['spr', 'sum', 'atm', 'win']
for season in seasons:
    
    Y1 = np.empty([sample.shape[0]])
    Y2 = np.empty([sample.shape[0]])
    Y3 = np.empty([sample.shape[0]])
    Y4 = np.empty([sample.shape[0]])
    Y5 = np.empty([sample.shape[0]])
    Y6 = np.empty([sample.shape[0]])
    
    for i in range(len(Y1)):
        
        if os.path.exists(os.path.join(path, "nee_ci1_HARV_{0}.pkl".format(i))):
            s1 = joblib.load(os.path.join(path, "nee_ci1_HARV_{0}.pkl".format(i)))
            s2 = joblib.load(os.path.join(path, "fpar_ci1_HARV_{0}.pkl".format(i)))
            s3 = joblib.load(os.path.join(path, "out_ci1_HARV_{0}.pkl".format(i)))
            s4 = joblib.load(os.path.join(path, "lst_ci1_HARV_{0}.pkl".format(i)))
            s5 = joblib.load(os.path.join(path, "refl_ci1_HARV_{0}.pkl".format(i)))
            
            v0['nee'], v0['fpar'], v0['lst'] = s1, s2, s4
            v1['lai'] = s3[:,-2]
            v2['red_sim'], v2['nir_sim'] = s5[:,0], s5[:,1]
            
            if season == "spr":
                v0_seasonal = v0[(v0['month'] >= 3)&(v0['month'] <= 5)]
                s1, s2, s4 = v0_seasonal['nee'], v0_seasonal['fpar'], v0_seasonal['lst'] 
                v1_seasonal = v1[(v1['month'] >= 3)&(v1['month'] <= 5)]
                s3 = v1['lai']                
                v2_seasonal = v2[(v2['month'] >= 3)&(v2['month'] <= 5)]
                s5, s6 = v2['red_sim'], v2['nir_sim'] 
                                
            elif season == "sum":
                v0_seasonal = v0[(v0['month'] >= 6)&(v0['month'] <= 8)]
                s1, s2, s4 = v0_seasonal['nee'], v0_seasonal['fpar'], v0_seasonal['lst'] 
                v1_seasonal = v1[(v1['month'] >= 6)&(v1['month'] <= 8)]
                s3 = v1['lai']                
                v2_seasonal = v2[(v2['month'] >= 6)&(v2['month'] <= 8)]
                s5, s6 = v2['red_sim'], v2['nir_sim'] 
                
            elif season == "atm":
                v0_seasonal = v0[(v0['month'] >= 9)&(v0['month'] <= 11)]
                s1, s2, s4 = v0_seasonal['nee'], v0_seasonal['fpar'], v0_seasonal['lst'] 
                v1_seasonal = v1[(v1['month'] >= 9)&(v1['month'] <= 11)]
                s3 = v1['lai']                
                v2_seasonal = v2[(v2['month'] >= 9)&(v2['month'] <= 11)]
                s5, s6 = v2['red_sim'], v2['nir_sim'] 
                
            elif season == "win":
                v0_seasonal = v0[(v0['month'] == 12)|(v0['month'] <= 2)]
                s1, s2, s4 = v0_seasonal['nee'], v0_seasonal['fpar'], v0_seasonal['lst'] 
                v1_seasonal = v1[(v1['month'] == 12)|(v1['month'] <= 2)]
                s3 = v1['lai']                
                v2_seasonal = v2[(v2['month'] == 12)|(v2['month'] <= 2)]
                s5, s6 = v2['red_sim'], v2['nir_sim'] 
                
            s1d = [sum(s1[x: x+24]) for x in range(0, len(s1), 24)]
            s2d = [sum(s2[x: x+24]) for x in range(0, len(s2), 24)]
            s3d = s3
            s4d = [sum(s4[x: x+24]) for x in range(0, len(s4), 24)]
            s5d = s5
            s6d = s6
            
            Y1[i] = np.nanmean(s1d)
            Y2[i] = np.nanmean(s2d)
            Y3[i] = np.nanmean(s3d)
            Y4[i] = np.nanmean(s4d)
            Y5[i] = np.nanmean(s5d)   
            Y6[i] = np.nanmean(s6d)   
            
        else:
            print(i)
    # estimate the sensitivity indices, using the Sobol' method
    """
    Y1:NEE, Y2:fPAR, Y3:LAI
    """
    sensitivity1 = sobol.analyze(problem, Y1)
    sensitivity2 = sobol.analyze(problem, Y2)
    sensitivity3 = sobol.analyze(problem, Y3)
    sensitivity4 = sobol.analyze(problem, Y4)
    sensitivity5 = sobol.analyze(problem, Y5)
    sensitivity6 = sobol.analyze(problem, Y6)
    
    joblib.dump(sensitivity1, "sensitivity_nee_{0}.pkl".format(season))
    joblib.dump(sensitivity2, "sensitivity_fpar_{0}.pkl".format(season))
    joblib.dump(sensitivity3, "sensitivity_lai_{0}.pkl".format(season))
    joblib.dump(sensitivity4, "sensitivity_lst_{0}.pkl".format(season))
    joblib.dump(sensitivity5, "sensitivity_red_{0}.pkl".format(season))
    joblib.dump(sensitivity6, "sensitivity_nir_{0}.pkl".format(season))
