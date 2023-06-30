# -*- coding: utf-8 -*-
"""
Created on Sun Aug 14 21:49:29 2022

@author: hliu
"""

# this will fail if SALib isn't properly installed
from SALib.sample import saltelli
import pandas as pd
import numpy as np

fields = ["clab", "cf", "cr", "cw", "cl", "cs",\
          "p0", "p1", "p2", "p3", "p4", "p5", "p6",\
          "p7", "p8", "p9", "p10", "p11", "p12","p13",\
          "p14", "p15", "Vcmax25", "BallBerrySlope",\
          "Cab", "Car", "Cbrown", "Cw", "Ant", "fLMA_k", "gLMA_k", "gLMA_b",\
          "lidfa", "lidfb", "CI_thres"]
    
pars = [100.0, 0.0001, 5.0, 5.0, 5.0, 9900.0,\
        0.00521223, 0.319483, 0.389599, 0.0110404, 1.01636, 6.08353e-05, 0.00682226,\
        0.0396876, 0.000565224, 0.0911751, 105.638, 0.392179, 85.6607, 321.312,\
        61.7368, 89.1099, 106.522, 10.0,\
        40.71, 7.69, 0.4, 0.01, 0.001, 2200, -500, 0.065,\
        0.35, -0.15, 0.7]


# Specify the model inputs and their bounds. The default probability
# distribution is a uniform distribution between lower and upper bounds.
#%% Leaf traits parameter sensitivity analysis
problem = {
    "num_vars": 6, 
    "names": ["clab", "cf", "cr", "cw", "cl", "cs"], 
    "bounds": [[20, 2000], [0, 2000], [0, 2000], [0, 1e5], [0, 2000], [100, 2e5]]
}

N = 700
# generate the input sample
sample = saltelli.sample(problem, N)
df = pd.DataFrame(sample, columns = ["clab", "cf", "cr", "cw", "cl", "cs"])
    
for i in range(len(fields)):
    field, par = fields[i], pars[i]
    if field in df.columns:
        continue
    df[field] = par
    
df = df[fields]       
df.to_csv('../../data/parameters/HARV_pars.csv', index=False) 

interval = 1
types = 0 #['Icar','Leaf','Dalec']
x = np.array(range(0, len(sample), interval)).astype(int)
y = np.repeat(interval, len(x))
z = np.repeat(types, len(x))
id_arr = np.hstack((x.reshape(-1,1), y.reshape(-1,1), z.reshape(-1,1)))
np.savetxt('../../data/parameters/pars1.txt', id_arr, '%-d', delimiter=',')   # X is an array 
