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
          "p14", "p15", "p16", "p17", "BallBerrySlope",\
          "Cab", "Car", "Cbrown", "Cw", "Ant"]
    
# Specify the model inputs and their bounds. The default probability
# distribution is a uniform distribution between lower and upper bounds.
#%% Leaf traits parameter sensitivity analysis
problem = {
    "num_vars": 30, 
    "names": ["clab", "cf", "cr", "cw", "cl", "cs",\
              "p0", "p1", "p2", "p3", "p4", "p5", "p6",\
              "p7", "p8", "p9", "p10", "p11", "p12","p13",\
              "p14", "p15", "p16", "p17", "BallBerrySlope",\
              "Cab", "Car", "Cbrown", "Cw", "Ant"], 
        
    "bounds": [[10, 1000], [0, 1000], [10, 1000], [3000, 4000], [10, 1000], [1000, 1e5],\
               [1e-5, 1e-2], [0.3, 0.7], [0.01, 0.5], [0.01, 0.5], [1.0001, 5], [2.5e-5, 1e-3], [1e-4, 1e-2],\
               [1e-4, 1e-2], [1e-7, 1e-3], [0.018, 0.08], [60, 150], [0.01, 0.5], [10, 100], [242, 332],\
               [10, 100], [50, 100], [0.7, 0.9], [0, 100], [0.0, 20.0],\
               [0, 40], [0, 10], [0, 1], [0, 0.1], [0, 30]]
}

N = 2048
# generate the input sample
sample = saltelli.sample(problem, N)
df = pd.DataFrame(sample, columns = fields)
    
df = df[fields]       
df.to_csv('../../data/parameters/HARV_pars.csv', index=False) 

interval = 6
sub_files = 4
sub_lens = int(len(sample)/sub_files)
types = 2 #['Icar','Leaf','Dalec']

for i in range(0,sub_files):
    sid, eid = i*sub_lens,(i+1)*sub_lens
    x = np.array(range(sid, eid, interval)).astype(int)
    y = np.repeat(interval, len(x))
    z = np.repeat(types, len(x))
    id_arr = np.hstack((x.reshape(-1,1), y.reshape(-1,1), z.reshape(-1,1)))
    np.savetxt('../../data/parameters/pars{0}.txt'.format(i), id_arr, '%-d', delimiter=',')   # X is an array  
