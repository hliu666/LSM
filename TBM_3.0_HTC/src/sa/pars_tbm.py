# -*- coding: utf-8 -*-
"""
Created on Sun Aug 14 21:49:29 2022

@author: hliu
"""

# this will fail if SALib isn't properly installed
from SALib.sample import saltelli
import pandas as pd
import numpy as np

fields = ["clspan", "lma", "f_auto", "f_fol", "f_lab", \
          "d_onset", "cronset", "d_fall", "crfall", "CI_thres", "lidfa", \
          "RUB", "Rdsc", "CB6F", "gm", "e", "BallBerrySlope", "BallBerry0", \
          "Cab", "Car", "Cbrown", "Cw", "Ant", "rho", "tau"]

# Specify the model inputs and their bounds. The default probability
# distribution is a uniform distribution between lower and upper bounds.
# %% Leaf traits parameter sensitivity analysis
problem = {
    "num_vars": 25,
    "names": ["clspan", "lma", "f_auto", "f_fol", "f_lab", \
              "d_onset", "cronset", "d_fall", "crfall", "CI_thres", "lidfa", \
              "RUB", "Rdsc", "CB6F", "gm", "e", "BallBerrySlope", "BallBerry0", \
              "Cab", "Car", "Cbrown", "Cw", "Ant", "rho", "tau"],

    "bounds": [[1.0001, 5], [80, 120], [0.3, 0.7], [0.01, 0.5], [0.01, 0.5], \
               [60, 150], [10, 100], [240, 365], [10, 100], [0.0, 1.0], [0, 100], \
               [0, 120], [0.01, 0.05], [0, 150], [0.01, 5], [0, 5], [0, 15], [0, 1], \
               [0, 40], [0, 10], [0, 1], [0, 0.1], [0, 30], [0.001, 0.05], [0.001, 0.05]]
}

N = 16384
# generate the input sample
sample = saltelli.sample(problem, N)
df = pd.DataFrame(sample, columns=fields)

df = df[fields]
df.to_csv('../../data/parameters/HARV_pars.csv', index=False)

interval = 50
sub_files = 10
sub_lens = int(len(sample) / sub_files)

for i in range(0, sub_files):
    sid, eid = i * sub_lens, (i + 1) * sub_lens
    x = np.array(range(sid, eid, interval)).astype(int)
    y = np.repeat(interval, len(x))
    id_arr = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))
    np.savetxt('../../data/parameters/pars{0}.txt'.format(i), id_arr, '%-d', delimiter=',')  # X is an array
