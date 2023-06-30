# -*- coding: utf-8 -*-
"""
Created on Sun Aug 14 21:49:29 2022

@author: hliu
"""

# this will fail if SALib isn't properly installed
from SALib.sample import saltelli
import pandas as pd
import numpy as np

fields = ["RUB", "CB6F", "BallBerry0"]

# Specify the model inputs and their bounds. The default probability
# distribution is a uniform distribution between lower and upper bounds.
# %% Leaf traits parameter sensitivity analysis
problem = {
    "num_vars": 3,
    "names": ["RUB", "CB6F", "BallBerry0"],
    "bounds": [[0, 120], [0, 150],  [0, 1]]
}

RUB = np.linspace(0, 120, 4)
CB6F = np.linspace(0, 150, 4)
BallBerry0 = np.linspace(0, 1, 4)

mesh = np.meshgrid(RUB, CB6F, BallBerry0)
sample = np.transpose(np.hstack([col.reshape(-1, 1) for col in mesh])).transpose()

df = pd.DataFrame(sample, columns=fields)

df = df[fields]
df.to_csv('../../data/parameters/HARV_pars.csv', index=False)

interval = 1
sub_files = 1
sub_lens = int(len(sample) / sub_files)

for i in range(0, sub_files):
    sid, eid = i * sub_lens, (i + 1) * sub_lens
    x = np.array(range(sid, eid, interval)).astype(int)
    y = np.repeat(interval, len(x))
    id_arr = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))
    np.savetxt('../../data/parameters/pars{0}.txt'.format(i), id_arr, '%-d', delimiter=',')  # X is an array
