# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 16:41:36 2022

@author: hliu
"""

import pandas as pd
import numpy as np

import spotpy
from spotpy.parameter import Uniform
from spotpy.objectivefunctions import rmse,bias,rsquared,covariance

from spotpy_class import spotpy_setup
import matplotlib.pyplot as plt


if __name__ == '__main__':

    import time
    start = time.time()

    ref_df = pd.read_csv("C:/Users/liuha/Desktop/dalecv5.2/data/parameters/HARV_NEON_spectra.csv", na_values="nan") 
    ref = np.array(ref_df.mean(axis = 0))
    Spot_setup = spotpy_setup(spotpy.objectivefunctions.rmse, ref)
    rep = 5000       #Select number of maximum repetitions
    dbname = "SCEUA"
  
    sampler = spotpy.algorithms.sceua(Spot_setup, dbname=dbname, dbformat='csv')
    sampler.sample(rep)
    
    results = spotpy.analyser.load_csv_results('{0}'.format(dbname))
    
    bestindex,bestobjf = spotpy.analyser.get_minlikeindex(results)
    best_model_run = results[bestindex]
    fields=[word for word in best_model_run.dtype.names if word.startswith('sim')]
    best_simulation = list(best_model_run[fields])
    
    fig = plt.figure(figsize=(8,6))
    ax = plt.subplot(1,1,1)
    ax.plot(best_simulation,color='black',linestyle='solid', label='Best objf.='+str(bestobjf))
    ax.plot(ref,'r.',markersize=3, label='Observation data')

    ax.set_xlabel('Best simulations', fontsize = 20, family="Times New Roman")    
    ax.set_ylabel('reflectance observations', fontsize = 20, family="Times New Roman")
    
    end = time.time()
    print(end - start)
