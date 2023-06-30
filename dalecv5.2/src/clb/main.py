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

    nee_df = pd.read_csv("../../data/verify/HARV_nee.csv", na_values="nan") 
    x = np.array(nee_df['nee'][0:26280])
    nee = x
    #nee = np.array([np.nanmean(x[m: m+1]) for m in range(11, len(x), 24)])
    
    Spot_setup = spotpy_setup(spotpy.objectivefunctions.rmse, nee)
    rep = 50       #Select number of maximum repetitions
    dbname = "SCEUA"
  
    #sampler = spotpy.algorithms.sceua(Spot_setup, dbname=dbname, dbformat='csv',parallel='mpi')
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
    ax.plot(nee,'r.',markersize=3, label='Observation data')

    ax.set_xlabel('Best simulations', fontsize = 20, family="Times New Roman")    
    ax.set_ylabel('NEE observations', fontsize = 20, family="Times New Roman")
    
    end = time.time()
    print(end - start)
