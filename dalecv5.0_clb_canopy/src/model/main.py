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

from run_model import run_model
import matplotlib.pyplot as plt


if __name__ == '__main__':

    import time
    start = time.time()

    brf_df = pd.read_csv("../../data/verify/HARV_brf.csv", na_values="nan") 
    brf = np.array(brf_df[['sur_refl_b01', 'sur_refl_b02', 'sur_refl_b06', 'sur_refl_b07']])
    #nee = np.array([np.nanmean(x[m: m+1]) for m in range(11, len(x), 24)])
    
    Spot_setup = spotpy_setup(spotpy.objectivefunctions.rmse, brf)
    rep = 500       #Select number of maximum repetitions
    dbname = "SCEUA"
  
    #sampler = spotpy.algorithms.sceua(Spot_setup, dbname=dbname, dbformat='csv',parallel='mpi')
    sampler = spotpy.algorithms.sceua(Spot_setup, dbname=dbname, dbformat='csv')
    sampler.sample(rep)
    #results = spotpy.analyser.load_csv_results('{0}'.format(dbname))
    #bestindex,bestobjf = spotpy.analyser.get_minlikeindex(results)
    #best_model_run = results[bestindex]
    results = pd.read_csv('{0}.csv'.format(dbname))
    best_model_run = results.iloc[results['like1'].idxmin()]
    pars = best_model_run[['parp0', 'parp1']]
    refl_y = run_model(pars)
    
    fig = plt.figure(figsize=(8,6))
    ax = plt.subplot(1,1,1)
    ax.plot(refl_y[:,0],color='black',linestyle='solid', label='Best objf.='+str(best_model_run['like1']))
    ax.plot(brf[:,0],'r.',markersize=3, label='Observation data')

    fig = plt.figure(figsize=(8,6))
    ax = plt.subplot(1,1,1)
    ax.plot(refl_y[:,1],color='black',linestyle='solid', label='Best objf.='+str(best_model_run['like1']))
    ax.plot(brf[:,1],'r.',markersize=3, label='Observation data')

    fig = plt.figure(figsize=(8,6))
    ax = plt.subplot(1,1,1)
    ax.plot(refl_y[:,2],color='black',linestyle='solid', label='Best objf.='+str(best_model_run['like1']))
    ax.plot(brf[:,2],'r.',markersize=3, label='Observation data')

    fig = plt.figure(figsize=(8,6))
    ax = plt.subplot(1,1,1)
    ax.plot(refl_y[:,3],color='black',linestyle='solid', label='Best objf.='+str(best_model_run['like1']))
    ax.plot(brf[:,3],'r.',markersize=3, label='Observation data')
    
    #ax.set_xlabel('Best simulations', fontsize = 20, family="Times New Roman")    
    #ax.set_ylabel('NEE observations', fontsize = 20, family="Times New Roman")
    
    end = time.time()
    print(end - start)
