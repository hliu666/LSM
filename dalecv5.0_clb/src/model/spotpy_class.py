# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 16:09:52 2021

@author: Haoran
"""
import numpy as np

import spotpy
from spotpy.parameter import Uniform
from spotpy.objectivefunctions import rmse,bias,rsquared,covariance

from run_model import run_model

class spotpy_setup(object):   
    p0      = Uniform(low=1e-5,   high=1.2e-5) 
    p1      = Uniform(low=0.48,   high=0.52)
    p2      = Uniform(low=0.13,   high=0.16)
    p3      = Uniform(low=0.4,    high=0.5)    
    p4      = Uniform(low=1.001,  high=1.006)
    p5      = Uniform(low=4e-5,   high=5e-5)
    p6      = Uniform(low=6e-3,   high=7e-3)
    p7      = Uniform(low=2e-2,   high=3e-2)
    p8      = Uniform(low=2e-5,   high=3e-5)
    p9      = Uniform(low=0.015,  high=0.025)
    p10     = Uniform(low=120,    high=140)
    p11     = Uniform(low=0.35,   high=0.45)
    p12     = Uniform(low=18,     high=22)
    p13     = Uniform(low=285,    high=315)
    p14     = Uniform(low=30,     high=40)
    p15     = Uniform(low=55,     high=75)
    p16     = Uniform(low=180,    high=250) #carbon clab    
        
    def __init__(self, obj_func, nee):
        #Just a way to keep this example flexible and applicable to various examples
        self.obj_func = obj_func 
        self.obs = nee

    def simulation(self,x):
        _, simulations = run_model(x)
        return simulations

    def evaluation(self):
        return self.obs

    def objectivefunction(self,simulation,evaluation):
            #SPOTPY expects to get one or multiple values back, 
            #that define the performence of the model run
            if not self.obj_func:
                # This is used if not overwritten by user
                # RMSE (root mean squared error) works good for the SCE-UA algorithm, 
                # as it minimizes the objective function.
                # All other implemented algorithm maximize the objectivefunction
                model_performance = spotpy.objectivefunctions.rmse(evaluation,simulation[1:])+simulation[0]
            else:
                cost_value = simulation[0]        
                if cost_value < 10000:
                    merge = np.hstack((evaluation.reshape(-1,1), np.array(simulation[1:]).reshape(-1,1)))
                    merge = merge[~np.isnan(merge).any(axis=1)]
                    evaluation,simulation = merge[:,0], merge[:,1]
                    #Way to ensure flexible spot setup class
                    model_performance = self.obj_func(evaluation,simulation)+cost_value
                else:
                    model_performance =  cost_value
                
            return model_performance    
        


       