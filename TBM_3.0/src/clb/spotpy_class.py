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
    #p0      = Uniform(low=1e-5,   high=1e-2) 
    p1      = Uniform(low=0.3,    high=0.7)
    p2      = Uniform(low=0.01,   high=0.5)
    #p3      = Uniform(low=0.01,   high=0.5)    
    p4      = Uniform(low=1.0, high=5)
    #p5      = Uniform(low=2.5e-5, high=1e-3)
    #p6      = Uniform(low=1e-4,   high=1e-2)
    #p7      = Uniform(low=1e-4,   high=1e-2)
    #p8      = Uniform(low=1e-7,   high=1e-3)
    #p9      = Uniform(low=0.018,  high=0.08)
    p10     = Uniform(low=70,    high=150)
    p11     = Uniform(low=0.0,   high=0.5)
    p12     = Uniform(low=70,     high=100)
    p13     = Uniform(low=270,    high=365)
    p14     = Uniform(low=70,     high=100)
    #p15     = Uniform(low=58,     high=62)
    p16     = Uniform(low=10,    high=500) #carbon clab    
    p17     = Uniform(low=10,    high=500) #carbon clab    
        
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
        


       