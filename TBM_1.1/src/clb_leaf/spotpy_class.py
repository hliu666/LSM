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
    Cab      = Uniform(low=0.0,  high=80) 
    Car      = Uniform(low=0.0,  high=20)
    Cbrown      = Uniform(low=0.0,  high=1)
    Cw      = Uniform(low=0.0,  high=0.05)    
    Ant      = Uniform(low=0.0, high=10.0)
    Cm      = Uniform(low=0.0, high=0.015)
       
    def __init__(self, obj_func, ref):
        #Just a way to keep this example flexible and applicable to various examples
        self.obj_func = obj_func 
        self.obs = ref

    def simulation(self,x):
        simulations = run_model(x)
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
                model_performance = spotpy.objectivefunctions.rmse(evaluation,simulation)
            else:
                model_performance = self.obj_func(evaluation,simulation)

                
            return model_performance    
        


       