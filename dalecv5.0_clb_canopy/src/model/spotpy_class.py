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
    p0      = Uniform(low=0.7,   high=0.8)  # clumping index 
    p1      = Uniform(low=0,     high=100)  # float Leaf Inclination Distribution at regular angle steps.
       
    def __init__(self, obj_func, refl):
        #Just a way to keep this example flexible and applicable to various examples
        self.obj_func = obj_func 
        self.obs = refl

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
                model_performance = spotpy.objectivefunctions.rmse(evaluation[:,0],simulation[:,0])*5+\
                                    spotpy.objectivefunctions.rmse(evaluation[:,1],simulation[:,1])#+\
                                    #spotpy.objectivefunctions.rmse(evaluation[:,2],simulation[:,2])+\
                                    #spotpy.objectivefunctions.rmse(evaluation[:,3],simulation[:,3])
            else:
                model_performance = spotpy.objectivefunctions.rmse(evaluation[:,0],simulation[:,0])*5+\
                                    spotpy.objectivefunctions.rmse(evaluation[:,1],simulation[:,1])#+\
                                    #spotpy.objectivefunctions.rmse(evaluation[:,2],simulation[:,2])+\
                                    #spotpy.objectivefunctions.rmse(evaluation[:,3],simulation[:,3])
            return model_performance    
        


       