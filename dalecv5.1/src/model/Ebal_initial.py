# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 14:23:29 2022

@author: hliu
"""
from numpy import exp, radians, cos, sin, pi
import numpy as np

from RTM_initial import weighted_sum_over_lidf_solar_vec, CIxy

#%% 1) 

def calc_extinc_coeff_pars(CI_flag, CI_thres, lidf):
    xx=np.array([0.9602898565, -0.9602898565, 0.7966664774, -0.7966664774, 0.5255324099, -0.5255324099, 0.1834346425, -0.1834346425])
    
    ww=np.array([0.1012285363,  0.1012285363, 0.2223810345,  0.2223810345, 0.3137066459,  0.3137066459, 0.3626837834,  0.3626837834])   
    
    # * define limits of integration and the convertion factors for integration
    # * over thetaL (note the tL suffix!)
    upperlimit_tL = np.pi/2.0
    lowerlimit_tL = 0.0
    conv1_tL = (upperlimit_tL-lowerlimit_tL)/2.0
    conv2_tL = (upperlimit_tL+lowerlimit_tL)/2.0        
    
    neword_tL = conv1_tL*xx + conv2_tL
    mu_tL     = np.cos(neword_tL)
    sin_tL    = np.sin(neword_tL)
    
    tta  =  neword_tL*180/pi    # observer zenith angle
 
    Ga,ka  = weighted_sum_over_lidf_solar_vec(tta,lidf)
    
    CIa = CIxy(CI_flag, tta, CI_thres)
    
    sum_tL0 = ww* mu_tL*sin_tL*2*conv1_tL
    
    return [ka*CIa, sum_tL0]