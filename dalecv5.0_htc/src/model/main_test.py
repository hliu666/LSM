# -*- coding: utf-8 -*-
"""
Created on Sun Aug 14 13:02:24 2022

@author: 16072
"""

import os
import pandas as pd
import numpy as np
import data_class as dc
import mod_class as mc
import sys
import pickle

root = os.path.dirname(os.path.dirname(os.getcwd()))

def main(datas, pars):   
    ci_flag = 1
    site = "HARV"
    
    """
    data1 = pd.read_csv(sys.argv[1], na_values="nan") #pd.read_csv("{0}.csv".format(site), na_values="nan") 
    data2 = pd.read_csv(sys.argv[2], na_values="nan") #pd.read_csv("../../data/parameters/{0}_nee.csv".format(site), na_values="nan")
    data3 = pd.read_csv(sys.argv[3], na_values="nan") #pd.read_csv("../../data/parameters/{0}_brf.csv".format(site), na_values="nan")
    data4 = pd.read_csv(sys.argv[4], na_values="nan") #pd.read_csv("../../data/parameters/{0}_wds.csv".format(site), na_values="nan")
    data5 = pd.read_csv(sys.argv[5], na_values="nan", header=None, index_col=0) #HARV_traits.csv    

    B = pickle.load(open(sys.argv[6], 'rb'),encoding='iso-8859-1')  #'b_edc.p' Uses background error cov. matrix B created using ecological    
    
    rsr_red = np.genfromtxt(sys.argv[7])#"rsr_red.txt"
    rsr_nir = np.genfromtxt(sys.argv[8])#"rsr_nir.txt"

    prospectpro = np.genfromtxt(sys.argv[9])#"dataSpec_PDB.txt"
    soil = np.genfromtxt(sys.argv[10])#"soil_reflectance.txt"
    TOCirr = np.genfromtxt(sys.argv[11])#"atmo.txt"
    """
    
    d = dc.DalecData(2019, 2022, site, datas, pars, ci_flag, 'nee')
    m = mc.DalecModel(d)
    
    model_output, nee_y, lst_y, fpar_y, refl_y = m.mod_list(d.xb)
    np.savetxt("test.csv", model_output, delimiter=",")

if __name__ == '__main__':
    site = "HARV"
    
    data1 = pd.read_csv("../../data/driving/{0}.csv".format(site), na_values="nan") 
    data2 = pd.read_csv("../../data/verify/{0}_nee.csv".format(site), na_values="nan")
    data3 = pd.read_csv("../../data/verify/{0}_brf.csv".format(site), na_values="nan")
    data4 = pd.read_csv("../../data/parameters/{0}_wds.csv".format(site), na_values="nan")
    data5 = pd.read_csv("../../data/parameters/{0}_traits.csv".format(site), na_values="nan", header=None, index_col=0) 

    B = pickle.load(open("../../src/model/b_edc.p", 'rb'),encoding='iso-8859-1')  #'b_edc.p' Uses background error cov. matrix B created using ecological    
    
    rsr_red = np.genfromtxt("../../data/parameters/rsr_red.txt")
    rsr_nir = np.genfromtxt("../../data/parameters/rsr_nir.txt")
    rsr_sw1 = np.genfromtxt("../../data/parameters/rsr_swir1.txt")
    rsr_sw2 = np.genfromtxt("../../data/parameters/rsr_swir2.txt")
    
    prospectpro = np.loadtxt("../../data/parameters/dataSpec_PDB.txt")
    soil = np.genfromtxt("../../data/parameters/soil_reflectance.txt")
    TOCirr = np.loadtxt("../../data/parameters/atmo.txt", skiprows=1)
    
    datas = [data1, data2, data3, data4, data5, B, rsr_red, rsr_nir, rsr_sw1, rsr_sw2, prospectpro, soil, TOCirr] 
    pars  = pd.read_csv("../../data/parameters/HARV_pars.csv").iloc[0]
    main(datas, pars)
