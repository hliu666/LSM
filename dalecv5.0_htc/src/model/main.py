import pandas as pd
import numpy as np
import data_class as dc
import mod_class as mc
import sys
import pickle
import joblib

def main():   
    ci_flag = 1
    site = "HARV"
    
    print(sys.argv)
    
    flux_arr = pd.read_csv(sys.argv[1], na_values="nan") #"{0}.csv"
    vrfy_arr = pd.read_csv(sys.argv[2], na_values="nan") #"{0}_nee.csv"
    brf_arr = pd.read_csv(sys.argv[3], na_values="nan") #"{0}_brf.csv"
    wds_arr = pd.read_csv(sys.argv[4], na_values="nan") #"{0}_wds.csv"
    tis_arr = pd.read_csv(sys.argv[5], na_values="nan", header=None, index_col=0) #"{0}_traits.csv"    

    B = pickle.load(open(sys.argv[6], 'rb'),encoding='iso-8859-1')  #'b_edc.p' 
    
    rsr_red = np.genfromtxt(sys.argv[7])#"rsr_red.txt"
    rsr_nir = np.genfromtxt(sys.argv[8])#"rsr_nir.txt"
    rsr_sw1 = np.genfromtxt(sys.argv[9])#"rsr_sw1.txt"
    rsr_sw2 = np.genfromtxt(sys.argv[10])#"rsr_sw2.txt"
    
    prospectpro = np.genfromtxt(sys.argv[11])#"dataSpec_PDB.txt"
    soil = np.genfromtxt(sys.argv[12])#"soil_reflectance.txt"
    TOCirr = np.loadtxt(sys.argv[13], skiprows=1)#"atmo.txt"
       
    datas = [flux_arr, vrfy_arr, brf_arr, wds_arr, tis_arr, B, rsr_red, rsr_nir, rsr_sw1, rsr_sw2, prospectpro, soil, TOCirr] 
    ps = ['Icar','Leaf','Dalec']
    i, js, p = int(sys.argv[15]), int(sys.argv[16]), ps[int(sys.argv[17])]
    for j in range(0, js):
        k = i+j
        pars = pd.read_csv(sys.argv[14]).iloc[k]
        
        d = dc.DalecData(2019, 2022, site, datas, pars, ci_flag, 'nee')
        m = mc.DalecModel(d)
        
        model_output, nee_y, lst_y, refl_y = m.mod_list(d.xb)
        
        joblib.dump(model_output,  "out_ci{0}_{1}_{2}_{3}.pkl".format(ci_flag, site, p, k))
        joblib.dump(nee_y,  "nee_ci{0}_{1}_{2}_{3}.pkl".format(ci_flag, site, p, k))
        joblib.dump(lst_y,  "lst_ci{0}_{1}_{2}_{3}.pkl".format(ci_flag, site, p, k))
        joblib.dump(refl_y, "refl_ci{0}_{1}_{2}_{3}.pkl".format(ci_flag, site, p, k))
    
if __name__ == '__main__':
   main()
