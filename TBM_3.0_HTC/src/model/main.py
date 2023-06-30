import joblib
import pandas as pd
import numpy as np
import scipy.io
import sys

from data import TBM_Data
from parameter import TBM_Pars
from model import TBM_Model

def main():
    site = "HARV"
    flux_arr = pd.read_csv(sys.argv[1], na_values="nan")  # "data/driving/{0}.csv".format(site)
    rsr_red = np.genfromtxt(sys.argv[2])  # "src/model/support/rsr_red.txt"
    rsr_nir = np.genfromtxt(sys.argv[3])  # "src/model/support/rsr_nir.txt"
    rsr_sw1 = np.genfromtxt(sys.argv[4])  # "src/model/support/rsr_swir1.txt"
    rsr_sw2 = np.genfromtxt(sys.argv[5])  # "src/model/support/rsr_swir2.txt"
    prospectpro = np.loadtxt(sys.argv[6])  # "src/model/support/dataSpec_PDB.txt"
    soil = np.genfromtxt(sys.argv[7])  # "src/model/support/soil_reflectance.txt"
    TOCirr = np.loadtxt(sys.argv[8], skiprows=1)  # "src/model/support/atmo.txt"
    phiI = scipy.io.loadmat(sys.argv[9])['phiI']  # 'src/model/support/phiI.mat'
    phiII = scipy.io.loadmat(sys.argv[10])['phiII']  # 'src/model/support/phiII.mat'
    driving_data = [flux_arr, rsr_red, rsr_nir, rsr_sw1, rsr_sw2, prospectpro, soil, TOCirr, phiI, phiII]
    # lat, lon = 53.9872, -105.1178
    lat, lon = 42.54, -72.17
    start_yr, end_yr = 2019, 2021
    output_dim1, output_dim2 = 8, 6

    i, js = int(sys.argv[12]), int(sys.argv[13])
    for j in range(0, js):
        k = i + j
        pars = pd.read_csv(sys.argv[11]).iloc[k]
        p = TBM_Pars(pars)
        d = TBM_Data(p, lat, lon, start_yr, end_yr, driving_data)
        m = TBM_Model(d, p)

        model_output_daily, model_output_hourly = m.mod_list(output_dim1, output_dim2)

        joblib.dump(model_output_daily, "{0}_model_output_daily.pkl".format(k), compress=3)
        joblib.dump(model_output_hourly, "{0}_model_output_hourly.pkl".format(k), compress=3)

if __name__ == '__main__':
   main()
