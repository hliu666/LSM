import os
import joblib
import time
import cProfile
import pstats
import io
import pandas as pd
import numpy as np
import scipy.io

from data import TBM_Data
from parameter import TBM_Pars
from model import TBM_Model


def main(lat, lon, start_yr, end_yr, output_dim1, output_dim2, driving_data):
    start = time.time()
    print("-----------start-------------")
    pr = cProfile.Profile()
    pr.enable()

    p = TBM_Pars()
    d = TBM_Data(p, lat, lon, start_yr, end_yr, driving_data)
    m = TBM_Model(d, p)

    model_output_daily, model_output_hourly = m.mod_list(output_dim1, output_dim2)

    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
    ps.print_stats()

    with open('result.txt', 'w+') as f:
        f.write(s.getvalue())

    joblib.dump(model_output_daily, "../../data/output/model/model_output_daily.pkl")
    joblib.dump(model_output_hourly, "../../data/output/model/model_output_hourly.pkl")

    end = time.time()
    print(end - start)


site = "OBS"
root = os.path.dirname(os.path.dirname(os.getcwd()))
flux_arr = pd.read_csv(root + os.sep + "data/driving/{0}.csv".format(site), na_values="nan")
rsr_red = np.genfromtxt(root + os.sep + "src/model/support/rsr_red.txt")
rsr_nir = np.genfromtxt(root + os.sep + "src/model/support/rsr_nir.txt")
rsr_sw1 = np.genfromtxt(root + os.sep + "src/model/support/rsr_swir1.txt")
rsr_sw2 = np.genfromtxt(root + os.sep + "src/model/support/rsr_swir2.txt")
prospectpro = np.loadtxt(root + os.sep + "src/model/support/dataSpec_PDB.txt")
soil = np.genfromtxt(root + os.sep + "src/model/support/soil_reflectance.txt")
TOCirr = np.loadtxt(root + os.sep + "src/model/support/atmo.txt", skiprows=1)
phiI = scipy.io.loadmat(root + os.sep + 'src/model/support/phiI.mat')['phiI']
phiII = scipy.io.loadmat(root + os.sep + 'src/model/support/phiII.mat')['phiII']
driving_data = [flux_arr, rsr_red, rsr_nir, rsr_sw1, rsr_sw2, prospectpro, soil, TOCirr, phiI, phiII]

lat, lon = 53.9872, -105.1178
start_yr, end_yr = 2019, 2021
output_dim1, output_dim2 = 8, 16
main(lat, lon, start_yr, end_yr, output_dim1, output_dim2, driving_data)
