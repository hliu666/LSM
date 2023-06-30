import os
import matplotlib.pyplot as plt
import numpy as np
import data_class as dc
import mod_class as mc
import joblib 

root = os.path.dirname(os.path.dirname(os.getcwd()))
"""
d = dc.DalecData(1999, 2000, 'nee')
m = mc.DalecModel(d)
model_output = m.mod_list(d.xb)
#assimilation_results = m.find_min_tnc(d.xb)
"""
import time
start = time.time()

"""
CI_flag
    0: CI varied with the zenith angle
    1: CI as a constant 
    2: Without considering CI effect            
"""
       
ci_flag = 1
site = "HARV"
#d = dc.DalecData(2019, 2022, site, ci_flag, 'nee')

print("-----------start-------------")

import cProfile
import pstats
import io
pr = cProfile.Profile()
pr.enable()

d = dc.DalecData(2019, 2022, site, ci_flag, 'nee')
m = mc.DalecModel(d)
model_output, nee_y, lst_y, refl_y = m.mod_list(d.xb)

pr.disable()
s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
ps.print_stats()

with open('result.txt', 'w+') as f:
    f.write(s.getvalue())


plt.figure(figsize=(8,4))
plt.title('NEE')
plt.plot(nee_y, 'k.')

x = np.array(d.vrfy_data['nee'][0:26280])
xd = x
#xd = np.array([np.nanmean(x[m: m+24]) for m in range(0, len(x), 24)])
plt.plot(xd, 'r+')

plt.figure(figsize=(8,4))
plt.title('LAI')
plt.plot(model_output[:,-1], 'k.')
plt.plot((d.lai_data["year"]-2019)*365+d.lai_data["doy"], d.lai_data["LAI"], 'r+')
"""
plt.figure(figsize=(12,4))
plt.title('GPP')
gpp = model_output[:,-2]
plt.plot(gpp, 'k.')

import pandas as pd
data = pd.read_csv("../../data/verify/HARV_gpp.csv", na_values="nan") 
data = data[(data['gpp'] >= 0)]
plt.plot(data['gpp'], 'r+')

plt.figure(figsize=(12,4))
plt.title('MODIS LST-Simulation LST')
plt.plot((d.lst_data["year"]-2019)*365+d.lst_data["doy"], d.lst_data["LST_Day_1km"]-lst_y[d.index], 'k.')
plt.axhline(y=0.0, color='r', linestyle='-')

plt.figure(figsize=(12,4))
plt.title('LST')
plt.plot((d.lst_data["year"]-2019)*365+d.lst_data["doy"], lst_y[d.index], 'k.')
plt.plot((d.lst_data["year"]-2019)*365+d.lst_data["doy"], d.lst_data["LST_Day_1km"], 'r+')

plt.figure(figsize=(12,4))
plt.title('sur_refl_b01')
plt.scatter((d.brf_data["year"]-2019)*365+d.brf_data["doy"], d.brf_data["sur_refl_b01"], color = 'r', alpha=0.5)
plt.scatter((d.brf_data["year"]-2019)*365+d.brf_data["doy"], refl_y[:,0], color = 'b', alpha=0.5)

plt.figure(figsize=(12,4))
plt.title('sur_refl_b02')
plt.scatter((d.brf_data["year"]-2019)*365+d.brf_data["doy"], d.brf_data["sur_refl_b02"], color = 'r', alpha=0.5)
plt.scatter((d.brf_data["year"]-2019)*365+d.brf_data["doy"], refl_y[:,1], color = 'b', alpha=0.5)
"""
"""
output
"""
joblib.dump(model_output,  "../../data/output/model/out_ci{0}_{1}.pkl".format(ci_flag, site))
joblib.dump(nee_y,  "../../data/output/model/nee_ci{0}_{1}.pkl".format(ci_flag, site))
joblib.dump(lst_y,  "../../data/output/model/lst_ci{0}_{1}.pkl".format(ci_flag, site))
joblib.dump(refl_y, "../../data/output/model/refl_ci{0}_{1}.pkl".format(ci_flag, site))

joblib.dump(d.lai_data,  "../../data/output/verify/d_lai{0}_{1}.pkl".format(ci_flag, site))
joblib.dump(d.lst_data,  "../../data/output/verify/d_lst{0}_{1}.pkl".format(ci_flag, site))
joblib.dump(d.vrfy_data, "../../data/output/verify/d_nee{0}_{1}.pkl".format(ci_flag, site))
joblib.dump(d.brf_data,  "../../data/output/verify/d_refl{0}_{1}.pkl".format(ci_flag, site))

end = time.time()
print(end - start)

