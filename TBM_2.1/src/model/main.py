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
site = "OBS"

print("-----------start-------------")

import cProfile
import pstats
import io
pr = cProfile.Profile()
pr.enable()

d = dc.DalecData(2019, 2022, site, ci_flag, 'nee')

joblib.dump(d.tts,  "../../data/output/model/sza_ci{0}_{1}.pkl".format(ci_flag, site))
joblib.dump(d.tto,  "../../data/output/model/vza_ci{0}_{1}.pkl".format(ci_flag, site))
joblib.dump(d.psi,  "../../data/output/model/saa_ci{0}_{1}.pkl".format(ci_flag, site))


m = mc.DalecModel(d)
model_output, nee_y, An_y, fPAR_y, lst_y, Rns_y, fqe_pars_y, sif_u_y, sif_h_y, Tcu_y, Tch_y, Tsu_y, Tsh_y = m.mod_list(d.xb)

pr.disable()
s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
ps.print_stats()

with open('result.txt', 'w+') as f:
    f.write(s.getvalue())

plt.figure(figsize=(8,4))
plt.title('NEE')
plt.plot(nee_y, 'k.')


"""
plt.figure(figsize=(8,4))
plt.title('GPP')
plt.plot(model_output[:,-2], 'k.')

plt.figure(figsize=(8,4))
plt.title('LAI')
plt.plot(model_output[:,-1], 'k.')
"""

"""
output
"""
joblib.dump(model_output,  "../../data/output/model/model_output.pkl")
joblib.dump(nee_y,  "../../data/output/model/NEE_ci{0}_{1}.pkl".format(ci_flag, site))
joblib.dump(An_y,  "../../data/output/model/An_ci{0}_{1}.pkl".format(ci_flag, site))
joblib.dump(fPAR_y,  "../../data/output/model/fPAR_ci{0}_{1}.pkl".format(ci_flag, site))
joblib.dump(lst_y,  "../../data/output/model/lst_ci{0}_{1}.pkl".format(ci_flag, site))
joblib.dump(Rns_y,  "../../data/output/model/Rns_ci{0}_{1}.pkl".format(ci_flag, site))
joblib.dump(fqe_pars_y,  "../../data/output/model/fqe_ci{0}_{1}.pkl".format(ci_flag, site))

joblib.dump(sif_u_y,  "../../data/output/model/SIFu_ci{0}_{1}.pkl".format(ci_flag, site))
joblib.dump(sif_h_y,  "../../data/output/model/SIFh_ci{0}_{1}.pkl".format(ci_flag, site))

joblib.dump(Tcu_y,  "../../data/output/model/Tcu_ci{0}_{1}.pkl".format(ci_flag, site))
joblib.dump(Tch_y,  "../../data/output/model/Tch_ci{0}_{1}.pkl".format(ci_flag, site))
joblib.dump(Tsu_y,  "../../data/output/model/Tsu_ci{0}_{1}.pkl".format(ci_flag, site))
joblib.dump(Tsh_y,  "../../data/output/model/Tsh_ci{0}_{1}.pkl".format(ci_flag, site))

end = time.time()
print(end - start)
