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
#d = dc.DalecData(2019, 2022, site, ci_flag, 'nee')

print("-----------start-------------")

import cProfile
import pstats
import io
pr = cProfile.Profile()
pr.enable()

d = dc.DalecData(2019, 2020, site, ci_flag, 'nee')
m = mc.DalecModel(d)
model_output, nee_y, sif_u_y, sif_h_y = m.mod_list(d.xb)

pr.disable()
s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
ps.print_stats()

with open('result.txt', 'w+') as f:
    f.write(s.getvalue())

"""
plt.figure(figsize=(8,4))
plt.title('NEE')
plt.plot(nee_y, 'k.')

"""

"""
output
"""
joblib.dump(sif_u_y,  "../../data/output/model/SIFu_ci{0}_{1}.pkl".format(ci_flag, site))
joblib.dump(sif_h_y,  "../../data/output/model/SIFh_ci{0}_{1}.pkl".format(ci_flag, site))

end = time.time()
print(end - start)

