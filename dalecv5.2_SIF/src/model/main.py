import os
import data_class as dc
import mod_class as mc

root = os.path.dirname(os.path.dirname(os.getcwd()))
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

print("-----------start-------------")

import cProfile
import pstats
import io
pr = cProfile.Profile()
pr.enable()

d = dc.DalecData(2019, 2022, site, ci_flag, 'nee')
m = mc.DalecModel(d)
model_output, nee_y, lst_y, sif_y = m.mod_list(d.xb)

pr.disable()
s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
ps.print_stats()

with open('result.txt', 'w+') as f:
    f.write(s.getvalue())

end = time.time()
print(end - start)

