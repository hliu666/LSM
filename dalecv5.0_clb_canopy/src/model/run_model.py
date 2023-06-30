import data_class as dc
import mod_class as mc
import numpy as np
  
"""
CI_flag
    0: CI varied with the zenith angle
    1: CI as a constant 
    2: Without considering CI effect            
"""
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def run_model(pars):      
    ci_flag = 1
    site = "HARV"
    
    d = dc.DalecData(2019, 2022, site, ci_flag, pars)
    m = mc.DalecModel(d)
    refl_y,_ = m.mod_list(d.lai, d.sai)
    refl_y = refl_y[:,[0,1,5,6]]
    
    print(rmse(np.array(d.brf_data[['sur_refl_b01']]),refl_y[:,0])*5)
    print(rmse(np.array(d.brf_data[['sur_refl_b02']]),refl_y[:,1]))
    
    return refl_y

#import time
#start = time.time()

#import cProfile
#cProfile.run('dc.DalecData(2019, 2022, site, ci_flag, "nee")')
#d = dc.DalecData(2019, 2022, site, ci_flag, 'nee')
#m = mc.DalecModel(d)
#cProfile.run('m.mod_list(d.xb)')
"""
import spotpy
dbname = "SCEUA"
results = spotpy.analyser.load_csv_results('{0}'.format(dbname))

bestindex,bestobjf = spotpy.analyser.get_minlikeindex(results)
best_model_run = results[bestindex]
fields=['parp0', 'parp1', 'parp2', 'parp3', 'parp4', 'parp5', 'parp6', 'parp7', 'parp8', 'parp9', 'parp10', 'parp11',\
        'parp12', 'parp13', 'parp14', 'parp15', 'parp16']
pars = list(best_model_run[fields])
        
model_output, nee_y = run_model(pars)
import matplotlib.pyplot as plt
plt.figure(figsize=(8,4))
plt.title('NEE')
plt.plot(nee_y, 'k.')

"""

#end = time.time()
#print(end - start)

