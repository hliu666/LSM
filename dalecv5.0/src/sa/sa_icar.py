import os
import numpy as np
import joblib

# this will fail if SALib isn't properly installed
from SALib.sample import saltelli
from SALib.analyze import sobol

problem = {
    "num_vars": 6, 
    "names": ["clab", "cf", "cr", "cw", "cl", "cs"], 
    "bounds": [[20, 2000], [0, 2000], [0, 2000], [0, 1e5], [0, 2000], [100, 2e5]]
}

N = 700

# generate the input sample
sample = saltelli.sample(problem, N)
Y1 = np.empty([sample.shape[0]])
Y2 = np.empty([sample.shape[0]])
Y3 = np.empty([sample.shape[0]])
Y4 = np.empty([sample.shape[0]])
Y5 = np.empty([sample.shape[0]])
Y6 = np.empty([sample.shape[0]])

path = r"I:\sa"
for i in range(len(Y1)):
    if os.path.exists(os.path.join(path, "nee_ci1_HARV_Icar_{0}.pkl".format(i))):
        s1 = joblib.load(os.path.join(path, "nee_ci1_HARV_Icar_{0}.pkl".format(i)))
        s2 = joblib.load(os.path.join(path, "fpar_ci1_HARV_Icar_{0}.pkl".format(i)))
        s3 = joblib.load(os.path.join(path, "out_ci1_HARV_Icar_{0}.pkl".format(i)))
        s4 = joblib.load(os.path.join(path, "lst_ci1_HARV_Icar_{0}.pkl".format(i)))
        s5 = joblib.load(os.path.join(path, "refl_ci1_HARV_Icar_{0}.pkl".format(i)))
        
        s1d = [np.nansum(s1[x: x+24]) for x in range(0, len(s1), 24)]
        s2d = [np.nansum(s2[x: x+24]) for x in range(0, len(s2), 24)]
        s3d = s3[:,-2]+s3[:,-1]
        s4d = [np.nansum(s4[x: x+24]) for x in range(0, len(s4), 24)]
        s5d = s5[:,0]
        s6d = s5[:,1]
        
        Y1[i] = np.nanmean(s1d)
        Y2[i] = np.nanmean(s2d)
        Y3[i] = np.nanmean(s3d)
        Y4[i] = np.nanmean(s4d)
        Y5[i] = np.nanmean(s5d)   
        Y6[i] = np.nanmean(s6d)   
        
    else:
        print(i)
# estimate the sensitivity indices, using the Sobol' method
"""
Y1:NEE, Y2:fPAR, Y3:LAI
"""
sensitivity1 = sobol.analyze(problem, Y1)
sensitivity2 = sobol.analyze(problem, Y2)
sensitivity3 = sobol.analyze(problem, Y3)
sensitivity4 = sobol.analyze(problem, Y4)
sensitivity5 = sobol.analyze(problem, Y5)
sensitivity6 = sobol.analyze(problem, Y6)

joblib.dump(sensitivity1, "./../../data/output/sa/sensitivity_nee_Icar.pkl")
joblib.dump(sensitivity2, "./../../data/output/sa/sensitivity_fpar_Icar.pkl")
joblib.dump(sensitivity3, "./../../data/output/sa/sensitivity_lai_Icar.pkl")
joblib.dump(sensitivity4, "./../../data/output/sa/sensitivity_lst_Icar.pkl")
joblib.dump(sensitivity5, "./../../data/output/sa/sensitivity_red_Icar.pkl")
joblib.dump(sensitivity6, "./../../data/output/sa/sensitivity_nir_Icar.pkl")



