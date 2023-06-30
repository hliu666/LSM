import os
import numpy as np
import joblib

# this will fail if SALib isn't properly installed
from SALib.sample import saltelli
from SALib.analyze import sobol

problem = {
    "num_vars": 18, 
    "names": ["p0", "p1", "p2", "p3", "p4", "p5", "p6",\
              "p7", "p8", "p9", "p10", "p11", "p12", "p13",\
              "p14", "p15", "Vcmax25", "BallBerrySlope"], 
        
    "bounds": [[0.001, 0.01], [0.2, 0.7], [0.01, 0.5], [0.01, 0.5], [1.01, 1.5], [2.5e-5, 1e-3], [1e-3, 1e-2],\
               [1e-4, 1e-2], [1e-7, 1e-3], [0.02, 0.08], [60, 150], [0.1, 0.5], [10, 60], [242, 332],\
               [10, 70], [70, 100], [20, 90], [10, 20]]
}

N = 2600

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
    if os.path.exists(os.path.join(path, "nee_ci1_HARV_Dalec_{0}.pkl".format(i))):
        s1 = joblib.load(os.path.join(path, "nee_ci1_HARV_Dalec_{0}.pkl".format(i)))
        s2 = joblib.load(os.path.join(path, "fpar_ci1_HARV_Dalec_{0}.pkl".format(i)))
        s3 = joblib.load(os.path.join(path, "out_ci1_HARV_Dalec_{0}.pkl".format(i)))
        s4 = joblib.load(os.path.join(path, "lst_ci1_HARV_Dalec_{0}.pkl".format(i)))
        s5 = joblib.load(os.path.join(path, "refl_ci1_HARV_Dalec_{0}.pkl".format(i)))
        
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

joblib.dump(sensitivity1, "./../../data/output/sa/sensitivity_nee_Dalec.pkl")
joblib.dump(sensitivity2, "./../../data/output/sa/sensitivity_fpar_Dalec.pkl")
joblib.dump(sensitivity3, "./../../data/output/sa/sensitivity_lai_Dalec.pkl")
joblib.dump(sensitivity4, "./../../data/output/sa/sensitivity_lst_Dalec.pkl")
joblib.dump(sensitivity5, "./../../data/output/sa/sensitivity_red_Dalec.pkl")
joblib.dump(sensitivity6, "./../../data/output/sa/sensitivity_nir_Dalec.pkl")



