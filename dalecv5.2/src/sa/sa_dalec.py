import os
import numpy as np
import joblib

# this will fail if SALib isn't properly installed
from SALib.sample import saltelli
from SALib.analyze import sobol

fields = ["clab", "cf", "cr", "cw", "cl", "cs",\
          "p0", "p1", "p2", "p3", "p4", "p5", "p6",\
          "p7", "p8", "p9", "p10", "p11", "p12","p13",\
          "p14", "p15", "p16", "p17", "BallBerrySlope",\
          "Cab", "Car", "Cbrown", "Cw", "Ant"]

# Specify the model inputs and their bounds. The default probability
# distribution is a uniform distribution between lower and upper bounds.
#%% Leaf traits parameter sensitivity analysis
problem = {
    "num_vars": 30, 
    "names": ["clab", "cf", "cr", "cw", "cl", "cs",\
              "p0", "p1", "p2", "p3", "p4", "p5", "p6",\
              "p7", "p8", "p9", "p10", "p11", "p12","p13",\
              "p14", "p15", "p16", "p17", "BallBerrySlope",\
              "Cab", "Car", "Cbrown", "Cw", "Ant"], 
        
    "bounds": [[10, 1000], [0, 1000], [10, 1000], [3000, 4000], [10, 1000], [1000, 1e5],\
               [1e-5, 1e-2], [0.3, 0.7], [0.01, 0.5], [0.01, 0.5], [1.0001, 5], [2.5e-5, 1e-3], [1e-4, 1e-2],\
               [1e-4, 1e-2], [1e-7, 1e-3], [0.018, 0.08], [60, 150], [0.01, 0.5], [10, 100], [242, 332],\
               [10, 100], [50, 100], [0.7, 0.9], [0, 100], [0.0, 20.0],\
               [0, 40], [0, 10], [0, 1], [0, 0.1], [0, 30]]
}

N = 2048

# generate the input sample
sample = saltelli.sample(problem, N)
Y1 = np.empty([sample.shape[0]])
Y2 = np.empty([sample.shape[0]])
Y3 = np.empty([sample.shape[0]])
Y4 = np.empty([sample.shape[0]])
Y5 = np.empty([sample.shape[0]])
Y6 = np.empty([sample.shape[0]])

path = r"D:\sa"
for i in range(len(Y1)):
    if i%10000 == 0:
        print(i)
    if os.path.exists(os.path.join(path, "nee_ci1_HARV_Dalec_{0}.pkl".format(i))) and  \
        os.path.exists(os.path.join(path, "out_ci1_HARV_Dalec_{0}.pkl".format(i))) and \
        os.path.exists(os.path.join(path, "lst_ci1_HARV_Dalec_{0}.pkl".format(i))) and \
        os.path.exists(os.path.join(path, "refl_ci1_HARV_Dalec_{0}.pkl".format(i))):
            
        s1 = joblib.load(os.path.join(path, "nee_ci1_HARV_Dalec_{0}.pkl".format(i)))
        s2 = joblib.load(os.path.join(path, "out_ci1_HARV_Dalec_{0}.pkl".format(i)))
        s3 = joblib.load(os.path.join(path, "lst_ci1_HARV_Dalec_{0}.pkl".format(i)))
        s4 = joblib.load(os.path.join(path, "refl_ci1_HARV_Dalec_{0}.pkl".format(i)))
        
        #if s1 is not None:
        s1d = s1 #np.nanmean(s1.reshape(24,1095), axis=1)#[np.nanmean(s1[x: x+24]) for x in range(0, len(s1), 24)]
        s2d = s2[:,-2]
        s3d = s3 #np.nanmean(s3.reshape(24,1095), axis=1)
        s4d = s4[:, 0]
        s5d = s4[:, 1]
  
        Y1[i] = np.clip(np.nanmean(s1d), -10, 2) #NEE
        Y2[i] = np.clip(np.nanmean(s2d), 0, 20)   #LAI
        Y3[i] = np.clip(np.nanmean(s3d), 0, 10)  #LST
        Y4[i] = np.clip(np.nanmean(s4d), 0, 1)   #red
        Y5[i] = np.clip(np.nanmean(s5d), 0, 1)   #nir

        #print(np.nanmean(s1d), np.nanmean(s2d), np.nanmean(s3d), np.nanmean(s4d), np.nanmean(s5d))
    #else:
    #    print(i)
# estimate the sensitivity indices, using the Sobol' method
"""
Y1:NEE, Y2:fPAR, Y3:LAI
"""
sensitivity1 = sobol.analyze(problem, Y1)
sensitivity2 = sobol.analyze(problem, Y2)
sensitivity3 = sobol.analyze(problem, Y3)
sensitivity4 = sobol.analyze(problem, Y4)
sensitivity5 = sobol.analyze(problem, Y5)

joblib.dump(sensitivity1, "sensitivity_nee_year.pkl")
joblib.dump(sensitivity2, "sensitivity_lai_year.pkl")
joblib.dump(sensitivity3, "sensitivity_lst_year.pkl")
joblib.dump(sensitivity4, "sensitivity_red_year.pkl")
joblib.dump(sensitivity5, "sensitivity_nir_year.pkl")



