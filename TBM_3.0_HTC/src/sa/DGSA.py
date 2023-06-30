import numpy as np
import joblib
from SALib.sample import saltelli
import os
from sklearn.preprocessing import MinMaxScaler

problem = {
    "num_vars": 30,
    "names": ["clab", "cf", "cr", "cw", "cl", "cs", \
              "p0", "p1", "p2", "p3", "p4", "p5", "p6", \
              "p7", "p8", "p9", "p10", "p11", "p12", "p13", \
              "p14", "p15", "p16", "p17", "BallBerrySlope", \
              "Cab", "Car", "Cbrown", "Cw", "Ant"],

    "bounds": [[10, 1000], [0, 1000], [10, 1000], [3000, 4000], [10, 1000], [1000, 1e5], \
               [1e-5, 1e-2], [0.3, 0.7], [0.01, 0.5], [0.01, 0.5], [1.0001, 5], [2.5e-5, 1e-3], [1e-4, 1e-2], \
               [1e-4, 1e-2], [1e-7, 1e-3], [0.018, 0.08], [60, 150], [0.01, 0.5], [10, 100], [242, 332], \
               [10, 100], [50, 100], [0.7, 0.9], [0, 100], [0.0, 20.0], \
               [0, 40], [0, 10], [0, 1], [0, 0.1], [0, 30]]
}

N = 2048
# generate the input sample
sample = saltelli.sample(problem, N)
# Initialize a scaler
scaler = MinMaxScaler()
# Fit the scaler to the data and transform the data
scaled_array = scaler.fit_transform(sample)

# Load model parameters
length = 20000
parameters = sample[0:length, :]
responses = np.empty((parameters.shape[0], 1096), dtype='float64')

path = r"G:\sa"
for i in range(0, length):  # len(sample)
    if i % 1000 == 0:
        print(i)
    output_file = os.path.join(path, "out_ci1_HARV_Dalec_{0}.pkl".format(i))
    if os.path.exists(output_file):
        s1 = joblib.load(output_file)
        responses[i, :] = s1[:, -1]

# Now, calculate the euclidean distances between model responses
from scipy.spatial.distance import pdist, squareform

distances = pdist(responses, metric='euclidean')
distances = squareform(distances)

# Cluster the responses using KMedoids
from pyDGSA.cluster import KMedoids

n_clusters = 2
clusterer = KMedoids(n_clusters=n_clusters, max_iter=100, tol=1e-3)
labels, medoids = clusterer.fit_predict(distances)

# Calculate mean sensitivity across clusters
from pyDGSA.dgsa import dgsa

parameter_names = ["Clab", "Cfol", "Croo", "Cwoo", "Clit", "Csom",
                   "fauto", "flab", "ffol", "froo", "Θwoo", "Θroo", "Θlit",
                   "Θsom", "Θmin", "Θ", "donset", "dfall", "cronset", "crfall",
                   "clspan", "Cab", "Car", "Cw", "Cbrown", "Ant", "LIDFa", "CI", "Vcmax25", "m"]

mean_sensitivity = dgsa(parameters, labels, parameter_names=parameter_names, quantile=0.99, n_boots=100,
                        confidence=True)

# Generate a pareto plot of the results
from pyDGSA.plot import vert_pareto_plot

import matplotlib.pyplot as plt

fig, ax = vert_pareto_plot(mean_sensitivity, np_plot='+1', confidence=True)
plt.show()
