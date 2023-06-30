from load import load_model, read_vars, read_pars
from spot_class import spot_setup
import spotpy
import numpy as np

import warnings
warnings.filterwarnings("ignore")

data_root = "../../data/par_set1/"
data_paths = [data_root + "HARV.csv", data_root + f"0_model_output_hourly.pkl"]
prospectpro_path = "../model/support/dataSpec_PDB_resample.txt"

lookback_periods = 5
batch_size = 32

model, x_pars, x_vars, hidden_dim_list, n_layers_list, output_dim_list = load_model()
hourly_df_vars, daily_df_vars, label_scaler, obs_arr = read_vars(data_paths, hour_length=17544)
daily_df, hourly_df = read_pars(hourly_df_vars, daily_df_vars, x_pars, prospectpro_path)

spot = spot_setup(model, obs_arr, hourly_df, daily_df, batch_size, x_vars, x_pars, lookback_periods, hidden_dim_list, n_layers_list, output_dim_list, label_scaler)
"""
RUB, CB6F, BallBerry0 = 0, 0, 1
pars = [RUB, CB6F, BallBerry0]
prd_arr1 = spot.simulation(pars)
prd_arr2 = spot.simulation(pars)
import matplotlib.pyplot as plt
# Plotting
fig, axs = plt.subplots(2, figsize=(10, 6))
axs[0].plot(prd_arr1, color='red')
axs[1].plot(prd_arr2, color='red')
plt.show()
"""
"""
MAE_min = 9999
RUB_min = 9999
CB6F_min = 9999
BallBerry0_min = 9999
count = 0
for RUB in np.linspace(0.0, 120.0, num=10):
    for CB6F in np.linspace(0.0, 150.0, num=10):
        for BallBerry0 in np.linspace(0.0, 1.0, num=4):
            pars = [RUB, CB6F, BallBerry0]
            prd_arr = spot.simulation(pars)

            MAE = np.mean(np.abs(prd_arr - obs_arr))
            if MAE < MAE_min:
                MAE_min = MAE
                RUB_min = RUB
                CB6F_min = CB6F
                BallBerry0_min = BallBerry0

            print(count, MAE_min, RUB_min, CB6F_min, BallBerry0_min)

            count += 1
print("---------Loop--------")
print(MAE_min)
print(RUB_min, CB6F_min, BallBerry0_min)
"""
sampler = spotpy.algorithms.sceua(spot, dbname='SCEUA', dbformat='csv')
rep = 5000
sampler.sample(rep)
