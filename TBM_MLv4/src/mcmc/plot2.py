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

RUB, CB6F, BallBerry0 = 111.69, 5.32, 0.51

pars = [RUB, CB6F, BallBerry0]
prd_arr = spot.simulation(pars)

import matplotlib.pyplot as plt

fig= plt.figure(figsize=(16,9))
ax = plt.subplot(1,1,1)
ax.plot(prd_arr,color='black',linestyle='solid', label='Simulation data')
ax.plot(obs_arr[0:16896],'r.',markersize=3, label='Observation data')
plt.xlabel('Number of Observation Points')
plt.ylabel ('Discharge [l s-1]')
plt.legend(loc='upper right')
plt.show()