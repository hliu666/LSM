import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os

def read_pkl_file(file_path):
    if os.path.exists(file_path):
        return joblib.load(file_path)
    else:
        raise FileNotFoundError(f"{file_path} does not exist")

def read_csv_file(file_path):
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        raise FileNotFoundError(f"{file_path} does not exist")

if __name__ == "__main__":
    nee_file_path = "../../data/output/model/NEE_ci1_OBS.pkl"
    gpp_file_path = "../../data/output/model/model_output.pkl"

    obs_file_path = "../../data/verify/OBS.csv"

    try:
        nee_arr = read_pkl_file(nee_file_path)
        gpp_arr = read_pkl_file(gpp_file_path)

        obs_data = read_csv_file(obs_file_path)
        nee_obs = np.array(obs_data['nee_gf'])
        gpp_obs = np.array(obs_data['gpp_gf'])

        gpp_arr_d = gpp_arr[:-1,-2]
        gpp_obs_d = np.array([np.nanmean(gpp_obs[m: m+24]) for m in range(0, len(gpp_obs), 24)])

        nee_arr_d = np.array([np.nanmean(nee_arr[m: m+24]) for m in range(0, len(nee_arr), 24)])
        nee_obs_d = np.array([np.nanmean(nee_obs[m: m+24]) for m in range(0, len(nee_obs), 24)])

        plt.figure(figsize=(10, 5))
        date_rng = pd.date_range(start='1/1/2019', end='12/25/2021', freq='D')

        plt.plot(date_rng, nee_arr_d, label='NEE simulations', marker='o')
        plt.plot(date_rng, nee_obs_d, label='NEE observations with gap-filling', marker='o')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(10, 5))
        date_rng = pd.date_range(start='1/1/2019', end='12/25/2021', freq='D')

        plt.plot(date_rng, gpp_arr_d, label='GPP simulations', marker='o')
        plt.plot(date_rng, gpp_obs_d, label='GPP observations with gap-filling', marker='o')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.show()

    except FileNotFoundError as e:
        print(e)
