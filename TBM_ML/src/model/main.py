from data import Data
from dataloader import Dataloader
from model_train import train
from model_evaluate import evaluate
import warnings

# warnings.simplefilter("error")

"""
1. Read data and select interested fields

hourly_data_paths:
0.OBS_path; 1.sifu_path; 2.sifh_path; 3.lai_path 
"""
hourly_data_paths = ["../../data/OBS.csv", "../../data/SIFu_ci1_OBS.pkl", "../../data/SIFh_ci1_OBS.pkl",
                     "../../data/model_output_hourly.pkl", "../../data/model_output_daily.pkl"]
daily_data_path = "../../data/model_output_daily.pkl"
spectral_data_path = "../../data/mod_list_spectral.pkl"
pars = ["support/dataSpec_PDB.txt"]

x_carp_vars = ['GPP', 'LST', 'doy']
x_carp_pars = ['clspan', 'lma', 'f_auto', 'f_fol', 'd_onset', 'cronset', 'd_fall', 'crfall', 'CI', 'LiDf']
y_carp_vars = ['LAI']

x_rtmo_vars = ['LAI', 'SZA', 'VZA', 'SAA']
x_rtmo_pars = ['Cab', 'Car', 'Cm', 'Cbrown', 'Cw', 'Ant', 'CI', 'LiDf']  + [f'leaf_b{i + 400}' for i in range(2101)]
y_rtmo_vars = ['fPAR']  + [f'canopy_b{i + 400}' for i in range(2101)]

x_enba_vars = ['LAI', 'SW', 'TA', 'SZA', 'VZA', 'SAA', 'wds']
x_enba_pars = []
y_enba_vars = ['Tcu', 'Tch', 'Tsu', 'Tsh']

x_rtmt_vars = ['LAI', 'Tcu', 'Tch', 'Tsu', 'Tsh', 'SZA', 'VZA', 'SAA']
x_rtmt_pars = ['CI', 'LiDf', 'rho', 'tau', 'rs']
y_rtmt_vars = ['LST']

x_bicm_vars = ['LAI', 'fPAR', 'PAR', 'LST', 'VPD']
x_bicm_pars = ['RUB', 'CB6F', 'Rdsc', 'gm', 'e', 'BallBerrySlope', 'BallBerry0']
y_bicm_vars = ['GPP', 'NEE', 'fqe_u', 'fqe_h']

x_rtms_vars = ['LAI', 'fPAR', 'PAR', 'fqe_u', 'fqe_h', 'SZA', 'VZA', 'SAA']
x_rtms_pars = ['CI', 'LiDf', 'eta']
y_rtms_vars = ['SIFu', 'SIFh']

x_vars = [x_carp_vars, x_rtmo_vars, x_enba_vars, x_rtmt_vars, x_bicm_vars, x_rtms_vars]
x_pars = [x_carp_pars, x_rtmo_pars, x_enba_pars, x_rtmt_pars, x_bicm_pars, x_rtms_pars]
y_vars = [y_carp_vars, y_rtmo_vars, y_enba_vars, y_rtmt_vars, y_bicm_vars, y_rtms_vars]

dC = Data(pars, hourly_data_paths, daily_data_path, spectral_data_path, x_pars)

"""
2. Normalize the data
"""
# define machine learning parameters
lookback_daily = 14
lookback_hourly = 24
batch_size = 32
test_portion = 0.2

dL = Dataloader(dC, x_vars, x_pars, y_vars, lookback_daily, lookback_hourly, batch_size, test_portion)

"""
3. Train Machine learning model
"""
carp_hidden_dim, rtmo_hidden_dim, enba_hidden_dim, rtmt_hidden_dim, bicm_hidden_dim, rtms_hidden_dim = 32, 32, 32, 32, 32, 32
carp_n_layers, rtmo_n_layers, enba_n_layers, rtmt_n_layers, bicm_n_layers, rtms_n_layers = 4, 4, 4, 4, 4, 4
learn_rate = 0.001
EPOCHS = 5

hidden_dim_list = [carp_hidden_dim, rtmo_hidden_dim, enba_hidden_dim, rtmt_hidden_dim, bicm_hidden_dim, rtms_hidden_dim]
n_layers_list = [carp_n_layers, rtmo_n_layers, enba_n_layers, rtmt_n_layers, bicm_n_layers, rtms_n_layers]

model = train(dL, batch_size, n_layers_list, learn_rate, hidden_dim_list, EPOCHS)

evaluate(model, dL, hidden_dim_list, n_layers_list, batch_size)
