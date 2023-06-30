import pandas as pd
from vars import Var
from pars import Par
from data import Data
from dataloader import Dataloader
from GRU import Model
from plot import cal_r2_bicm, plot_carp, plot_enba, plot_rtms, plot_rtmo

"""
Load data root and select interested fields
"""
data_root = "../../data/par_set1/"
hidden_dim_list = [8, 16, 32, 64]
n_layers_list = [1, 2, 4, 8]
lookback_periods_list = [1, 3, 5, 7]
batch_size_list = [16, 32, 64, 128]

def statistics(row):
    hidden_dim, n_layer, lookback_periods, batch_size = int(row[0]), int(row[1]), int(row[2]), int(row[3])
    print(hidden_dim, n_layer, lookback_periods, batch_size)
    p = Par(hidden_dim, n_layer, lookback_periods, batch_size)
    v = Var()

    i = 63
    dC = Data(i, p, data_root)
    dL = Dataloader(dC, v, p)
    m = Model(dL, p)
    m.load(hidden_dim, n_layer, lookback_periods, batch_size)

    r1, r2 = cal_r2_bicm(m.bicm_model, dL.bicm_test_data, dL.bicm_label_scaler)

    return r1, r2

df = pd.read_csv('out/sta_time.csv')
df[['GPP_R2', 'NEE_R2']] = df.apply(lambda row: pd.Series(statistics(row)), axis=1)
df.to_csv('out/sta_time1.csv', index=False)
