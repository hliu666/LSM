from vars import Var
from pars import Par
from data import Data
from dataloader import Dataloader
from model import Model
import torch
import time
import pandas as pd

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

"""
Load data root and select interested fields
"""
data_root = "../../data/par_set1/"

d_model_list = [4, 16, 32]
n_heads_list = [4, 16, 32]
dim_feedforward_list = [8, 32, 64]
n_layers_list = [2, 8, 16]

lookback_periods_list = [3, 7]
batch_size_list = [16, 64]

par_sets = []
for d_model in d_model_list:
    for n_heads in n_heads_list:
        for dim_feedforward in dim_feedforward_list:
            for n_layers in n_layers_list:
                for lookback_periods in lookback_periods_list:
                    for batch_size in batch_size_list:
                        p = Par(d_model, n_heads, dim_feedforward, n_layers, lookback_periods, batch_size)
                        v = Var()

                        """
                        Normalize the data and train Machine learning model
                        """
                        i = 63
                        dC = Data(i, p, data_root)
                        dL = Dataloader(dC, v, p)
                        start_time = time.time()
                        m = Model(dL, p)
                        current_time = time.time()
                        par_sets.append([d_model, n_heads, dim_feedforward, n_layers, lookback_periods, batch_size, current_time - start_time])

                        torch.save(m.bicm_model.state_dict(), 'out/bicm_gru_model_d{0}_n{1}_f{2}_l{3}_p{4}_b{5}.pth'.format(d_model, n_heads, dim_feedforward, n_layers, lookback_periods, batch_size))

# Define column names
column_names = ['d_model', 'n_heads', 'dim_feedforward', 'n_layers', 'lookback_periods', 'batch_size', 'time']
# Convert the list of lists to a DataFrame
df = pd.DataFrame(par_sets, columns=column_names)
df.to_csv('out/sta_time.csv', index=False)


