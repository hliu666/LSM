from vars import Var
from pars import Par
from GRU import Model
from data import Data
from dataloader import Dataloader
from train import train
import torch
import time
import pandas as pd

"""
Load data root and select interested fields
"""
data_root = "../../data/par_set1/"

hidden_dim_list = [8, 16, 32, 64]
n_layers_list = [1, 2, 4, 8]
lookback_periods_list = [1, 3, 5, 7]
batch_size_list = [16, 32, 64, 128]

par_sets = []
for hidden_dim in hidden_dim_list:
    for n_layer in n_layers_list:
        for lookback_periods in lookback_periods_list:
            for batch_size in batch_size_list:
                p = Par(hidden_dim, n_layer, lookback_periods, batch_size)
                v = Var()

                """
                Normalize the data and train Machine learning model
                """
                i = 63
                dC = Data(i, p, data_root)
                dL = Dataloader(dC, v, p)
                m = Model(dL, p)

                # m.carp_model = train(m.carp_model, dL.carp_train_loader, p, 'daily')
                # m.rtmo_model = train(m.rtmo_model, dL.rtmo_train_loader, p, 'hourly')
                # m.enba_model = train(m.enba_model, dL.enba_train_loader, p, 'hourly')
                start_time = time.time()
                m.bicm_model = train(m.bicm_model, dL.bicm_train_loader, p, 'hourly')
                current_time = time.time()
                # m.rtms_model = train(m.rtms_model, dL.rtms_train_loader, p, 'hourly')

                par_sets.append([hidden_dim, n_layer, lookback_periods, batch_size, current_time - start_time])
                print("Total Time Elapsed: {} seconds".format(str(current_time - start_time)))

                # torch.save(m.carp_model.state_dict(), 'out/carp_gru_model.pth')
                # torch.save(m.rtmo_model.state_dict(), 'out/rtmo_gru_model.pth')
                # torch.save(m.enba_model.state_dict(), 'out/enba_gru_model.pth')
                torch.save(m.bicm_model.state_dict(), 'out/bicm_gru_model_h{0}_n{1}_l{2}_b{3}.pth'.format(hidden_dim, n_layer, lookback_periods, batch_size))
                #torch.save(m.rtms_model.state_dict(), 'out/rtms_gru_model.pth')

# Define column names
column_names = ['hidden_dim', 'n_layer', 'lookback_periods', 'batch_size', 'time']
# Convert the list of lists to a DataFrame
df = pd.DataFrame(par_sets, columns=column_names)
df.to_csv('out/sta_time.csv', index=False)
