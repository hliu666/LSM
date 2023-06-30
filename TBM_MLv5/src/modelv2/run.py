from vars import Var
from pars import Par
from GRU import Model
from data import Data
from dataloader import Dataloader
from train import train
import torch

"""
Load data root and select interested fields
"""
data_root = "../../data/par_set1/"
p = Par()
v = Var()

"""
Normalize the data and train Machine learning model
"""
# for i in range(0, 64):
for i in range(63, 64):
    dC = Data(i, p, data_root)
    dL = Dataloader(dC, v, p)
    m = Model(dL, p)

    # m.carp_model = train(m.carp_model, dL.carp_train_loader, dL.carp_valid_loader, p, 'daily')
    # m.rtmo_model = train(m.rtmo_model, dL.rtmo_train_loader, dL.rtmo_valid_loader, p, 'hourly')
    # m.enba_model = train(m.enba_model, dL.enba_train_loader, dL.enba_valid_loader, p, 'hourly')
    m.bicm_model = train(m.bicm_model, dL.bicm_train_loader, dL.bicm_valid_loader, p, 'hourly')
    # m.rtms_model = train(m.rtms_model, dL.rtms_train_loader, dL.rtms_valid_loader, p, 'hourly')
"""
torch.save(m.carp_model.state_dict(), 'out/carp_gru_model.pth')
torch.save(m.rtmo_model.state_dict(), 'out/rtmo_gru_model.pth')
torch.save(m.enba_model.state_dict(), 'out/enba_gru_model.pth')
torch.save(m.bicm_model.state_dict(), 'out/bicm_gru_model.pth')
torch.save(m.rtms_model.state_dict(), 'out/rtms_gru_model.pth')
"""

torch.save(m.rtmo_model.state_dict(), 'out/rtmo_gru_model1.pth')

# evaluate(model, dL, hidden_dim_list, n_layers_list, batch_size)
