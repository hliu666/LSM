from vars import Var
from pars import Par
from data import Data
from dataloader import Dataloader
from model import Model
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

# evaluate(model, dL, hidden_dim_list, n_layers_list, batch_size)
from plot import plot_bicm
plot_bicm(m.bicm_model, dL.bicm_test_loader, dL.bicm_label_scaler)

"""
torch.save(m.carp_model.state_dict(), 'out/carp_gru_model.pth')
torch.save(m.rtmo_model.state_dict(), 'out/rtmo_gru_model.pth')
torch.save(m.enba_model.state_dict(), 'out/enba_gru_model.pth')
torch.save(m.bicm_model.state_dict(), 'out/bicm_gru_model.pth')
torch.save(m.rtms_model.state_dict(), 'out/rtms_gru_model.pth')
"""



