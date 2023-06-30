import torch.nn as nn
import torch

# Define the GRU model
class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, drop_prob=0.0):
        super(GRUModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, h):
        out, h = self.gru(x, h)
        out = self.sigmoid(self.fc(out[:, -1, :]))
        return out, h

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        hidden = weight.new_zeros(self.n_layers, batch_size, self.hidden_dim)
        return hidden

class Model:
    def __init__(self, dL, p):
        self.carp_model = GRUModel(dL.carp_input_dim, p.carp_hidden_dim, dL.carp_output_dim, p.carp_n_layers)
        self.rtmo_model = GRUModel(dL.rtmo_input_dim, p.rtmo_hidden_dim, dL.rtmo_output_dim, p.rtmo_n_layers)
        self.enba_model = GRUModel(dL.enba_input_dim, p.enba_hidden_dim, dL.enba_output_dim, p.enba_n_layers)
        self.bicm_model = GRUModel(dL.bicm_input_dim, p.bicm_hidden_dim, dL.bicm_output_dim, p.bicm_n_layers)
        self.rtms_model = GRUModel(dL.rtms_input_dim, p.rtms_hidden_dim, dL.rtms_output_dim, p.rtms_n_layers)

    def load(self, hidden_dim, n_layer, lookback_periods, batch_size):
        # self.carp_model.load_state_dict(torch.load('out/carp_gru_model.pth'))
        # self.rtmo_model.load_state_dict(torch.load('out/rtmo_gru_model.pth'))
        # self.enba_model.load_state_dict(torch.load('out/enba_gru_model.pth'))
        # self.bicm_model.load_state_dict(torch.load('out/bicm_gru_model.pth'))
        # self.rtms_model.load_state_dict(torch.load('out/rtms_gru_model.pth'))

        self.bicm_model.load_state_dict(torch.load('out/bicm_gru_model_h{0}_n{1}_l{2}_b{3}.pth'.format(hidden_dim, n_layer, lookback_periods, batch_size)))

        # self.carp_model.eval()
        # self.rtmo_model.eval()
        # self.enba_model.eval()
        self.bicm_model.eval()
        # self.rtms_model.eval()