import torch.nn as nn
import torch
from sklearn.preprocessing import MinMaxScaler


# Define the GRU model
class GRUModel(nn.Module):
    def __init__(self, input_dim_list, hidden_dim_list, output_dim_list, n_layers_list):
        super(GRUModel, self).__init__()

        # Use GRUCell instead of GRU
        [carp_input_dim, rtmo_input_dim, enba_input_dim, bicm_input_dim, rtms_input_dim] = input_dim_list
        [carp_hidden_dim, rtmo_hidden_dim, enba_hidden_dim, bicm_hidden_dim, rtms_hidden_dim] = hidden_dim_list
        [carp_n_layers, rtmo_n_layers, enba_n_layers, bicm_n_layers, rtms_n_layers] = n_layers_list

        self.carp_hidden_dim = carp_hidden_dim
        self.rtmo_hidden_dim = rtmo_hidden_dim
        self.enba_hidden_dim = enba_hidden_dim
        self.bicm_hidden_dim = bicm_hidden_dim
        self.rtms_hidden_dim = rtms_hidden_dim

        self.carp_n_layers = carp_n_layers
        self.rtmo_n_layers = rtmo_n_layers
        self.enba_n_layers = enba_n_layers
        self.bicm_n_layers = bicm_n_layers
        self.rtms_n_layers = rtms_n_layers

        # Create a list of GRU cells for each layer
        self.carp_gru_cells = nn.ModuleList()
        for i in range(carp_n_layers):
            if i == 0:
                input_dim = carp_input_dim
            else:
                input_dim = carp_hidden_dim
            self.carp_gru_cells.append(nn.GRUCell(input_dim, carp_hidden_dim))

        self.rtmo_gru_cells = nn.ModuleList()
        for i in range(rtmo_n_layers):
            if i == 0:
                input_dim = rtmo_input_dim
            else:
                input_dim = rtmo_hidden_dim
            self.rtmo_gru_cells.append(nn.GRUCell(input_dim, rtmo_hidden_dim))

        self.enba_gru_cells = nn.ModuleList()
        for i in range(enba_n_layers):
            if i == 0:
                input_dim = enba_input_dim
            else:
                input_dim = enba_hidden_dim
            self.enba_gru_cells.append(nn.GRUCell(input_dim, enba_hidden_dim))

        self.bicm_gru_cells = nn.ModuleList()
        for i in range(bicm_n_layers):
            if i == 0:
                input_dim = bicm_input_dim
            else:
                input_dim = bicm_hidden_dim
            self.bicm_gru_cells.append(nn.GRUCell(input_dim, bicm_hidden_dim))

        self.rtms_gru_cells = nn.ModuleList()
        for i in range(rtms_n_layers):
            if i == 0:
                input_dim = rtms_input_dim
            else:
                input_dim = rtms_hidden_dim
            self.rtms_gru_cells.append(nn.GRUCell(input_dim, rtms_hidden_dim))

        [carp_output_dim, rtmo_output_dim, enba_output_dim, bicm_output_dim, rtms_output_dim] = output_dim_list

        self.carp_fc = nn.Linear(carp_hidden_dim, carp_output_dim)
        self.rtmo_fc = nn.Linear(rtmo_hidden_dim, rtmo_output_dim)
        self.enba_fc = nn.Linear(enba_hidden_dim, enba_output_dim)
        self.bicm_fc = nn.Linear(bicm_hidden_dim, bicm_output_dim)
        self.rtms_fc = nn.Linear(rtms_hidden_dim, rtms_output_dim)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x_list, h_list):
        [carp_x, rtmo_x, enba_x, bicm_x, rtms_x] = x_list
        [carp_h, rtmo_h, enba_h, bicm_h, rtms_h] = h_list
        carp_out, rtmo_out, enba_out, bicm_out, rtms_out = [], [], [], [], []

        for i in range(carp_x.size(1)):  # loop over the sequence
            if i > 0:
                carp_x[:, i, 0] = bicm_daily[:, 0]
            carp_h[0] = self.carp_gru_cells[0](carp_x[:, i, :].clone(), carp_h[0])
            for k in range(1, self.carp_n_layers):
                carp_h[k] = self.carp_gru_cells[k](carp_h[k - 1], carp_h[k])

            rtmo_hourly_hidden, enba_hourly_hidden, rtmt_hourly_hidden, bicm_hourly_hidden, rtms_hourly_hidden = [], [], [], [], []
            lai_daily = self.sigmoid(self.carp_fc(carp_h[-1]))[:, 0]
            for j in range(0, 24):
                # RTM optical
                rtmo_x[:, i, j, 0] = lai_daily
                rtmo_h[0] = self.rtmo_gru_cells[0](rtmo_x[:, i, j, :].clone(), rtmo_h[0])
                for k in range(1, self.rtmo_n_layers):
                    rtmo_h[k] = self.rtmo_gru_cells[k](rtmo_h[k - 1], rtmo_h[k])
                fPAR_hourly = self.sigmoid(self.rtmo_fc(rtmo_h[-1]))[:, 0]
                Rnet_o_hourly = self.sigmoid(self.rtmo_fc(rtmo_h[-1]))[:, 1]
                rtmo_hourly_hidden.append(rtmo_h[-1])

                # Energy balance
                enba_x[:, i, j, 0] = lai_daily
                enba_x[:, i, j, 3] = Rnet_o_hourly
                enba_h[0] = self.enba_gru_cells[0](enba_x[:, i, j, :].clone(), enba_h[0])
                for k in range(1, self.enba_n_layers):
                    enba_h[k] = self.enba_gru_cells[k](enba_h[k - 1], enba_h[k])
                lst_hourly = self.sigmoid(self.enba_fc(enba_h[-1]))[:, 0]
                enba_hourly_hidden.append(enba_h[-1])

                # Biochemistry
                bicm_x[:, i, j, 0] = lai_daily
                bicm_x[:, i, j, 1] = fPAR_hourly
                bicm_x[:, i, j, 3] = lst_hourly
                bicm_h[0] = self.bicm_gru_cells[0](bicm_x[:, i, j, :].clone(), bicm_h[0])
                for k in range(1, self.bicm_n_layers):
                    bicm_h[k] = self.bicm_gru_cells[k](bicm_h[k - 1], bicm_h[k])
                fqe_u_hourly = self.sigmoid(self.bicm_fc(bicm_h[-1]))[:, 2]
                fqe_h_hourly = self.sigmoid(self.bicm_fc(bicm_h[-1]))[:, 3]
                bicm_hourly_hidden.append(bicm_h[-1])

                # RTM sif
                rtms_x[:, i, j, 0] = lai_daily
                rtms_x[:, i, j, 1] = fPAR_hourly
                rtms_x[:, i, j, 3] = fqe_u_hourly
                rtms_x[:, i, j, 4] = fqe_h_hourly

                rtms_h[0] = self.rtms_gru_cells[0](rtms_x[:, i, j, :].clone(), rtms_h[0])
                for k in range(1, self.rtms_n_layers):
                    rtms_h[k] = self.rtms_gru_cells[k](rtms_h[k - 1], rtms_h[k])
                rtms_hourly_hidden.append(rtms_h[-1])

            bicm_hourly_stack = torch.stack(bicm_hourly_hidden, dim=0)
            bicm_daily = torch.mean(self.sigmoid(self.bicm_fc(bicm_hourly_stack)), dim=0)
            # Solution2: gpp_daily = gpp_hourly_hidden[-1]

            carp_out.append(carp_h[-1].unsqueeze(0))
            rtmo_out.append(torch.stack(rtmo_hourly_hidden, dim=1))
            enba_out.append(torch.stack(enba_hourly_hidden, dim=1))
            bicm_out.append(torch.stack(bicm_hourly_hidden, dim=1))
            rtms_out.append(torch.stack(rtms_hourly_hidden, dim=1))

        carp_out = torch.cat(carp_out, dim=0)
        carp_out = self.sigmoid(self.carp_fc(carp_out[-1]))
        rtmo_out = torch.stack(rtmo_out, dim=0)
        rtmo_out = self.sigmoid(self.rtmo_fc(rtmo_out[-1]))
        enba_out = torch.stack(enba_out, dim=0)
        enba_out = self.sigmoid(self.enba_fc(enba_out[-1]))
        bicm_out = torch.stack(bicm_out, dim=0)
        bicm_out = self.sigmoid(self.bicm_fc(bicm_out[-1]))
        rtms_out = torch.stack(rtms_out, dim=0)
        rtms_out = self.sigmoid(self.rtms_fc(rtms_out[-1]))

        out_list = [carp_out, rtmo_out, enba_out, bicm_out, rtms_out]
        h_list = [carp_h, rtmo_h, enba_h, bicm_h, rtms_h]

        return out_list, h_list

    def init_hidden(self, batch_size, hidden_dim, n_layers):
        hidden = [torch.zeros(batch_size, hidden_dim) for _ in range(n_layers)]
        return hidden
