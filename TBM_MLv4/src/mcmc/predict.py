import numpy as np
from scipy.stats import pearsonr

def predict(model, data_loader, hidden_dim_list, n_layers_list, output_dim_list, batch_size, bicm_label_scaler):
    [carp_hidden_dim, rtmo_hidden_dim, enba_hidden_dim, bicm_hidden_dim, rtms_hidden_dim] = hidden_dim_list
    [carp_n_layers, rtmo_n_layers, enba_n_layers, bicm_n_layers, rtms_n_layers] = n_layers_list
    [carp_output_dim, rtmo_output_dim, enba_output_dim, bicm_output_dim, rtms_output_dim] = output_dim_list

    model.eval()
    flag_daily = 0
    flag_hourly = 0

    for (carp_X, rtmo_X, enba_X, bicm_X, rtms_X) in data_loader:
        carp_h = model.init_hidden(batch_size, carp_hidden_dim, carp_n_layers)
        rtmo_h = model.init_hidden(batch_size, rtmo_hidden_dim, rtmo_n_layers)
        enba_h = model.init_hidden(batch_size, enba_hidden_dim, enba_n_layers)
        bicm_h = model.init_hidden(batch_size, bicm_hidden_dim, bicm_n_layers)
        rtms_h = model.init_hidden(batch_size, rtms_hidden_dim, rtms_n_layers)

        x_list = [carp_X.float(), rtmo_X.float(), enba_X.float(), bicm_X.float(), rtms_X.float()]
        h_list = [carp_h, rtmo_h, enba_h, bicm_h, rtms_h]

        out_list, _ = model(x_list, h_list)
        [carp_out, rtmo_out, enba_out, bicm_out, rtms_out] = out_list

        #if flag_daily == 0:
        #    carp_pred = carp_out.detach().numpy().reshape(-1, carp_output_dim)
        #    flag_daily = 1
        #else:
        #    carp_pred = np.vstack((carp_pred, carp_out.detach().numpy().reshape(-1, dL.carp_output_dim)))

        if flag_hourly == 0:
            #rtmo_pred = rtmo_out.detach().numpy().reshape(-1, rtmo_output_dim)
            #enba_pred = enba_out.detach().numpy().reshape(-1, enba_output_dim)
            #rtmt_pred = rtmt_out.detach().numpy().reshape(-1, rtmt_output_dim)
            bicm_prd = bicm_out.detach().numpy().reshape(-1, bicm_output_dim)
            #rtms_pred = rtms_out.detach().numpy().reshape(-1, rtms_output_dim)

            flag_hourly = 1

        else:
            #rtmo_pred = np.vstack((rtmo_pred, rtmo_out.detach().numpy().reshape(-1, rtmo_output_dim)))
            #enba_pred = np.vstack((enba_pred, enba_out.detach().numpy().reshape(-1, enba_output_dim)))
            #rtmt_pred = np.vstack((rtmt_pred, rtmt_out.detach().numpy().reshape(-1, rtmt_output_dim)))
            bicm_prd = np.vstack((bicm_prd, bicm_out.detach().numpy().reshape(-1, bicm_output_dim)))
            #rtms_pred = np.vstack((rtms_pred, rtms_out.detach().numpy().reshape(-1, rtms_output_dim)))

    NEE = bicm_prd[:, 1]
    NEE = (NEE - np.min(NEE)) / (np.max(NEE) - np.min(NEE))
    bicm_prd_scaler = bicm_label_scaler.inverse_transform(NEE.reshape(-1, 1))

    return bicm_prd_scaler


