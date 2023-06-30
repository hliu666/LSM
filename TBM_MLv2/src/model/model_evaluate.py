import torch
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt


def evaluate(model, dL, hidden_dim_list, n_layers_list, batch_size):
    [carp_hidden_dim, rtmo_hidden_dim, enba_hidden_dim, rtmt_hidden_dim, bicm_hidden_dim,
     rtms_hidden_dim] = hidden_dim_list
    [carp_n_layers, rtmo_n_layers, enba_n_layers, rtmt_n_layers, bicm_n_layers, rtms_n_layers] = n_layers_list

    fig, axs = plt.subplots(2, 2)

    model.eval()
    flag_daily = 0
    flag_hourly = 0

    for (
    carp_X, carp_y, rtmo_X, rtmo_y, enba_X, enba_y, rtmt_X, rtmt_y, bicm_X, bicm_y, rtms_X, rtms_y) in dL.test_loader:
        carp_h = model.init_hidden(batch_size, carp_hidden_dim, carp_n_layers)
        rtmo_h = model.init_hidden(batch_size, rtmo_hidden_dim, rtmo_n_layers)
        enba_h = model.init_hidden(batch_size, enba_hidden_dim, enba_n_layers)
        rtmt_h = model.init_hidden(batch_size, rtmt_hidden_dim, rtmt_n_layers)
        bicm_h = model.init_hidden(batch_size, bicm_hidden_dim, bicm_n_layers)
        rtms_h = model.init_hidden(batch_size, rtms_hidden_dim, rtms_n_layers)

        x_list = [carp_X.float(), rtmo_X.float(), enba_X.float(), rtmt_X.float(), bicm_X.float(), rtms_X.float()]
        h_list = [carp_h, rtmo_h, enba_h, rtmt_h, bicm_h, rtms_h]

        out_list, _ = model(x_list, h_list)
        [carp_out, rtmo_out, enba_out, rtmt_out, bicm_out, rtms_out] = out_list

        if flag_daily == 0:
            carp_pred = carp_out.detach().numpy().reshape(-1, dL.carp_output_dim)
            carp_test = carp_y.detach().numpy().reshape(-1, dL.carp_output_dim)
            flag_daily = 1
        else:
            carp_pred = np.vstack((carp_pred, carp_out.detach().numpy().reshape(-1, dL.carp_output_dim)))
            carp_test = np.vstack((carp_test, carp_y.detach().numpy().reshape(-1, dL.carp_output_dim)))

        if flag_hourly == 0:
            rtmo_pred = rtmo_out.detach().numpy().reshape(-1, dL.rtmo_output_dim)
            rtmo_test = rtmo_y.detach().numpy().reshape(-1, dL.rtmo_output_dim)
            enba_pred = enba_out.detach().numpy().reshape(-1, dL.enba_output_dim)
            enba_test = enba_y.detach().numpy().reshape(-1, dL.enba_output_dim)
            rtmt_pred = rtmt_out.detach().numpy().reshape(-1, dL.rtmt_output_dim)
            rtmt_test = rtmt_y.detach().numpy().reshape(-1, dL.rtmt_output_dim)
            bicm_pred = bicm_out.detach().numpy().reshape(-1, dL.bicm_output_dim)
            bicm_test = bicm_y.detach().numpy().reshape(-1, dL.bicm_output_dim)
            rtms_pred = rtms_out.detach().numpy().reshape(-1, dL.rtms_output_dim)
            rtms_test = rtms_y.detach().numpy().reshape(-1, dL.rtms_output_dim)

            flag_hourly = 1
        else:
            rtmo_pred = np.vstack((rtmo_pred, rtmo_out.detach().numpy().reshape(-1, dL.rtmo_output_dim)))
            rtmo_test = np.vstack((rtmo_test, rtmo_y.detach().numpy().reshape(-1, dL.rtmo_output_dim)))
            enba_pred = np.vstack((enba_pred, enba_out.detach().numpy().reshape(-1, dL.enba_output_dim)))
            enba_test = np.vstack((enba_test, enba_y.detach().numpy().reshape(-1, dL.enba_output_dim)))
            rtmt_pred = np.vstack((rtmt_pred, rtmt_out.detach().numpy().reshape(-1, dL.rtmt_output_dim)))
            rtmt_test = np.vstack((rtmt_test, rtmt_y.detach().numpy().reshape(-1, dL.rtmt_output_dim)))
            bicm_pred = np.vstack((bicm_pred, bicm_out.detach().numpy().reshape(-1, dL.bicm_output_dim)))
            bicm_test = np.vstack((bicm_test, bicm_y.detach().numpy().reshape(-1, dL.bicm_output_dim)))
            rtms_pred = np.vstack((rtms_pred, rtms_out.detach().numpy().reshape(-1, dL.rtms_output_dim)))
            rtms_test = np.vstack((rtms_test, rtms_y.detach().numpy().reshape(-1, dL.rtms_output_dim)))

    carp_pred_scaler = dL.carp_label_scaler.inverse_transform(carp_pred)
    carp_test_scaler = dL.carp_label_scaler.inverse_transform(carp_test)
    rtmo_pred_scaler = dL.rtmo_label_scaler.inverse_transform(rtmo_pred)
    rtmo_test_scaler = dL.rtmo_label_scaler.inverse_transform(rtmo_test)
    enba_pred_scaler = dL.enba_label_scaler.inverse_transform(enba_pred)
    enba_test_scaler = dL.enba_label_scaler.inverse_transform(enba_test)
    rtmt_pred_scaler = dL.rtmt_label_scaler.inverse_transform(rtmt_pred)
    rtmt_test_scaler = dL.rtmt_label_scaler.inverse_transform(rtmt_test)
    bicm_pred_scaler = dL.bicm_label_scaler.inverse_transform(bicm_pred)
    bicm_test_scaler = dL.bicm_label_scaler.inverse_transform(bicm_test)
    rtms_pred_scaler = dL.rtms_label_scaler.inverse_transform(rtms_pred)
    rtms_test_scaler = dL.rtms_label_scaler.inverse_transform(rtms_test)

    r1, _ = pearsonr(carp_pred_scaler[:, 0], carp_test_scaler[:, 0])
    r2, _ = pearsonr(rtmo_pred_scaler[:, 0], rtmo_test_scaler[:, 0])
    r3, _ = pearsonr(rtmt_pred_scaler[:, 0], rtmt_test_scaler[:, 0])
    r4, _ = pearsonr(bicm_pred_scaler[:, 0], bicm_test_scaler[:, 0])
    r5, _ = pearsonr(rtms_pred_scaler[:, 0], rtms_test_scaler[:, 0])

    axs[0, 0].plot(carp_pred_scaler[:, 0], "-o", color="g", label="Predicted")
    axs[0, 0].plot(carp_test_scaler[:, 0], color="b", label="Actual")
    axs[0, 0].set_title('LAI prediction: {0}'.format(round(r1 ** 2, 3)))
    axs[0, 0].legend()

    axs[1, 0].plot(rtmt_pred_scaler[:, 0], "-o", color="g", label="Predicted")
    axs[1, 0].plot(rtmt_test_scaler[:, 0], color="b", label="Actual")
    axs[1, 0].set_title('LST prediction: {0}'.format(round(r3 ** 2, 3)))
    axs[1, 0].legend()

    axs[0, 1].plot(bicm_pred_scaler[:, 0], "-o", color="g", label="Predicted")
    axs[0, 1].plot(bicm_test_scaler[:, 0], color="b", label="Actual")
    axs[0, 1].set_title('GPP prediction: {0}'.format(round(r4 ** 2, 3)))
    axs[0, 1].legend()

    axs[1, 1].plot(rtms_pred_scaler[:, 0], "-o", color="g", label="Predicted")
    axs[1, 1].plot(rtms_test_scaler[:, 0], color="b", label="Actual")
    axs[1, 1].set_title('SIF prediction: {0}'.format(round(r5 ** 2, 3)))
    axs[1, 1].legend()

    plt.show()
