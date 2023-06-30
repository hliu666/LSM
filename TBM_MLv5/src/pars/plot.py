import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import pearsonr
import numpy as np

def cal_r2_bicm(model, test_data, label_scaler):
    X, y = test_data

    h = model.init_hidden(X.shape[0])
    out, h = model(X.float(), h)

    y_pred_scaler = label_scaler.inverse_transform(out.detach().numpy())
    y_test_scaler = label_scaler.inverse_transform(y.numpy())

    r1, _ = pearsonr(y_test_scaler[:, 0], y_pred_scaler[:, 0])
    r2, _ = pearsonr(y_test_scaler[:, 1], y_pred_scaler[:, 1])

    return r1, r2

def plot_carp(model, test_data, label_scaler):
    X, y = test_data

    h = model.init_hidden(X.shape[0])
    out, h = model(X.float(), h)

    y_pred_scaler = label_scaler.inverse_transform(out.detach().numpy())
    y_test_scaler = label_scaler.inverse_transform(y.numpy())

    r1, _ = pearsonr(y_test_scaler[:, 0], y_pred_scaler[:, 0])

    fig = plt.figure(figsize=(12, 4))
    gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1])

    axs00 = plt.subplot(gs[0])
    axs00.plot(y_pred_scaler[:, 0], "-o", color="g", label="Predicted")
    axs00.plot(y_test_scaler[:, 0], color="b", label="Actual")
    axs00.set_title('LAI prediction: {0}'.format(round(r1 ** 2, 3)))
    axs00.legend()

    axs01 = plt.subplot(gs[1])
    k1, b1 = np.polyfit(y_pred_scaler[:, 0], y_test_scaler[:, 0], 1)
    axs01.scatter(y_pred_scaler[:, 0], y_test_scaler[:, 0], color="b")
    axs01.plot(y_pred_scaler[:, 0], k1 * y_pred_scaler[:, 0] + b1, color="g", label=f'y={k1:.2f}x+{b1:.2f}')
    axs01.plot(y_pred_scaler[:, 0], y_pred_scaler[:, 0], color="r", linestyle='dashed')
    axs01.set_xlabel('Predicted LAI')
    axs01.set_ylabel('Modeled LAI')
    axs01.legend()

    # after plotting the data, find the maximum and minimum values across all data
    min_01 = np.min([y_pred_scaler[:, 0].min(), y_test_scaler[:, 0].min()])
    max_01 = np.max([y_pred_scaler[:, 0].max(), y_test_scaler[:, 0].max()])

    # set the same x and y limits for axs10 and axs11
    axs01.set_xlim(min_01, max_01)
    axs01.set_ylim(min_01, max_01)

    plt.tight_layout()
    plt.show()

def plot_enba(model, test_data, label_scaler):
    X, y = test_data

    h = model.init_hidden(X.shape[0])
    out, h = model(X.float(), h)

    y_pred_scaler = label_scaler.inverse_transform(out.detach().numpy())
    y_test_scaler = label_scaler.inverse_transform(y.numpy())

    r1, _ = pearsonr(y_test_scaler[:, 0], y_pred_scaler[:, 0])

    fig = plt.figure(figsize=(12, 4))
    gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1])

    axs00 = plt.subplot(gs[0])
    axs00.plot(y_pred_scaler[:, 0], "-o", color="g", label="Predicted")
    axs00.plot(y_test_scaler[:, 0], color="b", label="Actual")
    axs00.set_title('LST prediction: {0}'.format(round(r1 ** 2, 3)))
    axs00.legend()

    axs01 = plt.subplot(gs[1])
    k1, b1 = np.polyfit(y_pred_scaler[:, 0], y_test_scaler[:, 0], 1)
    axs01.scatter(y_pred_scaler[:, 0], y_test_scaler[:, 0], color="b")
    axs01.plot(y_pred_scaler[:, 0], k1 * y_pred_scaler[:, 0] + b1, color="g", label=f'y={k1:.2f}x+{b1:.2f}')
    axs01.plot(y_pred_scaler[:, 0], y_pred_scaler[:, 0], color="r", linestyle='dashed')
    axs01.set_xlabel('Predicted LST')
    axs01.set_ylabel('Modeled LST')
    axs01.legend()

    # after plotting the data, find the maximum and minimum values across all data
    min_01 = np.min([y_pred_scaler[:, 0].min(), y_test_scaler[:, 0].min()])
    max_01 = np.max([y_pred_scaler[:, 0].max(), y_test_scaler[:, 0].max()])

    # set the same x and y limits for axs10 and axs11
    axs01.set_xlim(min_01, max_01)
    axs01.set_ylim(min_01, max_01)

    plt.tight_layout()
    plt.show()

def plot_rtms(model, test_data, label_scaler):
    X, y = test_data

    h = model.init_hidden(X.shape[0])
    out, h = model(X.float(), h)

    y_pred_scaler = label_scaler.inverse_transform(out.detach().numpy())
    y_test_scaler = label_scaler.inverse_transform(y.numpy())

    r1, _ = pearsonr(y_test_scaler[:, 0], y_pred_scaler[:, 0])
    r2, _ = pearsonr(y_test_scaler[:, 1], y_pred_scaler[:, 1])

    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 2, width_ratios=[2, 1])

    axs00 = plt.subplot(gs[0])
    axs00.plot(y_pred_scaler[:, 0], "-o", color="g", label="Predicted")
    axs00.plot(y_test_scaler[:, 0], color="b", label="Actual")
    axs00.set_title('SIFu prediction: {0}'.format(round(r1 ** 2, 3)))
    axs00.legend()

    axs01 = plt.subplot(gs[1])
    k1, b1 = np.polyfit(y_pred_scaler[:, 0], y_test_scaler[:, 0], 1)
    axs01.scatter(y_pred_scaler[:, 0], y_test_scaler[:, 0], color="b")
    axs01.plot(y_pred_scaler[:, 0], k1 * y_pred_scaler[:, 0] + b1, color="g", label=f'y={k1:.2f}x+{b1:.2f}')
    axs01.plot(y_pred_scaler[:, 0], y_pred_scaler[:, 0], color="r", linestyle='dashed')
    axs01.set_xlabel('Predicted SIFu')
    axs01.set_ylabel('Modeled SIFu')
    axs01.legend()

    axs10 = plt.subplot(gs[2])
    axs10.plot(y_pred_scaler[:, 1], "-o", color="g", label="Predicted")
    axs10.plot(y_test_scaler[:, 1], color="b", label="Actual")
    axs10.set_title('SIFh prediction: {0}'.format(round(r2 ** 2, 3)))
    axs10.legend()

    axs11 = plt.subplot(gs[3])
    k2, b2 = np.polyfit(y_pred_scaler[:, 1], y_test_scaler[:, 1], 1)
    axs11.scatter(y_pred_scaler[:, 1], y_test_scaler[:, 1], color="b")
    axs11.plot(y_pred_scaler[:, 1], k2 * y_pred_scaler[:, 1] + b2, color="g", label=f'y={k2:.2f}x+{b2:.2f}')
    axs11.plot(y_pred_scaler[:, 1], y_pred_scaler[:, 1], color="r", linestyle='dashed')
    axs11.set_xlabel('Predicted SIFh')
    axs11.set_ylabel('Modeled SIFh')
    axs11.legend()

    # after plotting the data, find the maximum and minimum values across all data
    min_01 = np.min([y_pred_scaler[:, 0].min(), y_test_scaler[:, 0].min()])
    max_01 = np.max([y_pred_scaler[:, 0].max(), y_test_scaler[:, 0].max()])

    min_11 = np.min([y_pred_scaler[:, 1].min(), y_test_scaler[:, 1].min()])
    max_11 = np.max([y_pred_scaler[:, 1].max(), y_test_scaler[:, 1].max()])

    # set the same x and y limits for axs10 and axs11
    axs01.set_xlim(min_01, max_01)
    axs01.set_ylim(min_01, max_01)

    axs11.set_xlim(min_11, max_11)
    axs11.set_ylim(min_11, max_11)

    plt.tight_layout()
    plt.show()

def plot_rtmo(model, test_data, label_scaler):
    X, y = test_data

    h = model.init_hidden(X.shape[0])
    out, h = model(X.float(), h)

    y_pred_scaler = label_scaler.inverse_transform(out.detach().numpy())
    y_test_scaler = label_scaler.inverse_transform(y.numpy())

    r1, _ = pearsonr(y_test_scaler[:, 0], y_pred_scaler[:, 0])
    r2, _ = pearsonr(y_test_scaler[:, 1], y_pred_scaler[:, 1])

    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 2, width_ratios=[2, 1])

    axs00 = plt.subplot(gs[0])
    axs00.plot(y_pred_scaler[:, 0], "-o", color="g", label="Predicted")
    axs00.plot(y_test_scaler[:, 0], color="b", label="Actual")
    axs00.set_title('fPAR prediction: {0}'.format(round(r1 ** 2, 3)))
    axs00.legend()

    axs01 = plt.subplot(gs[1])
    k1, b1 = np.polyfit(y_pred_scaler[:, 0], y_test_scaler[:, 0], 1)
    axs01.scatter(y_pred_scaler[:, 0], y_test_scaler[:, 0], color="b")
    axs01.plot(y_pred_scaler[:, 0], k1 * y_pred_scaler[:, 0] + b1, color="g", label=f'y={k1:.2f}x+{b1:.2f}')
    axs01.plot(y_pred_scaler[:, 0], y_pred_scaler[:, 0], color="r", linestyle='dashed')
    axs01.set_xlabel('Predicted fPAR')
    axs01.set_ylabel('Modeled fPAR')
    axs01.legend()

    axs10 = plt.subplot(gs[2])
    axs10.plot(y_pred_scaler[:, 1], "-o", color="g", label="Predicted")
    axs10.plot(y_test_scaler[:, 1], color="b", label="Actual")
    axs10.set_title('Rnet_o prediction: {0}'.format(round(r2 ** 2, 3)))
    axs10.legend()

    axs11 = plt.subplot(gs[3])
    k2, b2 = np.polyfit(y_pred_scaler[:, 1], y_test_scaler[:, 1], 1)
    axs11.scatter(y_pred_scaler[:, 1], y_test_scaler[:, 1], color="b")
    axs11.plot(y_pred_scaler[:, 1], k2 * y_pred_scaler[:, 1] + b2, color="g", label=f'y={k2:.2f}x+{b2:.2f}')
    axs11.plot(y_pred_scaler[:, 1], y_pred_scaler[:, 1], color="r", linestyle='dashed')
    axs11.set_xlabel('Predicted Rnet_o')
    axs11.set_ylabel('Modeled Rnet_o')
    axs11.legend()

    # after plotting the data, find the maximum and minimum values across all data
    min_01 = np.min([y_pred_scaler[:, 0].min(), y_test_scaler[:, 0].min()])
    max_01 = np.max([y_pred_scaler[:, 0].max(), y_test_scaler[:, 0].max()])

    min_11 = np.min([y_pred_scaler[:, 1].min(), y_test_scaler[:, 1].min()])
    max_11 = np.max([y_pred_scaler[:, 1].max(), y_test_scaler[:, 1].max()])

    # set the same x and y limits for axs10 and axs11
    axs01.set_xlim(min_01, max_01)
    axs01.set_ylim(min_01, max_01)

    axs11.set_xlim(min_11, max_11)
    axs11.set_ylim(min_11, max_11)

    plt.tight_layout()
    plt.show()

    # averaged spectrum curve
    avg_test_spectral = y_test_scaler[:, 2:].mean(axis=0)
    avg_pred_spectral = y_pred_scaler[:, 2:].mean(axis=0)
    r3, _ = pearsonr(avg_test_spectral, avg_pred_spectral)

    fig = plt.figure(figsize=(12, 4))
    gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1])

    axs00 = plt.subplot(gs[0])
    axs00.plot(avg_pred_spectral, "-o", color="g", label="Predicted")
    axs00.plot(avg_test_spectral, color="b", label="Actual")
    axs00.set_title('averaged spectral prediction: {0}'.format(round(r1 ** 2, 3)))
    axs00.legend()

    axs01 = plt.subplot(gs[1])
    k1, b1 = np.polyfit(avg_pred_spectral, avg_test_spectral, 1)
    axs01.scatter(avg_pred_spectral, avg_test_spectral, color="b")
    axs01.plot(avg_pred_spectral, k1 * avg_pred_spectral + b1, color="g", label=f'y={k1:.2f}x+{b1:.2f}')
    axs01.plot(avg_pred_spectral, avg_pred_spectral, color="r", linestyle='dashed')
    axs01.set_xlabel('Predicted averaged spectral')
    axs01.set_ylabel('Modeled averaged spectral')
    axs01.legend()

    # after plotting the data, find the maximum and minimum values across all data
    min_01 = np.min([avg_pred_spectral.min(), avg_test_spectral.min()])
    max_01 = np.max([avg_pred_spectral.max(), avg_test_spectral.max()])

    # set the same x and y limits for axs10 and axs11
    axs01.set_xlim(min_01, max_01)
    axs01.set_ylim(min_01, max_01)

    plt.tight_layout()
    plt.show()
