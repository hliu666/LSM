import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import pearsonr
import numpy as np

def plot_bicm(transformer, test_data, label_scaler):
    for i, batch in enumerate(test_data):
        X, y, padding_masks = batch
        out = transformer(X.float(), padding_masks)

        y_pred_sub = out.detach().numpy()
        y_test_sub = y.numpy()
        if i == 0:
            y_test = y_pred_sub
            y_pred = y_test_sub
        else:
            y_test = np.vstack((y_test, y_pred_sub))
            y_pred = np.vstack((y_pred, y_test_sub))

    y_test_scaler = label_scaler.inverse_transform(y_test)
    y_pred_scaler = label_scaler.inverse_transform(y_pred)

    r1, _ = pearsonr(y_test_scaler[:, 0], y_pred_scaler[:, 0])
    r2, _ = pearsonr(y_test_scaler[:, 1], y_pred_scaler[:, 1])

    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 2, width_ratios=[2, 1])

    axs00 = plt.subplot(gs[0])
    axs00.plot(y_pred_scaler[:, 0], "-o", color="g", label="Predicted")
    axs00.plot(y_test_scaler[:, 0], color="b", label="Actual")
    axs00.set_title('GPP prediction: {0}'.format(round(r1 ** 2, 3)))
    axs00.legend()

    axs01 = plt.subplot(gs[1])
    k1, b1 = np.polyfit(y_pred_scaler[:, 0], y_test_scaler[:, 0], 1)
    axs01.scatter(y_pred_scaler[:, 0], y_test_scaler[:, 0], color="b")
    # You can also plot the regression line on top of this
    axs01.plot(y_pred_scaler[:, 0], k1 * y_pred_scaler[:, 0] + b1, color="g", label=f'y={k1:.2f}x+{b1:.2f}')
    axs01.plot(y_pred_scaler[:, 0], y_pred_scaler[:, 0], color="r", linestyle='dashed')
    axs01.set_xlabel('Predicted GPP')
    axs01.set_ylabel('Modeled GPP')
    axs01.legend()

    axs10 = plt.subplot(gs[2])
    axs10.plot(y_pred_scaler[:, 1], "-o", color="g", label="Predicted")
    axs10.plot(y_test_scaler[:, 1], color="b", label="Actual")
    axs10.set_title('NEE prediction: {0}'.format(round(r2 ** 2, 3)))
    axs10.legend()

    axs11 = plt.subplot(gs[3])
    k2, b2 = np.polyfit(y_pred_scaler[:, 1], y_test_scaler[:, 1], 1)
    axs11.scatter(y_pred_scaler[:, 1], y_test_scaler[:, 1], color="b")
    # Plot the regression line on top of this
    axs11.plot(y_pred_scaler[:, 1], k2 * y_pred_scaler[:, 1] + b2, color="g", label=f'y={k2:.2f}x+{b2:.2f}')
    axs11.plot(y_pred_scaler[:, 1], y_pred_scaler[:, 1], color="r", linestyle='dashed')
    axs11.set_xlabel('Predicted NEE')
    axs11.set_ylabel('Modeled NEE')
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
