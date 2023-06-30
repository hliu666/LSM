# evaluate(model, dL, hidden_dim_list, n_layers_list, batch_size)
from plot import plot_bicm
plot_bicm(m.bicm_model, dL.bicm_test_loader, dL.bicm_label_scaler)
