from GRU import GRUModel
import torch
from model_evaluate import evaluate

# Instantiating the models
transfer_model = GRUModel(input_dim_list, hidden_dim, output_dim_list, n_layers)

# Load the full state_dict
state_dict = torch.load('gru_model.pth')

# Filter out the fully connected layer's weights
filtered_state_dict = {k: v for k, v in state_dict.items() if 'fc' not in k}

# Load the filtered state_dict into the transfer_model
transfer_model.load_state_dict(filtered_state_dict, strict=False)