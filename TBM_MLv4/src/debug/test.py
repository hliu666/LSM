from vars import Var
from pars import Par
from data import Data
from dataloader import Dataloader
from Transformer import TSTransformerEncoder

import torch
from tqdm import tqdm
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

feat_dim = 12  # input features
num_classes = 4  # output features

d_model = 4  # the embed dimension (required)
n_heads = 2  # number of heads
dim_feedforward = 8  # the dimension of the feedforward network model (default=2048).

max_len = 72  # the max. length of the incoming sequence
num_layers = 2
dropout = 0.1
n_epochs = 10

transformer = TSTransformerEncoder(feat_dim, max_len, d_model, n_heads, num_layers, dim_feedforward, num_classes,
                                                  dropout=0.1, pos_encoding='fixed', activation='gelu',
                                                  norm='BatchNorm', freeze=False)
optimizer = torch.optim.RMSprop(transformer.parameters(), lr=0.001)

data_root = "../../data/par_set1/"
p = Par()
v = Var()

i = 63
dC = Data(i, p, data_root)
dL = Dataloader(dC, v, p)

train_dataset = dL.bicm_train_loader
test_dataset = dL.bicm_test_loader

start_time = time.time()
for e in tqdm(range(n_epochs)):
    # one epoch on train set
    transformer.train()
    running_loss = 0.0
    for i, batch in enumerate(train_dataset):
        X, y, padding_masks = batch
        predictions = transformer(X.float(), padding_masks)
        # predictions = transformer(X.float())

        # Cascade noise masks (batch_size, padded_length, feat_dim) and padding masks (batch_size, padded_length)
        # target_masks = target_masks * padding_masks.unsqueeze(-1)
        # loss_fn = MaskedMSELoss(reduction="mean")
        # loss = loss_fn(predictions, targets, target_masks)  # (num_active,) individual loss (square error per element) for each active value in batch
        loss = torch.nn.MSELoss()(predictions, y.float())
        loss.backward()
        running_loss += loss.item()
        optimizer.step()
    print(running_loss)
current_time = time.time()
print(current_time-start_time)

from plot import plot_bicm
plot_bicm(transformer, test_dataset)

