import torch
from torch import nn
import time
from tqdm import tqdm
from Transformer import TSTransformerEncoder
class Model:
    def __init__(self, dL, p):
        self.carp_model_init = TSTransformerEncoder(dL.carp_input_dim, p.lookback_daily, p.carp_d_model,
                                               p.carp_n_heads, p.carp_n_layers,
                                               p.carp_dim_feedforward, dL.carp_output_dim,
                                               dropout=p.dropout, pos_encoding='fixed', activation='gelu',
                                               norm='BatchNorm', freeze=False)

        self.rtmo_model_init = TSTransformerEncoder(dL.rtmo_input_dim, p.lookback_hourly, p.rtmo_d_model,
                                               p.rtmo_n_heads, p.rtmo_n_layers,
                                               p.rtmo_dim_feedforward, dL.rtmo_output_dim,
                                               dropout=p.dropout, pos_encoding='fixed', activation='gelu',
                                               norm='BatchNorm', freeze=False)

        self.enba_model_init = TSTransformerEncoder(dL.enba_input_dim, p.lookback_hourly, p.enba_d_model,
                                               p.enba_n_heads, p.enba_n_layers,
                                               p.enba_dim_feedforward, dL.enba_output_dim,
                                               dropout=p.dropout, pos_encoding='fixed', activation='gelu',
                                               norm='BatchNorm', freeze=False)

        self.bicm_model_init = TSTransformerEncoder(dL.bicm_input_dim, p.lookback_hourly, p.bicm_d_model,
                                               p.bicm_n_heads, p.bicm_n_layers,
                                               p.rtmo_dim_feedforward, dL.bicm_output_dim,
                                               dropout=p.dropout, pos_encoding='fixed', activation='gelu',
                                               norm='BatchNorm', freeze=False)

        self.rtms_model_init = TSTransformerEncoder(dL.rtms_input_dim, p.lookback_hourly, p.rtms_d_model,
                                               p.rtms_n_heads, p.rtms_n_layers,
                                               p.rtms_dim_feedforward, dL.rtms_output_dim,
                                               dropout=p.dropout, pos_encoding='fixed', activation='gelu',
                                               norm='BatchNorm', freeze=False)

        #self.carp_model = self.train(self.carp_model_init, dL.carp_train_loader, p)
        #self.rtmo_model = self.train(self.rtmo_model_init, dL.rtmo_train_loader, p)
        #self.enba_model = self.train(self.enba_model_init, dL.enba_train_loader, p)
        self.bicm_model = self.train(self.bicm_model_init, dL.bicm_train_loader, p, dL)
        #self.rtms_model = self.train(self.rtms_model_init, dL.rtms_train_loader, p)

    def train(self, model, train_loader, p, dL):
        # Defining loss function and optimizer
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.RMSprop(model.parameters(), lr=p.learn_rate)
        epoch_times, tra_loss, val_loss = [], [], []
        start_time = time.time() # Start training loop
        for epoch in tqdm(range(p.EPOCHS)):

            model = model.train()  # Turn on the train mode

            epoch_loss = 0  # total loss of epoch

            for _, batch in enumerate(train_loader):

                X, targets, padding_masks = batch
                predictions = model(X, padding_masks)

                # (batch_size,) loss for each sample in the batch
                loss = criterion(predictions, targets.float())

                # Zero gradients, perform a backward pass, and update the weights.
                optimizer.zero_grad()
                loss.backward()

                # torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=1.0)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0)
                optimizer.step()

                epoch_loss += loss.item()

            current_time = time.time()
            # print("Epoch {}/{} Done, Total Loss: {}".format(epoch, p.EPOCHS, epoch_loss / len(train_loader)))
            epoch_times.append(current_time - start_time)

        # print("Total Training Time: {} seconds".format(str(sum(epoch_times))))

        return model

    def load(self):
        self.carp_model.load_state_dict(torch.load('out/carp_gru_model.pth'))
        self.rtmo_model.load_state_dict(torch.load('out/rtmo_gru_model.pth'))
        self.enba_model.load_state_dict(torch.load('out/enba_gru_model.pth'))
        self.bicm_model.load_state_dict(torch.load('out/bicm_gru_model.pth'))
        self.rtms_model.load_state_dict(torch.load('out/rtms_gru_model.pth'))

        self.carp_model.eval()
        self.rtmo_model.eval()
        self.enba_model.eval()
        self.bicm_model.eval()
        self.rtms_model.eval()
