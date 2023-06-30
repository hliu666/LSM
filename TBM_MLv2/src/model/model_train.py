import torch
from torch import nn
from GRU import GRUModel
import time
import matplotlib.pyplot as plt

def train(dL, batch_size, n_layers_list, learn_rate, hidden_dim_list, EPOCHS):
    [carp_hidden_dim, rtmo_hidden_dim, enba_hidden_dim, bicm_hidden_dim, rtms_hidden_dim] = hidden_dim_list
    [carp_n_layers, rtmo_n_layers, enba_n_layers, bicm_n_layers, rtms_n_layers] = n_layers_list

    # Instantiating the models
    model = GRUModel(dL.input_dim_list, hidden_dim_list, dL.output_dim_list, n_layers_list)

    # Defining loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

    model.train()
    epoch_times = []
    tra_loss, val_loss = [], []
    # Start training loop
    for epoch in range(1, EPOCHS + 1):
        start_time = time.time()
        avg_loss = 0.
        avg_loss1 = 0.
        counter = 0
        counter1 = 0

        for (carp_X, carp_y, rtmo_X, rtmo_y, enba_X, enba_y, bicm_X, bicm_y, rtms_X, rtms_y) in dL.train_loader:
            carp_h = model.init_hidden(batch_size, carp_hidden_dim, carp_n_layers)
            rtmo_h = model.init_hidden(batch_size, rtmo_hidden_dim, rtmo_n_layers)
            enba_h = model.init_hidden(batch_size, enba_hidden_dim, enba_n_layers)
            bicm_h = model.init_hidden(batch_size, bicm_hidden_dim, bicm_n_layers)
            rtms_h = model.init_hidden(batch_size, rtms_hidden_dim, rtms_n_layers)

            optimizer.zero_grad()
            counter = counter + 1

            x_list = [carp_X.float(), rtmo_X.float(), enba_X.float(), bicm_X.float(), rtms_X.float()]
            h_list = [carp_h, rtmo_h, enba_h, bicm_h, rtms_h]

            out_list, _ = model(x_list, h_list)
            [carp_out, rtmo_out, enba_out, bicm_out, rtms_out] = out_list

            carp_loss = criterion(carp_out, carp_y.float())
            rtmo_loss = criterion(rtmo_out, rtmo_y.float())
            enba_loss = criterion(enba_out, enba_y.float())
            bicm_loss = criterion(bicm_out, bicm_y.float())
            rtms_loss = criterion(rtms_out, rtms_y.float())

            loss = carp_loss + rtmo_loss + enba_loss + bicm_loss + rtms_loss
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()

            if counter % 200 == 0:
                print("Epoch {}......Step: {}/{}....... Average Loss for Epoch: {}".format(epoch, counter,
                                                                                           len(dL.train_loader),
                                                                                           avg_loss / counter))

        for (carp_X, carp_y, rtmo_X, rtmo_y, enba_X, enba_y, bicm_X, bicm_y, rtms_X, rtms_y) in dL.test_loader:
            carp_h = model.init_hidden(batch_size, carp_hidden_dim, carp_n_layers)
            rtmo_h = model.init_hidden(batch_size, rtmo_hidden_dim, rtmo_n_layers)
            enba_h = model.init_hidden(batch_size, enba_hidden_dim, enba_n_layers)
            bicm_h = model.init_hidden(batch_size, bicm_hidden_dim, bicm_n_layers)
            rtms_h = model.init_hidden(batch_size, rtms_hidden_dim, rtms_n_layers)

            optimizer.zero_grad()
            counter1 = counter1 + 1

            x_list = [carp_X.float(), rtmo_X.float(), enba_X.float(), bicm_X.float(), rtms_X.float()]
            h_list = [carp_h, rtmo_h, enba_h, bicm_h, rtms_h]

            out_list, _ = model(x_list, h_list)
            [carp_out, rtmo_out, enba_out, bicm_out, rtms_out] = out_list

            carp_loss = criterion(carp_out, carp_y.float())
            rtmo_loss = criterion(rtmo_out, rtmo_y.float())
            enba_loss = criterion(enba_out, enba_y.float())
            bicm_loss = criterion(bicm_out, bicm_y.float())
            rtms_loss = criterion(rtms_out, rtms_y.float())

            loss = carp_loss + rtmo_loss + enba_loss + bicm_loss + rtms_loss

            avg_loss1 += loss.item()

        current_time = time.time()
        print("Epoch {}/{} Done, Total Loss: {}".format(epoch, EPOCHS, avg_loss / len(dL.train_loader)))
        print("Total Time Elapsed: {} seconds".format(str(current_time - start_time)))
        epoch_times.append(current_time - start_time)

        tra_loss.append(avg_loss / counter)
        val_loss.append(avg_loss1 / counter1)

    print("Total Training Time: {} seconds".format(str(sum(epoch_times))))

    torch.save(model.state_dict(), 'gru_model.pth')

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(tra_loss, label="Training loss")
    ax.plot(val_loss, label="Validation loss")
    ax.legend()
    plt.show()

    return model

