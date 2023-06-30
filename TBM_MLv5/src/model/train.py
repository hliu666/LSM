import torch
from torch import nn
import time

def train(model, train_loader, p, temporal_scale):
    # Defining loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=p.learn_rate)

    model.train()
    epoch_times, tra_loss, val_loss = [], [], []
    # Start training loop
    for epoch in range(1, p.EPOCHS + 1):
        start_time = time.time()
        if temporal_scale == "daily":
            h = model.init_hidden(p.batch_size_daily)
        elif temporal_scale == "hourly":
            h = model.init_hidden(p.batch_size_hourly)

        avg_loss = 0.
        counter = 0

        for x, label in train_loader:
            h = h.data
            optimizer.zero_grad()

            counter += 1

            out, h = model(x.float(), h)
            loss = criterion(out, label.float())
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()

            #if counter % 200 == 0:
            #    print("Epoch {}......Step: {}/{}....... Average Loss for Epoch: {}".format(epoch, counter,
            #                                                                               len(train_loader),
            #                                                                               avg_loss / counter))
        current_time = time.time()
        print("Epoch {}/{} Done, Total Loss: {}".format(epoch, p.EPOCHS, avg_loss / len(train_loader)))
        print("Total Time Elapsed: {} seconds".format(str(current_time - start_time)))
        epoch_times.append(current_time - start_time)

        # tra_loss.append(avg_loss / counter)

    print("Total Training Time: {} seconds".format(str(sum(epoch_times))))

    # fig, ax = plt.subplots(figsize=(4, 4))
    # ax.plot(tra_loss, label="Training loss")
    # ax.plot(val_loss, label="Validation loss")
    # ax.legend()
    # plt.show()

    return model
