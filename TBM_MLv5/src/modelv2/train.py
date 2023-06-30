import torch
from torch import nn
import time
from tqdm import tqdm

def train(model, train_loader, valid_loader, p, temporal_scale):
    # Defining loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=p.learn_rate)

    model.train()
    epoch_times, tra_loss, val_loss = [], [], []

    for epoch in tqdm(range(p.EPOCHS)):
        start_time = time.time()  # Start training loop

        if temporal_scale == "daily":
            h = model.init_hidden(p.batch_size_daily)
        elif temporal_scale == "hourly":
            h = model.init_hidden(p.batch_size_hourly)

        model = model.train()  # Turn on the train mode
        train_loss = 0  # total loss of epoch

        for x, label in train_loader:
            h = h.data
            out, h = model(x.float(), h)

            loss = criterion(out, label.float())

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            train_loss += loss.item()

        model = model.eval()
        valid_loss = 0  # total loss of epoch

        for x, label in valid_loader:
            h = h.data
            out, h = model(x.float(), h)

            loss = criterion(out, label.float())

            valid_loss += loss.item()

        current_time = time.time()
        epoch_times.append(current_time - start_time)

        tra_loss.append(train_loss / len(train_loader))
        val_loss.append(valid_loss / len(valid_loader))

    print("Total Training Time: {} seconds".format(str(sum(epoch_times))))

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(tra_loss, label="Training loss")
    ax.plot(val_loss, label="Validation loss")
    ax.legend()
    plt.show()

    return model
