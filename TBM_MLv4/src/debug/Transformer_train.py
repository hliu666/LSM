import torch
from torch import nn
from Transformer_1 import TransformerModel
import time
from sklearn.preprocessing import MinMaxScaler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import pandas as pd
import joblib
import numpy as np

i = 63
hour_length = 26304
data_root = "../../data/par_set1/"

# driving data
df = pd.read_csv(data_root+"HARV.csv")
df = df[['year', 'hour', 'doy', 'TA', 'VPD', 'PAR_up', 'SW', 'wds']]
df.rename(columns={'PAR_up': 'PAR'}, inplace=True)

output_daily = joblib.load(data_root+f"{i}_model_output_daily.pkl")
output_hourly = joblib.load(data_root+f"{i}_model_output_hourly.pkl")
output_hourly = output_hourly.reshape(-1, 21)

LAI = np.repeat(output_daily[:, -1], 24)[0:hour_length]
df['LAI'] = LAI
df['GPP'] = output_hourly[0:hour_length, 1]
df['LST'] = output_hourly[0:hour_length, 5]

scaler = MinMaxScaler()
label_scaler = MinMaxScaler()
df_fit = df.copy()
df_fit[['LAI', 'GPP', 'LST', 'doy', 'hour']] = scaler.fit_transform(df[['LAI', 'GPP', 'LST', 'doy', 'hour']])
label_scaler.fit(df['LAI'].values.reshape(-1, 1))

batch_size = 32 # 822 = 26302/32
nfeature = 4
nlayers = 4  # number of ``nn.TransformerEncoderLayer`` in ``nn.TransformerEncoder``
nhead = 1  # number of heads in ``nn.MultiheadAttention``
dropout = 0.2  # dropout probability
lr = 0.005  # learning rate
epochs = 3
dim_feedforward = 128

def Transformer_train():
    start_time = time.time()
    model = TransformerModel(nfeature, nhead, nlayers, dim_feedforward).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer


    model.train()  # turn on train mode
    epoch_times = []
    tra_loss, val_loss = [], []

    src = torch.from_numpy(np.array(df_fit[['GPP', 'LST', 'doy', 'hour']]))  # Input tensor
    tgt = torch.from_numpy(np.array(df_fit[['LAI']]))  # Output tensor

    # Transform tensors to batches
    src_batches = torch.split(src, batch_size)
    tgt_batches = torch.split(tgt, batch_size)

    for epoch in range(1, epochs + 1):
        total_loss = 0.
        counter = 0.
        avg_loss = 0.

        for i, (data, target) in enumerate(zip(src_batches, tgt_batches)):
            # data and target are the same shape with (input_window,batch_len,1)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output.float(), target.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.7)
            optimizer.step()

            total_loss += loss.item()
            avg_loss += loss.item()

            counter += 1
            if counter % 200 == 0:
                print("Epoch {}......Step: {}....... Average Loss for Epoch: {}".format(epoch, counter,
                                                                                           avg_loss / counter))

        print(avg_loss)

    current_time = time.time()
    epoch_times.append(current_time - start_time)

    tra_loss.append(avg_loss / counter)

    print("Total Training Time: {} seconds".format(str(sum(epoch_times))))

    return model

model = Transformer_train()

import matplotlib.pyplot as plt
# predict the next n steps based on the input data
def predict_future(eval_model):
    eval_model.eval()

    tgt = torch.from_numpy(np.array(df[['LAI']]))  # Output tensor
    input_window = 500 # int(len(df)*0.3)
    src = torch.from_numpy(np.array(df_fit[['GPP', 'LST', 'doy', 'hour']])[-input_window:])  # Input tensor

    output = eval_model(src)
    sim = label_scaler.inverse_transform(output.detach().numpy())
    obs = tgt.detach().numpy()[-input_window:, 0]

    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    plt.plot(sim, color="red")
    plt.plot(obs, color="blue")
    plt.show()

predict_future(model)