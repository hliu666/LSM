import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader

class Dataloader:
    def __init__(self, dC, v, p):

        carp_X_train, carp_y_train, carp_X_test, carp_y_test, self.carp_label_scaler = self.dataLoader_daily(
            dC.daily_df,
            p.lookback_periods,
            p.test_portion,
            v.y_carp_vars,
            v.x_carp_vars,
            v.x_carp_pars)

        rtmo_X_train, rtmo_y_train, rtmo_X_test, rtmo_y_test, self.rtmo_label_scaler = self.dataLoader_hourly(
            dC.hourly_df,
            p.lookback_periods,
            p.test_portion,
            v.y_rtmo_vars,
            v.x_rtmo_vars,
            v.x_rtmo_pars)

        enba_X_train, enba_y_train, enba_X_test, enba_y_test, self.enba_label_scaler = self.dataLoader_hourly(
            dC.hourly_df,
            p.lookback_periods,
            p.test_portion,
            v.y_enba_vars,
            v.x_enba_pars,
            v.x_enba_vars)

        bicm_X_train, bicm_y_train, bicm_X_test, bicm_y_test, self.bicm_label_scaler = self.dataLoader_hourly(
            dC.hourly_df,
            p.lookback_periods,
            p.test_portion,
            v.y_bicm_vars,
            v.x_bicm_vars,
            v.x_bicm_pars)

        rtms_X_train, rtms_y_train, rtms_X_test, rtms_y_test, self.rtms_label_scaler = self.dataLoader_hourly(
            dC.hourly_df,
            p.lookback_periods,
            p.test_portion,
            v.y_rtms_vars,
            v.x_rtms_pars,
            v.x_rtms_vars)

        carp_train_data = TensorDataset(torch.from_numpy(carp_X_train), torch.from_numpy(carp_y_train))
        rtmo_train_data = TensorDataset(torch.from_numpy(rtmo_X_train), torch.from_numpy(rtmo_y_train))
        enba_train_data = TensorDataset(torch.from_numpy(enba_X_train), torch.from_numpy(enba_y_train))
        bicm_train_data = TensorDataset(torch.from_numpy(bicm_X_train), torch.from_numpy(bicm_y_train))
        rtms_train_data = TensorDataset(torch.from_numpy(rtms_X_train), torch.from_numpy(rtms_y_train))

        self.carp_test_data = [torch.from_numpy(carp_X_test), torch.from_numpy(carp_y_test)]
        self.rtmo_test_data = [torch.from_numpy(rtmo_X_test), torch.from_numpy(rtmo_y_test)]
        self.enba_test_data = [torch.from_numpy(enba_X_test), torch.from_numpy(enba_y_test)]
        self.bicm_test_data = [torch.from_numpy(bicm_X_test), torch.from_numpy(bicm_y_test)]
        self.rtms_test_data = [torch.from_numpy(rtms_X_test), torch.from_numpy(rtms_y_test)]

        self.carp_train_loader = DataLoader(carp_train_data, shuffle=True, batch_size=p.batch_size_daily, drop_last=True)
        self.rtmo_train_loader = DataLoader(rtmo_train_data, shuffle=True, batch_size=p.batch_size_hourly, drop_last=True)
        self.enba_train_loader = DataLoader(enba_train_data, shuffle=True, batch_size=p.batch_size_hourly, drop_last=True)
        self.bicm_train_loader = DataLoader(bicm_train_data, shuffle=True, batch_size=p.batch_size_hourly, drop_last=True)
        self.rtms_train_loader = DataLoader(rtms_train_data, shuffle=True, batch_size=p.batch_size_hourly, drop_last=True)

        self.carp_input_dim = len(v.x_carp_vars+v.x_carp_pars)
        self.rtmo_input_dim = len(v.x_rtmo_vars+v.x_rtmo_pars)
        self.enba_input_dim = len(v.x_enba_vars+v.x_enba_pars)
        self.bicm_input_dim = len(v.x_bicm_vars+v.x_bicm_pars)
        self.rtms_input_dim = len(v.x_rtms_vars+v.x_rtms_pars)

        self.carp_output_dim = len(v.y_carp_vars)
        self.rtmo_output_dim = len(v.y_rtmo_vars)
        self.enba_output_dim = len(v.y_enba_vars)
        self.bicm_output_dim = len(v.y_bicm_vars)
        self.rtms_output_dim = len(v.y_rtms_vars)

    def dataLoader_daily(self, df, lookback_daily, test_portion, y_field, x_field_vars, x_field_pars):
        df = df[y_field + x_field_vars + x_field_pars]

        # Scaling the input data
        scaler = MinMaxScaler()
        label_scaler = MinMaxScaler()

        # Fit and transform the selected columns and replace them in the dataframe
        df_fit = df.copy()
        df_fit[y_field + x_field_vars] = scaler.fit_transform(df[y_field + x_field_vars])
        daily_data = df_fit.values
        label_scaler.fit(df[y_field].values.reshape(-1, len(y_field)))

        # Define lookback period and split inputs/labels
        inputs = np.zeros((len(daily_data) - lookback_daily, lookback_daily, len(x_field_vars + x_field_pars)))
        labels = np.zeros((len(daily_data) - lookback_daily, len(y_field)))

        for i in range(lookback_daily, len(daily_data)):
            inputs[i - lookback_daily] = daily_data[i - lookback_daily:i, len(y_field):]
            labels[i - lookback_daily] = daily_data[i, 0:len(y_field)]

        # Split data into train/test portions and combining all data from different files into a single array
        test_length = int(test_portion * len(inputs))

        X_train = inputs[:-test_length]
        y_train = labels[:-test_length]

        X_test = inputs[-test_length:]
        y_test = labels[-test_length:]

        return X_train, y_train, X_test, y_test, label_scaler

    def dataLoader_hourly(self, df, lookback_daily, test_portion, y_field, x_field_vars, x_field_pars):
        df = df[y_field + x_field_vars + x_field_pars]

        # Scaling the input data
        scaler = MinMaxScaler()
        label_scaler = MinMaxScaler()

        # Fit and transform the selected columns and replace them in the dataframe
        df_fit = df.copy()
        df_fit[y_field + x_field_vars] = scaler.fit_transform(df[y_field + x_field_vars])
        hourly_data = df_fit.values
        label_scaler.fit(df[y_field].values.reshape(-1, len(y_field)))

        # Define lookback period and split inputs/labels
        lookback = lookback_daily *24

        inputs = np.zeros((len(hourly_data) - lookback, lookback, len(x_field_vars + x_field_pars)))
        labels = np.zeros((len(hourly_data) - lookback, len(y_field)))

        for i in range(lookback, len(hourly_data)):
            inputs[i - lookback] = hourly_data[i - lookback:i, len(y_field):]
            labels[i - lookback] = hourly_data[i, 0:len(y_field)]

        # Split data into train/test portions and combining all data from different files into a single array
        test_length = int(test_portion * len(inputs))

        X_train = inputs[:-test_length]
        y_train = labels[:-test_length]

        X_test = inputs[-test_length:]
        y_test = labels[-test_length:]

        return X_train, y_train, X_test, y_test, label_scaler
