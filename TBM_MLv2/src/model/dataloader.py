import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader

class Dataloader:
    def __init__(self, dC, x_vars, x_pars, y_vars, lookback_periods, batch_size, test_portion):

        [x_carp_vars, x_rtmo_vars, x_enba_vars, x_bicm_vars, x_rtms_vars] = x_vars
        [x_carp_pars, x_rtmo_pars, x_enba_pars, x_bicm_pars, x_rtms_pars] = x_pars
        [y_carp_vars, y_rtmo_vars, y_enba_vars, y_bicm_vars, y_rtms_vars] = y_vars

        carp_X_train, carp_y_train, carp_X_test, carp_y_test, self.carp_label_scaler = self.dataLoader_daily(
            dC.daily_df,
            lookback_periods,
            test_portion,
            y_carp_vars,
            x_carp_vars,
            x_carp_pars)

        rtmo_X_train, rtmo_y_train, rtmo_X_test, rtmo_y_test, self.rtmo_label_scaler = self.dataLoader_hourly(
            dC.hourly_df,
            lookback_periods,
            test_portion,
            y_rtmo_vars,
            x_rtmo_vars,
            x_rtmo_pars)

        enba_X_train, enba_y_train, enba_X_test, enba_y_test, self.enba_label_scaler = self.dataLoader_hourly(
            dC.hourly_df,
            lookback_periods,
            test_portion,
            y_enba_vars,
            x_enba_pars,
            x_enba_vars)

        bicm_X_train, bicm_y_train, bicm_X_test, bicm_y_test, self.bicm_label_scaler = self.dataLoader_hourly(
            dC.hourly_df,
            lookback_periods,
            test_portion,
            y_bicm_vars,
            x_bicm_vars,
            x_bicm_pars)

        rtms_X_train, rtms_y_train, rtms_X_test, rtms_y_test, self.rtms_label_scaler = self.dataLoader_hourly(
            dC.hourly_df,
            lookback_periods,
            test_portion,
            y_rtms_vars,
            x_rtms_pars,
            x_rtms_vars)

        train_data = TensorDataset(torch.from_numpy(carp_X_train), torch.from_numpy(carp_y_train),
                                   torch.from_numpy(rtmo_X_train), torch.from_numpy(rtmo_y_train),
                                   torch.from_numpy(enba_X_train), torch.from_numpy(enba_y_train),
                                   torch.from_numpy(bicm_X_train), torch.from_numpy(bicm_y_train),
                                   torch.from_numpy(rtms_X_train), torch.from_numpy(rtms_y_train))

        test_data = TensorDataset(torch.from_numpy(carp_X_test), torch.from_numpy(carp_y_test),
                                  torch.from_numpy(rtmo_X_test), torch.from_numpy(rtmo_y_test),
                                  torch.from_numpy(enba_X_test), torch.from_numpy(enba_y_test),
                                  torch.from_numpy(bicm_X_test), torch.from_numpy(bicm_y_test),
                                  torch.from_numpy(rtms_X_test), torch.from_numpy(rtms_y_test))

        self.label_scalers = [self.carp_label_scaler, self.rtmo_label_scaler, self.enba_label_scaler, self.bicm_label_scaler, self.rtms_label_scaler]
        self.train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)
        self.test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size, drop_last=True)

        self.carp_input_dim = len(x_carp_vars+x_carp_pars)
        self.rtmo_input_dim = len(x_rtmo_vars+x_rtmo_pars)
        self.enba_input_dim = len(x_enba_pars+x_enba_vars)
        self.bicm_input_dim = len(x_bicm_vars+x_bicm_pars)
        self.rtms_input_dim = len(x_rtms_pars+x_rtms_vars)

        self.carp_output_dim = len(y_carp_vars)
        self.rtmo_output_dim = len(y_rtmo_vars)
        self.enba_output_dim = len(y_enba_vars)
        self.bicm_output_dim = len(y_bicm_vars)
        self.rtms_output_dim = len(y_rtms_vars)

        self.input_dim_list = [self.carp_input_dim, self.rtmo_input_dim, self.enba_input_dim, self.bicm_input_dim, self.rtms_input_dim]
        self.output_dim_list = [self.carp_output_dim, self.rtmo_output_dim, self.enba_output_dim, self.bicm_output_dim, self.rtms_output_dim]

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

        X_train = inputs  # [:-test_length]
        y_train = labels  # [:-test_length]

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

        num_hours = hourly_data.shape[0]
        num_columns = hourly_data.shape[1]
        num_days = num_hours // 24
        daily_hourly_data = hourly_data.reshape((num_days, 24, num_columns))

        # Define lookback period and split inputs/labels
        inputs = np.zeros((len(daily_hourly_data) - lookback_daily, lookback_daily, 24, len(x_field_vars + x_field_pars)))
        labels = np.zeros((len(daily_hourly_data) - lookback_daily, 24, len(y_field)))

        for i in range(lookback_daily, len(daily_hourly_data)):
            inputs[i - lookback_daily] = daily_hourly_data[i - lookback_daily:i, :, len(y_field):]
            labels[i - lookback_daily] = daily_hourly_data[i, :, 0:len(y_field)]

        # Split data into train/test portions and combining all data from different files into a single array
        test_length = int(test_portion * len(inputs))

        X_train = inputs  # [:-test_length]
        y_train = labels  # [:-test_length]

        X_test = inputs[-test_length:]
        y_test = labels[-test_length:]

        return X_train, y_train, X_test, y_test, label_scaler
