import torch
import sys
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader

sys.path.append('../model')
from GRU import GRUModel

def load_model():
    """
    1. load data root and select interested fields
    """
    wavelength_data = joblib.load("../../data/mod_list_wavelength_resample.pkl")

    x_carp_vars = ['GPP', 'LST', 'doy']
    x_carp_pars = ['clspan', 'lma', 'f_auto', 'f_fol', 'd_onset', 'cronset', 'd_fall', 'crfall', 'CI', 'LiDf']
    y_carp_vars = ['LAI']

    x_rtmo_vars = ['LAI', 'SW', 'SZA', 'VZA', 'SAA']
    x_rtmo_pars = ['Cab', 'Car', 'Cm', 'Cbrown', 'Cw', 'Ant', 'CI', 'LiDf'] + [f'leaf_b{i}' for i in wavelength_data]
    y_rtmo_vars = ['fPAR', 'Rnet_o'] + [f'canopy_b{i}' for i in wavelength_data]

    x_enba_vars = ['LAI', 'SW', 'TA', 'Rnet_o', 'wds']
    x_enba_pars = ['CI', 'LiDf', 'rho', 'tau', 'rs']
    y_enba_vars = ['LST']

    x_bicm_vars = ['LAI', 'fPAR', 'PAR', 'LST', 'VPD']
    x_bicm_pars = ['RUB', 'CB6F', 'Rdsc', 'gm', 'e', 'BallBerrySlope', 'BallBerry0']
    y_bicm_vars = ['GPP', 'NEE', 'fqe_u', 'fqe_h']

    x_rtms_vars = ['LAI', 'fPAR', 'PAR', 'fqe_u', 'fqe_h', 'SZA', 'VZA', 'SAA']
    x_rtms_pars = ['CI', 'LiDf', 'eta']
    y_rtms_vars = ['SIFu', 'SIFh']

    x_pars = [x_carp_pars, x_rtmo_pars, x_enba_pars, x_bicm_pars, x_rtms_pars]
    x_vars = [x_carp_vars, x_rtmo_vars, x_enba_vars, x_bicm_vars, x_rtms_vars]

    carp_input_dim = len(x_carp_vars + x_carp_pars)
    rtmo_input_dim = len(x_rtmo_vars + x_rtmo_pars)
    enba_input_dim = len(x_enba_vars + x_enba_pars)
    bicm_input_dim = len(x_bicm_vars + x_bicm_pars)
    rtms_input_dim = len(x_rtms_vars + x_rtms_pars)

    carp_output_dim = len(y_carp_vars)
    rtmo_output_dim = len(y_rtmo_vars)
    enba_output_dim = len(y_enba_vars)
    bicm_output_dim = len(y_bicm_vars)
    rtms_output_dim = len(y_rtms_vars)

    input_dim_list = [carp_input_dim, rtmo_input_dim, enba_input_dim, bicm_input_dim, rtms_input_dim]
    output_dim_list = [carp_output_dim, rtmo_output_dim, enba_output_dim, bicm_output_dim, rtms_output_dim]

    """
    2. define machine learning parameters
    """
    carp_hidden_dim, rtmo_hidden_dim, enba_hidden_dim, bicm_hidden_dim, rtms_hidden_dim = 32, 32, 32, 32, 32
    carp_n_layers, rtmo_n_layers, enba_n_layers, bicm_n_layers, rtms_n_layers = 4, 4, 4, 4, 4

    hidden_dim_list = [carp_hidden_dim, rtmo_hidden_dim, enba_hidden_dim, bicm_hidden_dim, rtms_hidden_dim]
    n_layers_list = [carp_n_layers, rtmo_n_layers, enba_n_layers, bicm_n_layers, rtms_n_layers]

    """
    3. load machine learning parameters
    """
    # Instantiating the models
    model = GRUModel(input_dim_list, hidden_dim_list, output_dim_list, n_layers_list)
    model.load_state_dict(torch.load('../model/gru_model.pth'))
    model.eval()

    return model, x_pars, x_vars, hidden_dim_list, n_layers_list, output_dim_list

def read_vars(data_paths, hour_length):
    input_path, output_hourly_path = data_paths

    # driving data
    df_input = pd.read_csv(input_path)
    df_input = df_input[df_input['year'] < 2021]

    df = df_input[['year', 'doy', 'TA', 'VPD', 'PAR_up', 'SW', 'wds']]
    df.rename(columns={'PAR_up': 'PAR'}, inplace=True)

    # Group the DataFrame by the 'Category' column and calculate the mean of the 'Value' column
    df_daily = df.groupby(["year", "doy"]).mean().reset_index().rename(columns={"Category": "doy"})
    df_daily["doy"] = np.sin(2 * np.pi * (df_daily["doy"] - 1) / 365)
    df_daily["LST"] = 0.0
    df_daily["LAI"] = 0.0
    df_daily["GPP"] = 0.0
    df_daily = df_daily[["doy", "GPP", "LST", "LAI"]]

    output_hourly = joblib.load(output_hourly_path)
    output_hourly = output_hourly.reshape(-1, 21)
    # LAI
    df['LAI'] = 0.0

    df['NEE'] = 0.0
    df['GPP'] = 0.0
    df['fPAR'] = 0.0
    df['APAR'] = df['fPAR'] * df['PAR']

    df['SIFu'] = 0.0
    df['SIFh'] = 0.0

    df['LST'] = 0.0

    df['fqe_u'] = 0.0
    df['fqe_h'] = 0.0

    df['Rnet_u_o'] = 0.0
    df['Rnet_u_t'] = 0.0
    df['Rnet_h_o'] = 0.0
    df['Rnet_h_t'] = 0.0

    df['Rnet_o'] = df['Rnet_u_o'] + df['Rnet_h_o']
    df['Rnet_t'] = df['Rnet_u_t'] + df['Rnet_h_t']

    df['Tcu'] = 0.0
    df['Tch'] = 0.0
    df['Tsu'] = 0.0
    df['Tsh'] = 0.0

    df['SZA'] = output_hourly[0:hour_length, 18]
    df['VZA'] = output_hourly[0:hour_length, 19]
    df['SAA'] = output_hourly[0:hour_length, 20]

    df = df.fillna(method='ffill').fillna(method='bfill')

    # Scaling the input data
    label_scaler = MinMaxScaler()

    # Fit and transform the selected columns and replace them in the dataframe
    label_scaler.fit_transform(np.array(df_input['NEE_obs']).reshape(-1, 1))

    return df, df_daily, label_scaler, np.array(df_input['NEE_obs'])[0:16896]

def read_pars(hourly_df, daily_df, x_pars, prospectpro_path):
    """
    Parameters for carbon pool
    """
    daily_df['clspan'] = 1.00116  # clspan, leaf lifespan               (1.0001 - 5)
    daily_df['lma'] = 55  # clma, leaf mass per area          (81 - 120) g C m-2
    daily_df['f_auto'] = 0.5  # f_auto, fraction of GPP respired  (0.3 - 0.7)
    daily_df['f_fol'] = 0.15  # f_fol, frac GPP to foliage        (0.01 - 0.5)
    daily_df['d_onset'] = 130.  # d_onset, clab release date       (1 - 365) (60,150)
    daily_df['cronset'] = 20.  # cronset, clab release period      (10 - 100)
    daily_df['d_fall'] = 300.  # d_fall, date of leaf fall        (1 - 365) (242,332)
    daily_df['crfall'] = 35.  # crfall, leaf fall period          (10 - 100)

    daily_df['clspan_max'] = 5.0  # clspan, leaf lifespan               (1.0001 - 5)
    daily_df['lma_max'] = 120
    daily_df['f_auto_max'] = 0.7  # f_auto, fraction of GPP respired  (0.3 - 0.7)
    daily_df['f_fol_max'] = 0.5  # f_fol, frac GPP to foliage        (0.01 - 0.5)
    daily_df['d_onset_max'] = 365  # d_onset, clab release date       (1 - 365) (60,150)
    daily_df['cronset_max'] = 100.  # cronset, clab release period      (10 - 100)
    daily_df['d_fall_max'] = 365.  # d_fall, date of leaf fall        (1 - 365) (242,332)
    daily_df['crfall_max'] = 100.  # crfall, leaf fall period          (10 - 100)

    daily_df['CI'] = 0.72  # clumping index
    daily_df['LiDf'] = 55

    daily_df['CI_max'] = 1.0
    daily_df['LiDf_max'] = 100

    hourly_df['CI'] = 0.72  # clumping index
    hourly_df['LiDf'] = 55

    hourly_df['CI_max'] = 1.0
    hourly_df['LiDf_max'] = 100
    """
    Parameters for biochemistry
    """
    hourly_df['RUB'] = 30  # [umol sites m-2] Rubisco density
    hourly_df['Rdsc'] = 0.01  # [] Scalar for mitochondrial (dark) respiration
    hourly_df['CB6F'] = 35  # [umol sites m-2] Cyt b6f density
    hourly_df['gm'] = 0.01  # [] mesophyll conductance to CO2
    hourly_df['e'] = 0.92

    hourly_df['RUB_max'] = 120  # [umol sites m-2] Rubisco density
    hourly_df['Rdsc_max'] = 0.05  # [] Scalar for mitochondrial (dark) respiration
    hourly_df['CB6F_max'] = 150  # [umol sites m-2] Cyt b6f density
    hourly_df['gm_max'] = 5.0  # [] mesophyll conductance to CO2
    hourly_df['e_max'] = 1.0

    hourly_df['BallBerrySlope'] = 10
    hourly_df['BallBerry0'] = 0.01

    hourly_df['BallBerrySlope_max'] = 100
    hourly_df['BallBerry0_max'] = 1.0

    """
    Parameters for Radiative Transfer Model in optical/thermal band
    """
    hourly_df['lma'] = 55.0

    hourly_df['Cab'] = 28.12
    hourly_df['Car'] = 5.56
    hourly_df['Cm'] = hourly_df['lma'] / 10000.0
    hourly_df['Cbrown'] = 0.185
    hourly_df['Cw'] = 0.00597
    hourly_df['Ant'] = 1.966

    hourly_df['Cab_max'] = 80
    hourly_df['Car_max'] = 20
    hourly_df['Cm_max'] = 120 / 10000.0
    hourly_df['Cbrown_max'] = 1
    hourly_df['Cw_max'] = 1
    hourly_df['Ant_max'] = 10

    hourly_df['rho'] = 0.01
    hourly_df['tau'] = 0.01
    hourly_df['rs'] = 0.06

    hourly_df['rho_max'] = 0.05
    hourly_df['tau_max'] = 0.05
    hourly_df['rs_max'] = 0.1

    prospectpro = np.loadtxt(prospectpro_path)
    Cab, Car, Ant, Cbrown, Cw, Cm = 28.12, 5.56, 55/10000.0, 0.185, 0.00597, 1.966
    reflectance = sip_leaf(prospectpro, Cab, Car, Ant, Cbrown, Cw, Cm)

    wavelength_data = joblib.load("../../data/mod_list_wavelength_resample.pkl")

    hyperspectral_data = np.repeat(reflectance.flatten(), len(hourly_df)).reshape(len(hourly_df), -1)
    hyperspectral_df = pd.DataFrame(hyperspectral_data)
    hyperspectral_df.columns = [f'leaf_b{i}' for i in wavelength_data]

    hourly_df = pd.concat([hourly_df, hyperspectral_df], axis=1)

    """
    Parameters for Radiative Transfer Model in fluorescence
    """
    hourly_df['eta'] = 5E-5
    hourly_df['eta_max'] = 1E-4

    [x_carp_pars, x_rtmo_pars, x_enba_pars, x_bicm_pars, x_rtms_pars] = x_pars
    daily_field = x_carp_pars
    hourly_field = list(set(['Cab', 'Car', 'Cm', 'Cbrown', 'Cw', 'Ant', 'CI', 'LiDf'] + x_enba_pars + x_bicm_pars + x_rtms_pars))

    for field in daily_field:
        daily_df[field] = daily_df[f"{field}_max"]

    for field in hourly_field:
        hourly_df[field] = hourly_df[f"{field}_max"]

    return daily_df, hourly_df

def read_par_set(pars, hourly_df, daily_df, lookback_daily, batch_size, x_vars, x_pars):
    RUB, CB6F, BallBerry0 = pars
    hourly_df['RUB'] = RUB/hourly_df['RUB_max'] # [umol sites m-2] Rubisco density
    hourly_df['CB6F'] = CB6F/hourly_df['CB6F_max']  # [umol sites m-2] Cyt b6f density
    hourly_df['BallBerry0'] = BallBerry0/hourly_df['BallBerry0_max']

    [x_carp_vars, x_rtmo_vars, x_enba_vars, x_bicm_vars, x_rtms_vars] = x_vars
    [x_carp_pars, x_rtmo_pars, x_enba_pars, x_bicm_pars, x_rtms_pars] = x_pars

    carp_X = dataLoader_daily(daily_df, lookback_daily, x_carp_vars, x_carp_pars)
    rtmo_X = dataLoader_hourly(hourly_df, lookback_daily, x_rtmo_vars, x_rtmo_pars)
    enba_X = dataLoader_hourly(hourly_df, lookback_daily, x_enba_vars, x_enba_pars)
    bicm_X = dataLoader_hourly(hourly_df, lookback_daily, x_bicm_vars, x_bicm_pars)
    rtms_X = dataLoader_hourly(hourly_df, lookback_daily, x_rtms_vars, x_rtms_pars)

    test_data = TensorDataset(torch.from_numpy(carp_X), torch.from_numpy(rtmo_X),
                               torch.from_numpy(enba_X), torch.from_numpy(bicm_X),
                               torch.from_numpy(rtms_X))

    test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size, drop_last=True)

    return test_loader

def dataLoader_daily(df, lookback_daily, x_field_vars, x_field_pars):
    df = df[x_field_vars + x_field_pars]

    # Scaling the input data
    scaler = MinMaxScaler()

    # Fit and transform the selected columns and replace them in the dataframe
    df_fit = df.copy()
    df_fit[x_field_vars] = scaler.fit_transform(df[x_field_vars])
    daily_data = df_fit.values

    # Define lookback period and split inputs/labels
    inputs = np.zeros((len(daily_data) - lookback_daily, lookback_daily, len(x_field_vars + x_field_pars)))

    for i in range(lookback_daily, len(daily_data)):
        inputs[i - lookback_daily] = daily_data[i - lookback_daily:i, :]

    return inputs

def dataLoader_hourly(df, lookback_daily, x_field_vars, x_field_pars):
    df = df[x_field_vars + x_field_pars]

    # Scaling the input data
    scaler = MinMaxScaler()

    # Fit and transform the selected columns and replace them in the dataframe
    df_fit = df.copy()
    df_fit[x_field_vars] = scaler.fit_transform(df[x_field_vars])
    hourly_data = df_fit.values

    num_hours = hourly_data.shape[0]
    num_columns = hourly_data.shape[1]
    num_days = num_hours // 24
    daily_hourly_data = hourly_data.reshape((num_days, 24, num_columns))

    # Define lookback period and split inputs/labels
    inputs = np.zeros((len(daily_hourly_data) - lookback_daily, lookback_daily, 24, len(x_field_vars + x_field_pars)))

    for i in range(lookback_daily, len(daily_hourly_data)):
        inputs[i - lookback_daily] = daily_hourly_data[i - lookback_daily:i, :, :]

    return inputs

def sip_leaf(prospectpro, Cab, Car, Ant, Cbrown, Cw, Cm):
    """SIP D Plant leaf reflectance and transmittance modeled
    from 400 nm to 2500 nm (1 nm step).
    Parameters
    ----------
    Cab : float
        chlorophyll a+b content (mug cm-2).
    Car : float
        carotenoids content (mug cm-2).
    Cbrown : float
        brown pigments concentration (unitless).
    Cw : float
        equivalent water thickness (g cm-2 or cm).
    Cm : float
        dry matter content (g cm-2).
    Ant : float
        Anthocianins concentration (mug cm-2).
    Alpha: float
        Constant for the optimal size of the leaf scattering element
    Returns
    -------
    l : array_like
        wavelenght (nm).
    rho : array_like
        leaf reflectance .
    tau : array_like
        leaf transmittance .
    """

    Alpha = 600  # constant for the optimal size of the leaf scattering element
    fLMA_k = 2519.65
    gLMA_k = -631.54
    gLMA_b = 0.0086

    Cab_k = prospectpro[:, 2].reshape(-1, 1)
    Car_k = prospectpro[:, 3].reshape(-1, 1)
    Ant_k = prospectpro[:, 4].reshape(-1, 1)
    Cbrown_k = prospectpro[:, 5].reshape(-1, 1)
    Cw_k = prospectpro[:, 6].reshape(-1, 1)
    Cm_k = prospectpro[:, 7].reshape(-1, 1)

    kall = (Cab * Cab_k + Car * Car_k + Ant * Ant_k + Cbrown * Cbrown_k + Cw * Cw_k + Cm * Cm_k) / (Cm * Alpha)
    w0 = np.exp(-kall)

    # spectral invariant parameters
    fLMA = fLMA_k * Cm
    gLMA = gLMA_k * (Cm - gLMA_b)

    p = 1 - (1 - np.exp(-fLMA)) / fLMA
    q = 2 / (1 + np.exp(gLMA)) - 1
    qabs = np.sqrt(q ** 2)

    # leaf single scattering albedo
    w = w0 * (1 - p) / (1 - p * w0)

    # leaf reflectance and leaf transmittance
    refl = w * (1 / 2 + q / 2 * (1 - p * w0) / (1 - qabs * p * w0))

    return refl


