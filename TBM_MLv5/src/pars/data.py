import pandas as pd
import numpy as np
import joblib


class Data:
    """
    Data class for the machine learning model
    """

    def __init__(self, i, p, data_root):

        hourly_data_paths = [data_root+"HARV.csv", data_root+f"{i}_model_output_hourly.pkl", data_root+f"{i}_model_output_daily.pkl"]
        daily_data_path = data_root+f"{i}_model_output_daily.pkl"
        spectral_data_paths = [data_root+f"{i}_model_output_spectral.pkl", "../../data/mod_list_wavelength_resample.pkl"]

        self.params = p.model_params()
        self.params['RUB'], self.params['CB6F'], self.params['BallBerry0'] = pd.read_csv(data_root + "HARV_pars.csv").iloc[i]

        self.params_max = p.model_params_max()

        self.prospectpro = np.loadtxt("support/dataSpec_PDB_resample.txt")

        """
        Define the length of the series data
        """
        self.hour_length = 26304
        self.daily_length = 1096
        self.spectral_length = 323

        hourly_df = self.read_hourly_data(hourly_data_paths)
        daily_df = self.read_daily_data(daily_data_path, hourly_df)
        spectral_df = self.read_spectral_data(spectral_data_paths)

        carbon_pool_df = self.create_pars(len(daily_df), 'carbon_pool')

        common_df_d = self.create_pars(len(daily_df), 'common')
        common_df_h = self.create_pars(len(hourly_df), 'common')
        RTM_df = self.create_pars(len(hourly_df), 'radiative_transfer')
        biochemistry_df = self.create_pars(len(hourly_df), 'biochemistry')
        SIF_df = self.create_pars(len(hourly_df), 'fluorescence')

        spec_optical_df = self.hyperspectral_pars(len(hourly_df))

        # If you want to concatenate along columns, specify axis=1
        self.daily_df = pd.concat([daily_df, common_df_d, carbon_pool_df], axis=1)
        self.hourly_df = pd.concat([hourly_df, common_df_h, spectral_df, RTM_df, biochemistry_df, SIF_df, spec_optical_df], axis=1)

        self.daily_df = self.daily_df.loc[:, ~self.daily_df.columns.duplicated()]
        self.hourly_df = self.hourly_df.loc[:, ~self.hourly_df.columns.duplicated()]

    def read_hourly_data(self, hourly_data_paths):
        input_path, output_hourly_path, output_daily_path = hourly_data_paths

        # driving data
        df = pd.read_csv(input_path)
        df = df[['year', 'doy', 'TA', 'VPD', 'PAR_up', 'SW', 'wds']]
        df.rename(columns={'PAR_up': 'PAR'}, inplace=True)

        output_daily = joblib.load(output_daily_path)
        output_hourly = joblib.load(output_hourly_path)
        output_hourly = output_hourly.reshape(-1, 21)

        # LAI
        LAI = np.repeat(output_daily[:, -1], 24)[0:self.hour_length]
        df['LAI'] = LAI

        df['NEE'] = output_hourly[0:self.hour_length, 0]
        df['GPP'] = output_hourly[0:self.hour_length, 1]
        df['fPAR'] = output_hourly[0:self.hour_length, 2]
        df['APAR'] = df['fPAR'] * df['PAR']

        df['SIFu'] = output_hourly[0:self.hour_length, 3]
        df['SIFh'] = output_hourly[0:self.hour_length, 4]

        df['LST'] = output_hourly[0:self.hour_length, 5]

        df['fqe_u'] = output_hourly[0:self.hour_length, 6] + output_hourly[0:self.hour_length, 7]
        df['fqe_h'] = output_hourly[0:self.hour_length, 8] + output_hourly[0:self.hour_length, 9]

        df['Rnet_u_o'] = output_hourly[0:self.hour_length, 10]
        df['Rnet_u_t'] = output_hourly[0:self.hour_length, 11]
        df['Rnet_h_o'] = output_hourly[0:self.hour_length, 12]
        df['Rnet_h_t'] = output_hourly[0:self.hour_length, 13]

        df['Rnet_o'] = df['Rnet_u_o'] + df['Rnet_h_o']
        df['Rnet_t'] = df['Rnet_u_t'] + df['Rnet_h_t']

        df['Tcu'] = output_hourly[0:self.hour_length, 14]
        df['Tch'] = output_hourly[0:self.hour_length, 15]
        df['Tsu'] = output_hourly[0:self.hour_length, 16]
        df['Tsh'] = output_hourly[0:self.hour_length, 17]

        df['SZA'] = output_hourly[0:self.hour_length, 18]
        df['VZA'] = output_hourly[0:self.hour_length, 19]
        df['SAA'] = output_hourly[0:self.hour_length, 20]

        df = df.fillna(method='ffill').fillna(method='bfill')

        return df

    def read_daily_data(self, daily_data_path, hourly_df):
        model_output = joblib.load(daily_data_path)
        data = {
            "LAI": model_output[:self.daily_length, -1],
            "GPP": model_output[:self.daily_length, -2]
        }

        # Group the DataFrame by the 'Category' column and calculate the mean of the 'Value' column
        df_doy = hourly_df.groupby(["year", "doy"]).mean().reset_index().rename(columns={"Category": "doy"})
        df_daily = pd.DataFrame(data)
        df_daily["doy"] = np.sin(2 * np.pi * (df_doy["doy"] - 1) / 365)
        df_daily["LST"] = df_doy["LST"]

        return df_daily

    def read_spectral_data(self, spectral_data_paths):
        spectral_data_path, wavelength_data_path = spectral_data_paths
        spectral_data = joblib.load(spectral_data_path)
        spectral_data = spectral_data.reshape(-1, self.spectral_length)
        spectral_df = pd.DataFrame(spectral_data[:self.hour_length, 0:self.spectral_length])
        self.wavelength_data = joblib.load(wavelength_data_path)
        spectral_df.columns = [f'canopy_b{i}' for i in self.wavelength_data]

        return spectral_df

    def create_pars(self, length, params_section):
        # Create an empty dictionary to store data
        data_dict = {}

        # Iterate over all items in the chosen section of params dictionary
        for attr, value in self.params[params_section].items():
            max_value = self.params_max[params_section][attr+'_max']
            normalized_value = value / max_value
            data_dict[attr] = np.full(length, normalized_value)

        # Create DataFrame from the data dictionary
        df = pd.DataFrame(data_dict)

        return df

    def hyperspectral_pars(self, length):

        reflectance = self.sip_leaf()
        hyperspectral_data = np.repeat(reflectance.flatten(), length).reshape(length, -1)
        hyperspectral_df = pd.DataFrame(hyperspectral_data)
        hyperspectral_df.columns = [f'leaf_b{i}' for i in self.wavelength_data]

        return hyperspectral_df

    def sip_leaf(self):
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

        Cab_k = self.prospectpro[:, 2].reshape(-1, 1)
        Car_k = self.prospectpro[:, 3].reshape(-1, 1)
        Ant_k = self.prospectpro[:, 4].reshape(-1, 1)
        Cbrown_k = self.prospectpro[:, 5].reshape(-1, 1)
        Cw_k = self.prospectpro[:, 6].reshape(-1, 1)
        Cm_k = self.prospectpro[:, 7].reshape(-1, 1)

        Cab = self.params['radiative_transfer']['Cab']
        Car = self.params['radiative_transfer']['Car']
        Ant = self.params['radiative_transfer']['Cab']
        Cbrown = self.params['radiative_transfer']['Cbrown']
        Cw = self.params['radiative_transfer']['Cw']
        Cm = self.params['radiative_transfer']['Cm']

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

        return refl[0:self.spectral_length]
