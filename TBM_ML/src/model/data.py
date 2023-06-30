import pandas as pd
import numpy as np
import joblib


class Data:
    """
    Data class for the machine learning model
    """

    def __init__(self, pars, hourly_data_paths, daily_data_path, spectral_data_path, fields):
        [prospectpro_path] = pars

        """
        Parameters for carbon pool
        """
        self.clspan = 1.00116  # clspan, leaf lifespan               (1.0001 - 5)
        self.lma = 55  # clma, leaf mass per area          (81 - 120) g C m-2
        self.f_auto = 0.5  # f_auto, fraction of GPP respired  (0.3 - 0.7)
        self.f_fol = 0.15  # f_fol, frac GPP to foliage        (0.01 - 0.5)
        self.d_onset = 130.  # d_onset, clab release date       (1 - 365) (60,150)
        self.cronset = 20.  # cronset, clab release period      (10 - 100)
        self.d_fall = 300.  # d_fall, date of leaf fall        (1 - 365) (242,332)
        self.crfall = 35.  # crfall, leaf fall period          (10 - 100)

        self.clspan_max = 5.0  # clspan, leaf lifespan               (1.0001 - 5)
        self.lma_max = 120
        self.f_auto_max = 0.7  # f_auto, fraction of GPP respired  (0.3 - 0.7)
        self.f_fol_max = 0.5  # f_fol, frac GPP to foliage        (0.01 - 0.5)
        self.d_onset_max = 365  # d_onset, clab release date       (1 - 365) (60,150)
        self.cronset_max = 100.  # cronset, clab release period      (10 - 100)
        self.d_fall_max = 365.  # d_fall, date of leaf fall        (1 - 365) (242,332)
        self.crfall_max = 100.  # crfall, leaf fall period          (10 - 100)

        self.CI = 0.72  # clumping index
        self.LiDf = 55

        self.CI_max = 1.0
        self.LiDf_max = 100

        """
        Parameters for biochemistry
        """
        self.RUB = 30  # [umol sites m-2] Rubisco density
        self.Rdsc = 0.01  # [] Scalar for mitochondrial (dark) respiration
        self.CB6F = 35  # [umol sites m-2] Cyt b6f density
        self.gm = 0.01  # [] mesophyll conductance to CO2
        self.e = 0.92

        self.RUB_max = 120  # [umol sites m-2] Rubisco density
        self.Rdsc_max = 0.05  # [] Scalar for mitochondrial (dark) respiration
        self.CB6F_max = 150  # [umol sites m-2] Cyt b6f density
        self.gm_max = 5.0  # [] mesophyll conductance to CO2
        self.e_max = 1.0

        self.BallBerrySlope = 10
        self.BallBerry0 = 0.01

        self.BallBerrySlope_max = 100
        self.BallBerry0_max = 1.0

        """
        Parameters for Radiative Transfer Model in optical/thermal band
        """
        self.Cab = 28.12
        self.Car = 5.56
        self.Cm = self.lma / 10000.0
        self.Cbrown = 0.185
        self.Cw = 0.00597
        self.Ant = 1.966

        self.Cab_max = 80
        self.Car_max = 20
        self.Cm_max = 120 / 10000.0
        self.Cbrown_max = 1
        self.Cw_max = 1
        self.Ant_max = 10

        self.rho = 0.01
        self.tau = 0.01
        self.rs = 0.06

        self.rho_max = 0.05
        self.tau_max = 0.05
        self.rs_max = 0.1

        self.prospectpro = np.loadtxt(prospectpro_path)

        """
        Parameters for Radiative Transfer Model in fluorescence
        """
        self.eta = 5E-5
        self.eta_max = 1E-4

        """
        Define the length of the series data
        """
        self.hour_length = 26160
        self.daily_length = 1090

        hourly_df = self.read_hourly_data(hourly_data_paths)
        daily_df = self.read_daily_data(daily_data_path, hourly_df)
        spectral_df = self.read_spectral_data(spectral_data_path)

        [carbon_pool_fields, RTM_optical_fields, Energy_balance_fields, RTM_thermal_fields, biochemistry_fields, RTM_SIF_fields] = fields
        carbon_pool_df = self.create_pars(len(daily_df), carbon_pool_fields)
        RTM_optical_df = self.create_pars(len(hourly_df), RTM_optical_fields[:-2101])
        spec_optical_df = self.hyperspectral_pars(len(hourly_df))
        RTM_thermal_df = self.create_pars(len(hourly_df), RTM_thermal_fields)
        biochemistry_df = self.create_pars(len(hourly_df), biochemistry_fields)
        RTM_SIF_df = self.create_pars(len(hourly_df), RTM_SIF_fields)

        # If you want to concatenate along columns, specify axis=1
        self.daily_df = pd.concat([daily_df, carbon_pool_df], axis=1)
        self.hourly_df = pd.concat([hourly_df, spectral_df, RTM_optical_df, RTM_thermal_df, biochemistry_df, RTM_SIF_df, spec_optical_df], axis=1)

        self.daily_df = self.daily_df.loc[:, ~self.daily_df.columns.duplicated()]
        self.hourly_df = self.hourly_df.loc[:, ~self.hourly_df.columns.duplicated()]


    def read_hourly_data(self, hourly_data_paths):
        input_path, sifu_path, sifh_path, output_hourly_path, output_daily_path = hourly_data_paths

        # driving data
        df = pd.read_csv(input_path)
        df = df[['year', 'doy', 'TA', 'VPD', 'PAR_up', 'SW', 'wds', 'nee_gf', 'gpp_gf']]
        df.rename(columns={'PAR_up': 'PAR'}, inplace=True)
        df.rename(columns={'nee_gf': 'NEE_obs'}, inplace=True)
        df.rename(columns={'gpp_gf': 'GPP_obs'}, inplace=True)

        # directional SIF at canopy level
        sif_u = joblib.load(sifu_path)
        sif_h = joblib.load(sifh_path)
        df['SIFu'] = sif_u[:, 2]
        df['SIFh'] = sif_h[:, 2]

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
        df['LST'] = output_hourly[0:self.hour_length, 5]

        df['fqe_u'] = output_hourly[0:self.hour_length, 6] + output_hourly[0:self.hour_length, 7]
        df['fqe_h'] = output_hourly[0:self.hour_length, 8] + output_hourly[0:self.hour_length, 9]

        df['Rnet_u_o'] = output_hourly[0:self.hour_length, 10]
        df['Rnet_u_t'] = output_hourly[0:self.hour_length, 11]
        df['Rnet_h_o'] = output_hourly[0:self.hour_length, 12]
        df['Rnet_h_t'] = output_hourly[0:self.hour_length, 13]

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

    def read_spectral_data(self, spectral_data_path):

        spectral_data = joblib.load(spectral_data_path)
        spectral_data = spectral_data.reshape(-1, 2101)
        spectral_df = pd.DataFrame(spectral_data[:self.hour_length, 0:2101])
        spectral_df.columns = [f'canopy_b{i + 400}' for i in range(spectral_df.shape[1])]

        return spectral_df

    def create_pars(self, length, attributes):

        # Use a dictionary comprehension to generate the dictionary for the DataFrame
        data_dict = {attr: np.full(length, getattr(self, attr) / getattr(self, f"{attr}_max")) for attr in attributes}

        # Create DataFrame
        df = pd.DataFrame(data_dict)

        return df

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

        kall = (self.Cab * Cab_k + self.Car * Car_k + self.Ant * Ant_k + self.Cbrown * Cbrown_k + self.Cw * Cw_k + self.Cm * Cm_k) / (self.Cm * Alpha)
        w0 = np.exp(-kall)

        # spectral invariant parameters
        fLMA = fLMA_k * self.Cm
        gLMA = gLMA_k * (self.Cm - gLMA_b)

        p = 1 - (1 - np.exp(-fLMA)) / fLMA
        q = 2 / (1 + np.exp(gLMA)) - 1
        qabs = np.sqrt(q ** 2)

        # leaf single scattering albedo
        w = w0 * (1 - p) / (1 - p * w0)

        # leaf reflectance and leaf transmittance
        refl = w * (1 / 2 + q / 2 * (1 - p * w0) / (1 - qabs * p * w0))

        return refl[0:2101]

    def hyperspectral_pars(self, length):

        reflectance = self.sip_leaf()
        hyperspectral_data = np.repeat(reflectance.flatten(), length).reshape(length, -1)
        hyperspectral_df = pd.DataFrame(hyperspectral_data)
        hyperspectral_df.columns = [f'leaf_b{i + 400}' for i in range(hyperspectral_data.shape[1])]

        return hyperspectral_df


