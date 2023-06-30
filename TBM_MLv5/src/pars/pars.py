class Par:

    def __init__(self, hidden_dim, n_layer, lookback_periods, batch_size):
        """
        2. define machine learning parameters
        """
        self.carp_hidden_dim = 32
        self.rtmo_hidden_dim = 32
        self.enba_hidden_dim = 32
        self.bicm_hidden_dim = hidden_dim
        self.rtms_hidden_dim = 32

        self.carp_n_layers = 4
        self.rtmo_n_layers = 4
        self.enba_n_layers = 4
        self.bicm_n_layers = n_layer
        self.rtms_n_layers = 4

        self.lookback_periods = lookback_periods
        self.batch_size_daily = 32
        self.batch_size_hourly = batch_size # self.batch_size_daily * 24

        self.test_portion = 0.2
        self.learn_rate = 0.001
        self.EPOCHS = 10  # 200

    def model_params_max(self):
        model_params_max = {
            'common': {
                'CI_max': 1.0,
                'LiDf_max': 100
            },
            'carbon_pool': {
                'clspan_max': 5.0,  # leaf lifespan (1.0001 - 5)
                'lma_max': 120,  # leaf mass per area (81 - 120) g C m-2
                'f_auto_max': 0.7,  # fraction of GPP respired (0.3 - 0.7)
                'f_fol_max': 0.5,  # frac GPP to foliage (0.01 - 0.5)
                'd_onset_max': 365.,  # clab release date (1 - 365) (60,150)
                'cronset_max': 100.,  # clab release period (10 - 100)
                'd_fall_max': 365.,  # date of leaf fall (1 - 365) (242,332)
                'crfall_max': 100.,  # leaf fall period (10 - 100)
            },
            'biochemistry': {
                'RUB_max': 120,  # [umol sites m-2] Rubisco density
                'CB6F_max': 150,  # [umol sites m-2] Cyt b6f density
                'Rdsc_max': 0.05,  # Scalar for mitochondrial (dark) respiration
                'gm_max': 5.0,  # mesophyll conductance to CO2
                'e_max': 1.0,
                'BallBerrySlope_max': 100,
                'BallBerry0_max': 1.0
            },
            'radiative_transfer': {
                'Cab_max': 80.0,
                'Car_max': 20.0,
                'Cbrown_max': 1,
                'Cw_max': 1,
                'Ant_max': 10.0,
                'rho_max': 0.05,
                'tau_max': 0.05,
                'rs_max': 0.1
            },
            'fluorescence': {
                'eta_max': 1E-4
            }
        }
        model_params_max['radiative_transfer']['Cm_max'] = model_params_max['carbon_pool']['lma_max'] / 10000.0

        return model_params_max

    def model_params(self):
        model_params = {
            'common': {
                'CI': 0.72,  # clumping index
                'LiDf': 55,
            },
            'carbon_pool': {
                'clspan': 1.00116,  # leaf lifespan (1.0001 - 5)
                'lma': 55,  # leaf mass per area (81 - 120) g C m-2
                'f_auto': 0.5,  # fraction of GPP respired (0.3 - 0.7)
                'f_fol': 0.15,  # frac GPP to foliage (0.01 - 0.5)
                'd_onset': 130.,  # clab release date (1 - 365) (60,150)
                'cronset': 20.,  # clab release period (10 - 100)
                'd_fall': 300.,  # date of leaf fall (1 - 365) (242,332)
                'crfall': 35.,  # leaf fall period (10 - 100)
            },
            'biochemistry': {
                'RUB': 60,  # [umol sites m-2] Rubisco density
                'CB6F': 75,  # [umol sites m-2] Cyt b6f density
                'Rdsc': 0.01,  # Scalar for mitochondrial (dark) respiration
                'gm': 0.01,  # mesophyll conductance to CO2
                'e': 0.92,
                'BallBerrySlope': 10,
                'BallBerry0': 0.1
            },
            'radiative_transfer': {
                'Cab': 28.12,
                'Car': 5.56,
                'Cbrown': 0.185,
                'Cw': 0.00597,
                'Ant': 1.966,
                'rho': 0.01,
                'tau': 0.01,
                'rs': 0.06
            },
            'fluorescence': {
                'eta': 5E-5
            }
        }
        model_params['radiative_transfer']['Cm'] = model_params['carbon_pool']['lma'] / 10000.0

        return model_params