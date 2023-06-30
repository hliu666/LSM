import joblib
class Var:

    def __init__(self):
        """
        1. load data root and select interested fields
        """

        wavelength_data = joblib.load("../../data/mod_list_wavelength_resample.pkl")

        self.x_carp_vars = ['GPP', 'LST', 'doy']
        self.x_carp_pars = ['clspan', 'lma', 'f_auto', 'f_fol', 'd_onset', 'cronset', 'd_fall', 'crfall', 'CI', 'LiDf']
        self.y_carp_vars = ['LAI']

        self.x_rtmo_vars = ['LAI', 'SW', 'SZA', 'VZA', 'SAA']
        self.x_rtmo_pars = ['Cab', 'Car', 'Cm', 'Cbrown', 'Cw', 'Ant', 'CI', 'LiDf'] + [f'leaf_b{i}' for i in wavelength_data]
        self.y_rtmo_vars = ['fPAR', 'Rnet_o'] + [f'canopy_b{i}' for i in wavelength_data]

        self.x_enba_vars = ['LAI', 'SW', 'TA', 'Rnet_o', 'wds']
        self.x_enba_pars = ['CI', 'LiDf', 'rho', 'tau', 'rs']
        self.y_enba_vars = ['LST']

        self.x_bicm_vars = ['LAI', 'fPAR', 'PAR', 'LST', 'VPD']
        self.x_bicm_pars = ['RUB', 'CB6F', 'Rdsc', 'gm', 'e', 'BallBerrySlope', 'BallBerry0']
        self.y_bicm_vars = ['GPP', 'NEE', 'fqe_u', 'fqe_h']

        self.x_rtms_vars = ['LAI', 'fPAR', 'PAR', 'fqe_u', 'fqe_h', 'SZA', 'VZA', 'SAA']
        self.x_rtms_pars = ['CI', 'LiDf', 'eta']
        self.y_rtms_vars = ['SIFu', 'SIFh']
