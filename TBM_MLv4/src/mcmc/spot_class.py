import spotpy
from spotpy.parameter import Uniform, Normal
from load import read_par_set
from predict import predict


class spot_setup(object):
    # RUB = Uniform(low=60, high=120)
    # CB6F = Uniform(low=60, high=150)
    # BallBerry0 = Uniform(low=0, high=1)

    RUB = Normal(mean=60, stddev=15)
    CB6F = Normal(mean=75, stddev=15)
    BallBerry0 = Normal(mean=0.5, stddev=0.15)

    def __init__(self, model, obs_arr, hourly_df, daily_df, batch_size, x_vars, x_pars, lookback_periods, hidden_dim_list,
                 n_layers_list, output_dim_list, label_scaler):
        self.model = model
        self.hourly_df = hourly_df
        self.daily_df = daily_df
        self.batch_size = batch_size
        self.x_vars = x_vars
        self.x_pars = x_pars
        self.lookback_periods = lookback_periods
        self.hidden_dim_list = hidden_dim_list
        self.n_layers_list = n_layers_list
        self.output_dim_list = output_dim_list
        self.label_scaler = label_scaler

        self.obs_arr = obs_arr

        self.obj_func = spotpy.objectivefunctions.rmse

    def simulation(self, x):
        # Here the model is actualy started with a unique parameter combination that it gets from spotpy for each time the model is called
        pars = [x[0], x[1], x[2]]
        data_loader = read_par_set(pars, self.hourly_df, self.daily_df, self.lookback_periods, self.batch_size, self.x_vars, self.x_pars)
        prd_arr = predict(self.model, data_loader, self.hidden_dim_list, self.n_layers_list, self.output_dim_list, self.batch_size, self.label_scaler)
        return prd_arr

    def evaluation(self):
        return self.obs_arr

    def objectivefunction(self, simulation, evaluation, params=None):
        # SPOTPY expects to get one or multiple values back,
        # that define the performance of the model run
        if not self.obj_func:
            # This is used if not overwritten by user
            like = spotpy.objectivefunctions.rmse(evaluation, simulation)
        else:
            # Way to ensure flexible spot setup class
            like = self.obj_func(evaluation, simulation)
        return like