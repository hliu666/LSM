import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm

# Define a simple terrestrial biosphere model
def biosphere_model(soil_carbon_turnover, input_carbon):
    # The rate of change of soil carbon is the difference between the input and the output
    output_carbon = input_carbon / soil_carbon_turnover
    return output_carbon

# Define the 'true' parameter
true_soil_carbon_turnover = 50 # years

# Generate some 'observed' data
np.random.seed(0)
input_carbon = np.linspace(0, 100, 100)
observed_carbon = biosphere_model(true_soil_carbon_turnover, input_carbon)
observed_carbon += np.random.normal(scale=5, size=observed_carbon.shape)  # add some noise

# Plot the observed data
plt.scatter(input_carbon, observed_carbon)
plt.xlabel('Input Carbon')
plt.ylabel('Observed Carbon')
plt.show()

# Now we define the MCMC model
with pm.Model() as model:
    # Prior: Uniform distribution. We are totally ignorant about this parameter.
    soil_carbon_turnover = pm.Uniform('soil_carbon_turnover', lower=10, upper=100)

    # Likelihood: Normal distribution. We assume the observations are normally distributed around the model prediction.
    mu = biosphere_model(soil_carbon_turnover, input_carbon)
    sigma = pm.HalfNormal('sigma', sd=10)  # assume we know the standard deviation
    Y_obs = pm.Normal('Y_obs', mu=mu, sd=sigma, observed=observed_carbon)

    # Run the MCMC method
    trace = pm.sample(2000, tune=1000, cores=1, step=pm.Metropolis())

# We can then inspect the results
pm.traceplot(trace)
plt.show()
