import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm

# Define a simple terrestrial biosphere model
def biosphere_model(a, b, c, input_carbon):
    # The rate of change of soil carbon is the difference between the input and the output
    output_carbon = input_carbon / (a + b + c)
    return output_carbon

# Define the 'true' parameter
a, b, c = 10, 20, 30

# Generate some 'observed' data
np.random.seed(0)
input_carbon = np.linspace(0, 100, 100)
observed_carbon = biosphere_model(a, b, c, input_carbon)
observed_carbon += np.random.normal(scale=5, size=observed_carbon.shape)  # add some noise

# Plot the observed data
plt.scatter(input_carbon, observed_carbon)
plt.xlabel('Input Carbon')
plt.ylabel('Observed Carbon')
plt.show()

# Now we define the MCMC model
with pm.Model() as model:
    # Prior: Uniform distribution. We are totally ignorant about this parameter.
    a = pm.Uniform('a', lower=10, upper=100)
    b = pm.Uniform('b', lower=10, upper=100)
    c = pm.Uniform('c', lower=10, upper=100)

    # Likelihood: Normal distribution. We assume the observations are normally distributed around the model prediction.
    mu = biosphere_model(a, b, c, input_carbon)
    Y_obs = pm.Normal('obs', mu, observed=observed_carbon)

    # Run the MCMC method
    trace = pm.sample(5000, cores=1, step=pm.Metropolis())

# We can then inspect the results
pm.traceplot(trace)
plt.show()

# Compute the mean of the posterior samples
a = trace['a'].mean()
b = trace['b'].mean()
c = trace['c'].mean()
print("Mean: ", a, b, c)

# Compute the median of the posterior samples
a = np.median(trace['a'])
b = np.median(trace['b'])
c = np.median(trace['c'])
print("Median: ", a, b, c)

# Compute the maximum a posteriori estimate
with model:
    map_estimate = pm.find_MAP()
print("MAP estimate: ", map_estimate['a'], map_estimate['b'], map_estimate['c'])
