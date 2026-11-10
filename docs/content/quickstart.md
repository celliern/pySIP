# Quickstart

## Installation

Two options are available:

1. Install the package from the git main branch:

    ```bash
    pip install git+https://github.com/locie/pySIP
    ```

2. Install the package from source:

    ```bash
    git clone https://github.com/locie/pySIP
    cd pySIP
    pip install .
    ```

Note that these two options need git on your system.

## Usage

As a first example, we will use the `pysip` package to estimate the parameters
of a simple model from a set of artificial data, generated from a sinusoidal
function with additive Gaussian noise.

The model will be a Matern32 kernel.

### Data preparation

First, we have to create the data, and store it in a `pandas.DataFrame` object.

```python
>>> import numpy as np
>>> import pandas as pd
>>> from pysip import Regressor
>>> from pysip.statespace import Matern32

>>> N = 20
>>> t = np.linspace(0, 1, N)
>>> rng = np.random.default_rng(1)
>>> y = np.sin(12 * t) + 0.66 * np.cos(25 * t) + rng.normal(0, 0.1, N)
>>> df = pd.DataFrame(index=t, data=y, columns=["y"])

```

In that simple case, we only have one output variable, and no internal state or
input but the package can handle multiple inputs, states and outputs.

### Parameters configuration and model initialization

The parameters are defined as a list of dictionaries, each dictionary
containing the following keys:

- `name`: the name of the parameter
- `value`: the initial value of the parameter (optional, default: 0)
- `bounds`: the bounds of the parameter (optional, default: no bounds)
- `scale` (optional): the scale of the parameter (default: 1)
- `transform`: the transformation to apply to the parameter (optional, deduced
  from the bounds if not provided).
- `prior`: the prior distribution of the parameter (optional, but needed for
  Bayesian parameter estimation).

Our model needs three parameters: the Matern32 length scale, the Matern32 scale
and the noise standard deviation. First, no need to include priors: they are
only needed for Bayesian parameter estimation.

```python
>>> par = [
...     dict(name="mscale", value=1.11, bounds=(0, None)),
...     dict(name="lscale", value=0.15, bounds=(0, None)),
...     dict(name="sigv", value=0.1, bounds=(0, None)),
... ]
>>> model = Matern32(par)

```

### Model evaluation

The model, using the provided parameters, can be evaluated using a statespace
estimator. The one provided by the package is a (fast) Square root Kalman
filter. The `Regressor` class is used to evaluate the model via that estimator
(estimate the state, filter the output or evaluate the state or outputs at
in-between time points).

The output and input names are provided at the initialization of the
`Regressor`. Most of the `Regressor` methods return a `xarray.Dataset` object:
they are a collection of numpy arrays with named dimensions that allow easy
plotting and post-processing. More details are available in the method docstrings
(available in the [API reference](../api.md)).

```python
>>> reg = Regressor(model, outputs=["y"])
>>> nofit_estimation = reg.estimate_output(df)

```

### Simple (frequentist) parameter estimation

We provided a first guess for the parameters. We have seen that the evaluation
result is far from the data, as our guess is away from the true parameters. We
can now estimate the parameters using the `fit` method of the `Regressor` class.
This method will use the `scipy.optimize.minimize` function to find the
parameters that minimize the negative log-likelihood of the model given the
data. After fitting, we can evaluate the model again:

```python
>>> summary = reg.fit(df, disp=False)[0]
>>> fit_estimation = reg.estimate_output(df)

```

We can see how the predictions compare to the data:

![Fit comparison](../_static/figures/quickstart_matern_estimation.svg)

*Fit comparison*

The starting values have been chosen to be not too close to the true values, so
we can see the difference between the fit and the initial values.

### Bayesian parameter estimation

For now, we have done what is a frequentist, deterministic parameter estimation.
But the package also provides a Bayesian parameter estimation method, using the
`pymc` package. The `sample` method of the `Regressor` class will use the `pymc`
package to sample the posterior distribution of the parameters, given the data
and the prior parameter distributions.

This allows us to estimate the mean value of the parameters, but also to estimate
the uncertainty on the parameters fit: instead of a unique value, we will
have multiple samples of plausible parameter sets.

First, we add the prior distributions to the parameters configuration. They
should be instances of the [`BasePrior`](../api.md#pysip.params.prior.BasePrior) class, and chosen
according to the prior knowledge on the parameters. For example, a uniform
distribution is a good choice for a parameter where we have no prior knowledge
but their hypothetical bounds. A normal distribution is a good choice for a
parameter where we have a prior knowledge on the mean value and the standard
deviation.

```python
>>> from pysip.params.prior import Gamma, InverseGamma, Uniform

>>> par = [
...     dict(name="mscale", value=5.11, bounds=(0, None), prior=Gamma(4, 4)),
...     dict(name="lscale", value=2.15, bounds=(0, None), prior=Uniform(0.01, 5)),
...     dict(name="sigv", value=0.3, bounds=(0, None), prior=InverseGamma(3.5, 0.5)),
... ]
>>> model = Matern32(par)
>>> reg = Regressor(model, outputs=["y"])

```

The `Regressor.prior_predictive` method samples parameters and complete noisy
trajectories independently of the observed outputs, including initial-state,
process, and measurement uncertainty.

```python
>>> prior_estimation = reg.prior_predictive(df, samples=20, random_seed=1)

```

Then, the `Regressor.sample` method will use the `pymc` package to sample the
posterior distribution of the parameters, given the data and the prior. The
prior distribution will be updated during the `tune` phase, and then the
posterior distribution will be sampled during the `draws` phase.

These two phases are controlled by the `tune` and `draws` arguments of the
`Regressor.sample` method. These operations can be repeated multiple times using
the `chains` argument (default: 4). If there are enough cores available, the
chains will be run in parallel. The number of cores used can be controlled using
the `cores` argument (default: all available cores).

```python
>>> trace = reg.sample(df, draws=20, tune=20, chains=2, cores=1,
...                    random_seed=1, progressbar=False,
...                    compute_convergence_checks=False)
>>> posterior_estimation = reg.posterior_predictive(df, random_seed=1)

```

After that, the model can be evaluated on the posterior distribution of the
parameters, using the `Regressor.posterior_predictive` method.

The short chains above demonstrate the API; use longer chains and convergence
diagnostics for inference. Predictive methods run serially and accept
`random_seed` for reproducible trajectories. Bayesian trace variables use
`reg.parameters.ids_free` (hierarchical names for composed models), and their
values use the standardized constrained coordinates described in the [advanced
guide](advanced.md#transform-and-scale).

All the data from both the prior and posterior predictive evaluations can be
found under the `Regressor.trace` attribute. It is an `xarray.DataTree`, with
datasets grouped under names such as `prior` and `posterior`. For example,
`reg.trace["posterior"].to_dataset()` gives the posterior samples.
ArviZ 1 provides analysis and plotting methods for these trees; see the
[ArviZ documentation](https://python.arviz.org/) for details.

We can plot the posterior distribution of the parameters, and compare it to the
prior distribution.

![Posterior distribution of the parameters](../_static/figures/quickstart_matern_parameters_dist.svg)

*Posterior distribution of the parameters*

We can see that the prior distribution of the parameters is spread over a large
range of values, while the posterior distribution is more concentrated around
the true values. This is especially true for the `lscale` parameter, where we
used a uniform prior distribution with a large range of values.

We can also plot the prior and the posterior predictive distribution of the
outputs, and compare it to the data.

![Posterior predictive distribution of the outputs](../_static/figures/quickstart_matern_prior_posterior.svg)

*Posterior predictive distribution of the outputs*

As expected, the prior predictive distribution is far from the data, with a huge
standard deviation. The posterior predictive distribution is much closer to the
data, with a smaller standard deviation.

The results here are impressive, but this is a toy model fit on an artificial
dataset. You can see more complex examples in the [cookbook](../cookbook.md),
and have more detail on the internals and the advanced usage of the package in
the [advanced guide](advanced.md).
