# PySIP advanced usage and internals

This part of the documentation focuses on what happens behind the scenes of PySIP, to build
the intuition of how to use it and how to extend it.

## Parameters and Priors

The [`Parameter`](../api.md#pysip.params.parameter.Parameter) class holds a parameter value with some extra
information: name, bounds, prior, scale...

In its simplest form, a parameter is just a value and a name (the value is even
optional and defaults to 0.0):

```python
>>> from pysip.params import Parameter
>>> p = Parameter(name="p", value=1.0)
>>> p.name, p.theta
('p', 1.0)

```

As you can see in the parameter representation, there is a lot of extra
information you can provide to the parameter. For example, you can specify
bounds and scale:

```python
>>> p = Parameter(name="p", value=1.0, bounds=(0.0, 2.0), scale=0.1)
>>> p.bounds, p.theta
((0.0, 2.0), 0.1)

```

And something really interesting happened here. The transform has been
automatically set to [`logit`](../api.md#pysip.params.transforms.LogitTransform):
this transformation is used to constrain the parameter value to be within the
bounds. This leads to the constrained ($\theta$) parameter space and
the unconstrained ($\eta$) parameter space.

### Constrained and unconstrained parameter space

The parameters are defined both in the unconstrained space and the constrained
space. The former corresponds to the $\mathbb{R}$ space (all the real
numbers). At best, the parameter is standardized and lies around 0.0 with a
standard deviation of 1.0. Of course, this can be difficult to ensure for
unknown parameters.

The latter corresponds to the space where the parameter is constrained and that
is used in the model. For example, a parameter that is constrained to be
positive will have a constrained space of $\mathbb{R}^+$ (all the positive
real numbers).

The next section will explain how to define the transform between the two
spaces.

### Transform and scale

The transformation between the two spaces is done by a transformation function
then is rescaled by the scale parameter.

The transform is automatically set to a [`NoneTransform` transform](../api.md#pysip.params.transforms.NoneTransform)
if no bounds are provided, to a [`LogTransform`](../api.md#pysip.params.transforms.LogTransform)
if the lower bound is 0.0 and to an [`UpperTransform`](../api.md#pysip.params.transforms.UpperTransform)
(resp. [`LowerTransform`](../api.md#pysip.params.transforms.LowerTransform)) if the
upper (resp. lower) bound is set.

!!! warning

    The user can also provide a transform to the parameter: in that case, the bounds
    might not be enforced depending on the transform! It is up to the user to be consistent
    between the bounds and the transform.

The standardized constrained value is mapped to the physical parameter as
`theta = loc + scale * value`. Scale defaults to 1.0 and location to 0.0.

```python
>>> from pysip.params import Parameter
>>> p = Parameter(name="p", value=1.0, scale=0.1)
>>> p.theta, p.eta  # physical constrained value and unconstrained value
(0.1, 1.0)

```

Why all this fuss about the constrained and unconstrained space? Thanks to
that, the optimization algorithm can work in the unconstrained space, which is
much easier to optimize and will naturally respect the bounds in the
constrained space instead of relying on a penalty function or using specialized
algorithms.

The [`transforms`](../api.md#pysip.params.transforms) API documentation will give you
more information about the available transforms and their mathematical
formulation.

Alternatively, the `penalty` attribute can be used to define a penalty
function that will be used to penalize the parameter value if it is out of
bounds.

As an example of implementation, see:

```python
from pysip.params.transforms import ParameterTransform, register_transform

@register_transform
class LogTransform(ParameterTransform):
    """Log transform, i.e. θ = exp(η)"""

    name = "log"

    def transform(self, θ: float) -> float:
        return np.log(θ)

    def untransform(self, η: float) -> float:
        return np.exp(η)

    def grad_transform(self, θ: float) -> float:
        return 1.0 / θ

    def grad_untransform(self, η: float) -> float:
        return np.exp(η)

    def penalty(self, θ: float) -> float:
        return 1e-12 / (θ - 1e-12)

    def grad_penalty(self, θ: float) -> float:
        return -1e-12 / (θ - 1e-12) ** 2
```

You can follow this template to implement your own transform. Using
`@register_transform` will automatically register the transform in the
transform registry, and make the transform available in the
[`pysip.params.Parameter`](../api.md#pysip.params.parameter.Parameter) class.

### Prior

The prior is a probability distribution that is used by Bayesian inference
as the prior knowledge about the parameter. It is used to compute the posterior
probability distribution of the parameter by updating this prior knowledge with
the likelihood of the statespace estimator.

!!! warning

    Priors and bounds are defined on the standardized constrained value
    `value = (theta - loc) / scale` in both inference workflows.

A good prior is heavily dependent on the problem at hand and expert knowledge. It
can be a uniform distribution between bounds if no prior knowledge is available,
or a Gaussian around a value if the parameter is known to be close to this
value, for example.

Priors are defined in the [`prior`](../api.md#pysip.params.prior) module. The API
documentation will give you more information about the available priors and
their mathematical formulation.

They hold both a `scipy.stats.rv_continuous` as well as a
`pymc.distributions.Continuous`, used for frequentist and Bayesian
inference respectively. An example of implementation is given here:

```python
from pysip.params.prior import BasePrior, PriorMeta
from collections.abc import Callable
from typing import ClassVar, cast
from pytensor.tensor.variable import TensorVariable
from scipy.stats._distn_infrastructure import rv_continuous_frozen
import pymc as pm
from scipy import stats

class Gamma(BasePrior, metaclass=PriorMeta):
    """Gamma prior distribution

    Parameters
    ----------
    alpha: float
        Shape parameter of the gamma distribution
    beta: float
        Rate parameter of the gamma distribution
    """

    alpha: float = 3.0
    beta: float = 1.0
    _scipy_factory: ClassVar[Callable[..., rv_continuous_frozen]] = (
        lambda a, b: cast(rv_continuous_frozen, stats.gamma(a=a, scale=1.0 / b))
    )
    _pymc_factory: ClassVar[Callable[..., TensorVariable]] = pm.Gamma
```

The metaclass generates the public `scipy_dist` and `pymc_dist` properties
from the two typed class-level factories. Numeric annotated fields become
constructor parameters, in declaration order. The factories receive those
values positionally. Keep distribution factories out of field annotations:
annotations describe types, not executable distribution objects.

### Parameters collection and fixed parameters

Finally, a [`Parameters`](../api.md#pysip.params.parameters.Parameters) collection holds a
collection of parameters: it is used to define the parameters of a model, and
their values can be updated at once by setting the
[`eta`](../api.md#pysip.params.parameters.Parameters.eta) or
[`theta`](../api.md#pysip.params.parameters.Parameters.theta) attribute.

```python
>>> from pysip.params import Parameters
>>> parameters = Parameters(
...     [
...         Parameter(name="a", value=1.0, bounds=(0.0, 10.0)),
...         Parameter(name="b", value=3.0, bounds=(0.0, None)),
...         Parameter(name="c", value=2.0, transform="fixed"),
...     ]
... )
>>> parameters.theta = [1, 2, 3]
>>> parameters.theta.tolist()
[1.0, 2.0, 3.0]
>>> parameters.eta_free = [2, 1]
>>> import numpy as np
>>> np.testing.assert_allclose(parameters.theta, [10 / (1 + np.exp(-2)), np.e, 3])

```

You may have noticed the "fixed" transform. It is used to define a parameter
that is not supposed to be optimized.

That leads to the notion of fixed and free parameters: the latter can be updated
using the `_free` suffix:
[`eta_free`](../api.md#pysip.params.parameters.Parameters.eta_free) or
[`theta_free`](../api.md#pysip.params.parameters.Parameters.theta_free) attribute.

```python
>>> parameters.theta_free = [1, 2]
>>> parameters.theta.tolist()
[1.0, 2.0, 3.0]

```

## Statespaces

The [`StateSpace`](../api.md#pysip.statespace.base.StateSpace) class is the base class
that defines a continuous model in PySIP. The models are defined using a
stochastic statespace representation as described here:

$$
\begin{aligned}
x_{t+1} &= A x_t + B u_t + \mathcal{N}(0, Q) \\
y_t &= C x_t + D u_t + \mathcal{N}(0, R)
\end{aligned}
$$

with $x_t$ the state vector, $u_t$ the input vector and $y_t$
the observation vector.

$A$, $B$, $C$ and $D$ are the state, input, output and
feedthrough matrices respectively. $Q$ and $R$ are the state and
output covariance matrices.

The states are internal to the model and are not accessible to the user. The
input vector holds exogenous variables that are known at the time of the
prediction. The observation vector holds the output of the model that can be
compared to the real output to estimate the parameters.

As the models are defined in continuous time, the statespace model has to be
discretized to be used in the optimization algorithm. The discretization is done
using the [`StateSpace.discretization`](../api.md#pysip.statespace.base.StateSpace.discretization)
method, which returns discrete matrices that depend on the local timestep
$\Delta t$. Within one objective evaluation, equal timesteps share a
discretization. Changing model parameters recomputes those matrices.

Transitions use matrix exponentials by default; thermal models can select
`method="analytic"`. GP process covariance is integrated from the continuous
diffusion, and sums/products compose independent covariance increments without
subtracting large stationary covariances. This matters especially for long
length scales and small sampling intervals.

By default, a zoh (zero order hold) discretization is used, but foh (first order
hold) is also available.

A large list of models is available in the [model list](../model_list.md).

To implement your model, you can follow the template below:

```python
from dataclasses import dataclass
from pysip.statespace.base import RCModel

@dataclass
class Ti_RA(RCModel):

    states = [("TEMPERATURE", "xi", "indoor temperature")]

    params = [
        ("THERMAL_RESISTANCE", "R", "between the outdoor and the indoor"),
        ("THERMAL_CAPACITY", "C", "effective overall capacity"),
        ("SOLAR_APERTURE", "A", "effective solar aperture"),
        ("STATE_DEVIATION", "sigw", "of the indoor dynamic"),
        ("MEASURE_DEVIATION", "sigv", "of the indoor temperature measurements"),
        ("INITIAL_MEAN", "x0", "of the indoor temperature"),
        ("INITIAL_DEVIATION", "sigx0", "of the indoor temperature"),
    ]

    inputs = [
        ("TEMPERATURE", "To", "outdoor temperature"),
        ("POWER", "Qgh", "solar irradiance"),
        ("POWER", "Qh", "HVAC system heat"),
    ]

    outputs = [("TEMPERATURE", "xi", "indoor temperature")]

    def set_constant_continuous_ssm(self):
        self.C[0, 0] = 1.0

    def update_continuous_ssm(self):
        R, C, A, sigw, sigv, x0, sigx0, *_ = self.parameters.theta

        self.A[0, 0] = -1.0 / (C * R)
        self.B[0, :] = [1.0 / (C * R), A / C, 1.0 / C]
        self.Q[0, 0] = sigw
        self.R[0, 0] = sigv
        self.x0[0, 0] = x0
        self.P0[0, 0] = sigx0
```

In the `set_constant_continuous_ssm` method, you define the constant part of the
continuous matrices. In the `update_continuous_ssm` method, you define the
dynamic part of the continuous matrices that needs to be updated at each
parameter change.

## Statespace estimator

The Statespace Estimator is the class that holds the data and the model and that
is used to estimate the parameters. In pySIP, a Square Root Kalman filter is
used to estimate the states and the parameters.

It is written using [numba](https://numba.pydata.org/) to speed up the
computation as much as possible.

The estimator holds the model (and its parameters) and will use provided data
to estimate and predict the states and their covariance. You can have a look at
the [statespace estimators API reference](../api.md#statespace-estimators) for more
detail on the available methods.

Usually, users will not have to use the estimator directly, but will use the
[`pysip.regressors.Regressor`](../api.md#pysip.regressors.Regressor) class that will
handle the estimator and the model for them.

## Regressor

The [`Regressor`](../api.md#pysip.regressors.Regressor) class is the main class
that is used to estimate the parameters of a model. It holds the statespace, the
statespace estimator and gives access to both frequentist and Bayesian inference
methods.

```python
import numpy as np
import pandas as pd
from pysip import Regressor
from pysip.statespace import Matern32

N = 50
np.random.seed(1)
t = np.linspace(0, 1, N)
y = np.sin(12 * t) + 0.66 * np.cos(25 * t) + np.random.randn(N) * 0.25
df = pd.DataFrame(index=pd.Index(t, name="t"), data=y, columns=["y"])

par = [
    dict(name="mscale", value=5.11, bounds=(0, None)),
    dict(name="lscale", value=8.15, bounds=(0, None)),
    dict(name="sigv", value=0.8, bounds=(0, None)),
]
model = Matern32(par)
reg = Regressor(model, outputs=["y"])
```

You have to provide the output and input names that link the statespace model
to the data column names.

The regressor will not be heavily described here, as it is already the focus of
the [quickstart](quickstart.md) and well documented in the [cookbook](../cookbook.md).
