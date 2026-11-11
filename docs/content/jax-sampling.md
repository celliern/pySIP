# Bayesian sampling with JAX

pySIP supports real PyMC NumPyro and BlackJAX NUTS, in addition to native PyMC sampling. Both use the same canonical model equations and covariance Kalman likelihood as `fit(backend="jax")`; Bayesian sampling does **not** optimize the frequentist posterior objective.

## Installation and versions

Install the optional JAX sampling dependencies in the same environment as pySIP:

```sh
pip install 'pysip[jax]'
```

These integration choices target **PyMC 6.4.0 / PyTensor 3.3.8** and Python 3.13/3.14, not the environment recorded in the [starting PyMC fast-sampling example](https://www.pymc.io/projects/examples/en/latest/samplers/fast_sampling_with_jax_and_numba.html) (PyMC 5.6.0). NumPyro 0.22.0 requires Python >=3.11 and JAX/JAXlib >=0.7.0. BlackJAX 1.5 requires Python >=3.11 and JAX/JAXlib >=0.9.0. Their Python-independent wheels admit both Python versions, though BlackJAX's classifiers only explicitly list through 3.13; dependency metadata is not a runtime compatibility guarantee. See [NumPyro metadata](https://pypi.org/pypi/numpyro/0.22.0/json), [BlackJAX metadata](https://pypi.org/pypi/blackjax/1.5/json), and [PyMC requirements](https://github.com/pymc-devs/pymc/blob/v6.4.0/pyproject.toml).

**Do not upgrade BlackJAX to 1.6+ with this PyMC version.** PyMC 6.4 calls `blackjax.progress_bar.gen_scan_fn`, even with progress disabled; BlackJAX 1.6 replaced that API. Sources: [PyMC JAX driver](https://github.com/pymc-devs/pymc/blob/v6.4.0/pymc/sampling/jax.py), [BlackJAX 1.5 progress module](https://github.com/blackjax-devs/blackjax/blob/1.5/blackjax/progress_bar.py), [1.6 release notes](https://github.com/blackjax-devs/blackjax/releases/tag/1.6).

## Regressor interface

Give every free parameter a prior, construct your `Regressor` and dataframe as usual, then:

```python
trace = reg.sample(
    df,
    draws=1000,
    tune=1000,
    chains=4,
    random_seed=42,
    nuts_sampler="numpyro",  # or "blackjax"
    nuts={
        "target_accept": 0.9,
        "chain_method": "vectorized",
        "postprocessing_backend": "cpu",
    },
)
```

`Regressor.sample` scopes model construction, compilation, sampling and postprocessing to CPU/float64 and stores the returned `xarray.DataTree` in `reg.trace`. Draws remain **standardized constrained** values. Physical values are `parameter.loc + parameter.scale * draw`. Fixed parameters are absent from the posterior and remain at their snapshotted physical values.

PyMC 6.4 uses **`nuts={...}`**, not the deprecated `nuts_sampler_kwargs={...}`. Its dispatcher lifts `chain_method`, `target_accept` and `postprocessing_backend` out of that dictionary and passes the remaining options to the NUTS kernel. `cores` controls native PyMC multiprocessing, not JAX chain execution. `chain_method="vectorized"` supports multiple chains on a single CPU device; `"parallel"` requires enough visible devices. Neither setting is silently overridden by pySIP. [Versioned dispatch source](https://github.com/pymc-devs/pymc/blob/v6.4.0/pymc/sampling/mcmc.py).

## Direct PyMC interface

The likelihood lowers automatically through PyTensor's `JAXOp` dispatch: no registration function, monkey patch or JAX import is required when constructing an ordinary native model. Direct `pm.sample` works as well. Establish CPU and float64 contexts **around model construction and the entire sample call**:

```python
import jax
import pymc as pm
import pytensor
from pytensor.link.jax import dispatch  # import before scoped JAX settings

with (
    pytensor.config.change_flags(floatX="float64"),
    jax.enable_x64(),
    jax.default_device(jax.devices("cpu")[0]),
    reg.pymc_model(df),
):
    trace = pm.sample(
        draws=1000,
        tune=1000,
        chains=4,
        random_seed=42,
        nuts_sampler="blackjax",
        nuts={"chain_method": "vectorized", "postprocessing_backend": "cpu"},
    )
```

The first import of PyTensor's JAX dispatcher sets global `jax_enable_x64` from `pytensor.config.floatX`; import it before temporary contexts when using the direct interface. `Regressor.sample` handles and restores that import side effect itself. Requesting a float64 dtype with x64 disabled is insufficient. `jax.default_device` does not control `pmap` multi-device execution; use vectorized chains for this CPU workflow. For strict process-wide CPU selection, set `JAX_PLATFORMS=cpu` before starting Python. Sources: [PyTensor dispatcher](https://github.com/pymc-devs/pytensor/blob/rel-3.3.8/pytensor/link/jax/dispatch/basic.py), [JAX precision](https://docs.jax.dev/en/latest/101/default_dtypes.html), [default device scope](https://docs.jax.dev/en/latest/_autosummary/jax.default_device.html).

## Statistical and numerical contract

- `pymc_model` gives the likelihood a vector of **constrained, standardized free parameters** in collection order. The JAX adapter applies only `loc + scale * value`; fixed leaves and model configuration come from a snapshot.
- PyMC supplies prior densities, unconstrained RV transforms and their Jacobians. The adapter adds **none** of pySIP's frequentist priors, parameter penalties, eta inverse transforms, or additional Jacobians. A positive-prior PyMC variable can therefore have a different unconstrained coordinate from a pySIP optimization parameter without changing the likelihood.
- `_Loglike.perform` retains the native square-root estimator. Its symbolic pullback retains the native finite-difference fallback, allowing PyMC's initial gradient eligibility check and native sampling to work without JAX. JAX lowering of both likelihood and symbolic gradient uses the shared pure scan and exact JAX differentiation; it never calls the native estimator or finite-differences the model. NumPyro/BlackJAX differentiate the lowered forward graph themselves.
- PyTensor 3.3.8 provides a lightweight `JAXOp` base with registered `jax_funcify` translation to `perform_jax`; pySIP reuses it instead of installing eager custom dispatch hooks. The Op's native `perform`/`pullback` are still independent of optional JAX. See [official Op extension guide](https://pytensor.readthedocs.io/en/latest/extending/creating_a_numba_jax_op.html), [versioned JAXOp](https://github.com/pymc-devs/pytensor/blob/rel-3.3.8/pytensor/link/jax/ops.py), and [PyMC gradient eligibility logic](https://github.com/pymc-devs/pymc/blob/v6.4.0/pymc/sampling/mcmc.py).
- Custom models must implement traceable `continuous_matrices(theta, *, xp=...)` and, if customized, `discrete_covariance(theta, dt, *, xp=...)`. The default NumPy adapter uses the very same definition. Legacy custom discretization overrides without a corresponding pure operation are rejected rather than ignored.
- NaN output channels are marginalized individually, completely missing rows still propagate state, and trailing all-missing observations are omitted. Rank-deficient or zero process/initial covariance is allowed; **observed innovation covariance must be finite positive definite**. No jitter or covariance clipping is added. Invalid JAX proposals receive infinite negative likelihood rather than a repaired statistical model.
- The existing periodic-kernel Bessel callback is retained with its analytic derivative and `uint32` IEEE transport: callback worker threads do not inherit scoped x64. It is not a callback to the Python likelihood. Models using reverse-mode-only custom derivatives should not request NumPyro's `forward_mode_differentiation=True` ([JAX custom VJP constraints](https://docs.jax.dev/en/latest/_autosummary/jax.custom_vjp.html)).
- This is a CPU/float64 integration, not a GPU performance promise. Compilation is a real upfront cost; use meaningful warmup and check posterior diagnostics. A likelihood installed as `pm.Potential` is not an observed-RV likelihood group for automatic pointwise LOO/WAIC.

## Verification

The regression tests compare compiled JAX log densities and both JAX/symbolic gradient translations with native PyMC, including positive-prior transform Jacobians. A canonical linear-gain model has nontrivial parameter location/scale, a fixed noise parameter, missing observations, and an analytic Normal conjugate posterior. Genuine NumPyro **and** BlackJAX runs through both interfaces must recover its posterior mean/variance without divergences or parameter mutation.

Run after installing the sampler dependencies, once per supported Python environment:

```sh
python -m pytest tests/test_bayesian_jax.py tests/test_bayesian.py tests/test_jax.py -q
```

The implementation research used Context7's official PyMC/PyTensor documentation and inspected the installed PyMC 6.4.0/PyTensor 3.3.8 sources; the versioned links above resolve API ambiguities in older examples. Research and metadata checks alone are not evidence that a sampler has run successfully; use the real posterior checks above for runtime verification.
