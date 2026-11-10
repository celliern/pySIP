"""Pandas-facing state-space regression, optimization and predictive sampling."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import dataclass
from multiprocessing import cpu_count
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import xarray as xr
from pytensor.graph.basic import Apply, Variable
from pytensor.graph.op import Op
from pytensor.tensor.variable import TensorVariable
from scipy.optimize import OptimizeResult, approx_fprime, minimize

from .params.parameters import Parameters
from .statespace.base import Node, StateSpace
from .statespace_estimator import KalmanQR, output_moments
from .utils.statistics import ttest


class _FdiffLoglikeGrad(Op):
    """Finite-difference negative-likelihood gradient in standardized coordinates."""

    itypes = (pt.dvector,)
    otypes = (pt.dvector,)

    def __init__(
        self, reg: Regressor, df: pd.DataFrame, eps: float | np.ndarray | None = None,
    ) -> None:
        """Capture prepared data and the estimator used by the numerical gradient."""
        self.estimator = reg.estimator
        self.data = reg.prepare_data(df)
        self.eps = eps

    def perform(
        self, node: Apply, inputs: Sequence[np.ndarray], output_storage: list[list[Any]],
    ) -> None:
        """Write the gradient into PyTensor's mutable output storage."""
        def _target(eta: np.ndarray) -> float:
            """Evaluate negative likelihood on an independent estimator copy."""
            estimator = deepcopy(self.estimator)
            # StateSpace.__post_init__ always initializes its parameter collection.
            cast(Parameters, estimator.ss.parameters).theta_sd = eta
            return estimator.log_likelihood(*self.data)

        (eta,) = inputs  # this will contain my variables
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            output_storage[0][0] = approx_fprime(eta, _target, self.eps)


class _Loglike(Op):
    """PyTensor scalar negative likelihood for standardized free parameters."""

    itypes = (pt.dvector,)
    otypes = (pt.dscalar,)

    def __init__(
        self, reg: Regressor, df: pd.DataFrame, eps: float | np.ndarray | None = None,
    ) -> None:
        """Capture data and construct the finite-difference gradient operation."""
        self.estimator = reg.estimator
        self.data = reg.prepare_data(df)
        self.eps = eps
        self.logpgrad = _FdiffLoglikeGrad(reg, df, eps)

    def perform(
        self, node: Apply, inputs: Sequence[np.ndarray], output_storage: list[list[Any]],
    ) -> None:
        """Write scalar negative likelihood into PyTensor's dynamic storage."""
        estimator = deepcopy(self.estimator)
        (eta,) = inputs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # StateSpace.__post_init__ always initializes its parameter collection.
            cast(Parameters, estimator.ss.parameters).theta_sd = eta
            logl = estimator.log_likelihood(*self.data)
        output_storage[0][0] = np.array(logl)

    def pullback(
        self, inputs: Sequence[Variable], outputs: Sequence[Variable],
        cotangents: Sequence[Variable],
    ) -> list[Variable]:
        """Multiply tensor cotangents by the numerical parameter gradient.

        The declared input/output tensor types guarantee tensor variables here.
        """
        (eta,) = inputs
        cotangent = cast(TensorVariable, cotangents[0])
        gradient = cast(TensorVariable, self.logpgrad(eta))
        return [cotangent * gradient]


@dataclass
class Regressor:
    """Fit, filter and simulate a state-space model with named pandas data.

    Fitting mutates model parameters; posterior/prior samples use standardized
    constrained parameter coordinates, not physical or unconstrained values.

    Parameters
    ----------
    ss : StateSpace()
        State-space model
    inputs : str or list of str, optional
        Input column names, by default None. If None, they are inferred from the
        state-space model.
    outputs : str or list of str, optional
        Output column names, by default None. If None, they are inferred from the
        state-space model.
    time_scale : str
        Time series frequency, e.g. 's': seconds, 'D': days, etc.
    """

    ss: StateSpace
    inputs: str | Sequence[str] | None = None
    outputs: str | Sequence[str] | None = None
    time_scale: str = "s"

    def __post_init__(self) -> None:
        """Create the estimator and normalize names to sequences for later use."""
        self.estimator: KalmanQR = KalmanQR(self.ss)
        if self.inputs is None:
            self.inputs = [node.name for node in cast(list[Node], self.ss.inputs)]
        if self.outputs is None:
            self.outputs = [node.name for node in cast(list[Node], self.ss.outputs)]
        self.states: list[str] = [
            node.name for node in cast(list[Node], self.ss.states)
        ]
        if isinstance(self.inputs, str):
            self.inputs = [self.inputs]
        if isinstance(self.outputs, str):
            self.outputs = [self.outputs]

    @property
    def parameters(self) -> Parameters:
        """Return parameters initialized by StateSpace.__post_init__."""
        return cast(Parameters, self.ss.parameters)

    def prepare_data(
        self, df: pd.DataFrame, with_outputs: bool = True,
    ) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Prepare time steps, inputs, input derivatives and observations.

        Parameters
        ----------
        df : pd.DataFrame
            Time-indexed data with at least two rows and configured columns.
        with_outputs : bool, optional
            Read observations when true; otherwise produce NaN observations.

        Returns
        -------
        tuple
            dt: Series ``(nt,)`` of forward intervals; datetime intervals use
            time_scale, numeric indexes retain their units. u/dtu: DataFrames
            ``(nt, nu)``; y: DataFrame ``(nt, ny)``, or one NaN column when
            outputs are omitted. The source dataframe is not modified.

        Raises
        ------
        KeyError
            A configured input or output column is missing.
        ValueError
            Time intervals, inputs or computed derivatives are nonfinite.
        TypeError
            Data is not a dataframe or observations contain infinite values.
        """
        return self.ss.prepare_data(
            df, self.inputs, self.outputs if with_outputs else False, self.time_scale
        )

    def simulate(self, df: pd.DataFrame) -> xr.Dataset:
        """Simulate noisy states/outputs at df's times without conditioning on y.

        Parameters
        ----------
        df : pd.DataFrame
            Time-indexed exogenous inputs; observation columns are not required.

        Returns
        -------
        xr.Dataset
            Simulated x ``(time, states)`` and y ``(time, outputs)`` using
            current model parameters and default initial distribution.
        """

        dt, u, dtu, _ = self.prepare_data(df, with_outputs=False)
        return self.estimator.simulate(dt, u, dtu).to_xarray(
            df.index.name or "time", df.index, self.states, cast(Sequence[str], self.outputs)
        )

    def estimate_output(
        self,
        df: pd.DataFrame,
        x0: np.ndarray | None = None,
        P0: np.ndarray | None = None,
        use_outputs: bool = True,
    ) -> xr.Dataset:
        """Estimate the output filtered distribution

        Parameters
        ----------
        df : pandas.DataFrame
            Training data
        x0 : numpy.ndarray, optional
            Initial column mean (nx, 1); defaults to the model's x0.
        P0 : numpy.ndarray, optional
            Upper initial covariance factor (nx, nx), with covariance
            P0.T @ P0; defaults to the model's P0.
        use_outputs : bool, optional
            Whether to use the data outputs to do the estimation, by default True

        Returns
        -------
        xr.Dataset
            y and y_std on (time, outputs), including measurement noise.
        """

        dt, u, dtu, y = self.prepare_data(df, with_outputs=use_outputs)
        return self.estimator.estimate_output(dt, u, dtu, y, x0, P0).to_xarray(
            df.index.name or "time", df.index, self.states, cast(Sequence[str], self.outputs)
        )

    def estimate_states(
        self,
        df: pd.DataFrame,
        x0: np.ndarray | None = None,
        P0: np.ndarray | None = None,
        smooth: bool = False,
        use_outputs: bool = True,
    ) -> xr.Dataset:
        """Estimate filtered or RTS-smoothed states.

        Parameters
        ----------
        df : pd.DataFrame
            Time-indexed exogenous inputs and observations.
        x0 : np.ndarray, shape (nx, 1), optional
            Initial mean; defaults to the model's x0.
        P0 : np.ndarray, shape (nx, nx), optional
            Upper initial covariance factor; defaults to the model's P0.
        smooth : bool, optional
            Use RTS smoothing instead of filtering.
        use_outputs : bool, optional
            Condition on observed outputs when true; otherwise propagate freely.

        Returns
        -------
        xr.Dataset
            x: means (time, states); P: full covariances (time, states, states);
            k: filtering whitened residuals (time, outputs); S: upper innovation
            factors (time, outputs, outputs). Missing channels retain NaNs.
        """

        dt, u, dtu, y = self.prepare_data(df, with_outputs=use_outputs)

        return (
            self.estimator.smoothing(dt, u, dtu, y, x0, P0)
            if smooth
            else self.estimator.filtering(dt, u, dtu, y, x0, P0)
        ).to_xarray(
            df.index.name or "time", df.index, self.states, cast(Sequence[str], self.outputs)
        )

    def _target(
        self,
        eta: np.ndarray,
        dt: pd.Series,
        u: pd.DataFrame,
        dtu: pd.DataFrame,
        y: pd.DataFrame,
    ) -> float:
        """Return penalized negative posterior for unconstrained free eta.

        Uses prepared pandas dt/u/dtu/y and an independent estimator copy;
        neither this regressor's parameters nor source data are modified.
        """
        estimator = deepcopy(self.estimator)
        # The copied StateSpace retains its post-init parameter collection.
        parameters = cast(Parameters, estimator.ss.parameters)
        parameters.eta_free = eta
        log_likelihood = estimator.log_likelihood(dt, u, dtu, y)
        log_posterior = (
            log_likelihood
            - parameters.prior
            + parameters.penalty
        )

        return log_posterior

    def fit(
        self,
        df: pd.DataFrame,
        options: dict[str, Any] | None = None,
        *,
        init: Literal["unconstrained", "prior", "zero", "fixed", "value"] = "fixed",
        hpd: float = 0.95,
        jac: Literal["2-point", "3-point", "cs"] | Callable[..., np.ndarray] | bool | None = "3-point",
        method: str | Callable[..., OptimizeResult] = "BFGS",
        **minimize_options: Any,
    ) -> tuple[pd.DataFrame, pd.DataFrame, OptimizeResult]:
        """Estimate the parameters of the state-space model.

        Parameters
        ----------
        df : pandas.DataFrame
            Training data
        options : dict, optional, deprecated
            Options for the minimization method. You should use named arguments instead.
            Usage of this argument will raise a warning and will be removed in the
            future.
        init : str, optional
            Method to initialize the parameters. Options are:
            - 'unconstrained': initialize the parameters in the unconstrained space
            - 'prior': initialize the parameters using the prior distribution
            - 'zero': initialize the parameters to zero
            - 'fixed': initialize the parameters to the fixed values
            - 'value': initialize the parameters to the given values
        hpd : float, optional
            Highest posterior density interval. Used only when `init='prior'`.
        jac : str, callable, bool or None, optional
            Gradient specification forwarded to scipy.optimize.minimize.
        method : str or callable, optional
            Optimization method; defaults to BFGS.
        minimize_options : dict, optional
            Options for the minimization method. See `scipy.optimize.minimize` for
            details. Compared to the original `scipy.optimize.minimize` function, the
            following options are set by default:
            - `method='BFGS'`
            - `jac='3-point'`
            - `disp=True`
            - `gtol=1e-4`

        Returns
        -------
        pandas.DataFrame
            Dataframe with the estimated parameters, their standard deviation, the
            p-value of the t-test and penalty values.
        pandas.DataFrame
            Dataframe with the correlation matrix of the estimated parameters.
        scipy.optimize.OptimizeResult
            Optimizer result. Fitted physical parameter values are written back
            to the model; ``summary_``, ``corr_`` and ``results_`` are stored
            on this object.
        """
        if options is not None:
            warnings.warn(
                "Use of the options argument is deprecated and will raise an error in "
                "the future. Prefer using named arguments (kwargs) instead.",
                DeprecationWarning,
            )
            minimize_options.update(options)
        minimize_options = {"disp": True, "gtol": 1e-4} | minimize_options

        self.parameters.eta_free = self.parameters.init_parameters(1, init, hpd)
        data = self.prepare_data(df)

        results = minimize(
            fun=self._target,
            x0=self.parameters.eta_free,
            args=data,
            method=method,
            jac=jac,
            options=minimize_options,
        )

        self.parameters.eta_free = results.x
        # inverse jacobian of the transform eta = f(theta)
        inv_jac = np.diag(1.0 / np.array(self.parameters.eta_jacobian))

        # covariance matrix in the constrained space (e.g. theta)
        cov_theta = inv_jac @ results.hess_inv @ inv_jac

        # standard deviation of the constrained parameters
        sig_theta = np.sqrt(np.diag(cov_theta)) * self.parameters.scale
        inv_sig_theta = np.diag(1.0 / np.sqrt(np.diag(cov_theta)))

        # correlation matrix of the constrained parameters
        corr_matrix = inv_sig_theta @ cov_theta @ inv_sig_theta
        df = pd.DataFrame(
            data=np.vstack(
                [
                    self.parameters.theta_free,
                    sig_theta,
                    ttest(self.parameters.theta_free, sig_theta, len(data[0])),
                    np.abs(results.jac),
                    np.abs(self.parameters.d_penalty),
                ]
            ).T,
            columns=["θ", "σ(θ)", "pvalue", "|g(η)|", "|dpen(θ)|"],
            index=self.parameters.names_free,
        )
        df_corr = pd.DataFrame(
            data=corr_matrix,
            index=self.parameters.names_free,
            columns=self.parameters.names_free,
        )

        self.summary_ = df
        self.corr_ = df_corr
        self.results_ = results

        return df, df_corr, results

    def eval_residuals(
        self,
        df: pd.DataFrame,
        x0: np.ndarray | None = None,
        P0: np.ndarray | None = None,
    ) -> xr.Dataset:
        """Compute the standardized residuals

        Parameters
        ----------
        df : pandas.DataFrame
            Training data
        x0 : numpy.ndarray, optional
            Initial column mean (nx, 1); defaults to the model's x0.
        P0 : numpy.ndarray, optional
            Upper initial covariance factor (nx, nx); defaults to model P0.

        Returns
        -------
        xr.Dataset
            residual: whitened innovations (time, outputs); residual_std: upper
            innovation factors (time, outputs, outputs), not marginal deviations.
        """

        dt, u, dtu, y = self.prepare_data(df)
        res = self.estimator.filtering(dt, u, dtu, y, x0, P0).to_xarray(
            df.index.name or "time", df.index, self.states, cast(Sequence[str], self.outputs)
        )
        return res[["k", "S"]].rename({"k": "residual", "S": "residual_std"})

    def log_likelihood(self, df: pd.DataFrame) -> float:
        """Return scalar negative log likelihood using df's observed channels.

        NaN observations are omitted. Configured input/output names and time
        units are used; refreshes model matrices from current parameters.
        """

        dt, u, dtu, y = self.prepare_data(df)
        return self.estimator.log_likelihood(dt, u, dtu, y)

    def log_posterior(
        self,
        df: pd.DataFrame,
    ) -> float:
        """Return negative log likelihood minus log prior plus parameter penalty.

        df supplies time-indexed configured inputs/outputs; missing observations
        are omitted. Returns a scalar at the current model parameters.
        """
        return (
            self.log_likelihood(df)
            - self.parameters.prior
            + self.parameters.penalty
        )

    def predict(
        self,
        df: pd.DataFrame,
        tnew: np.ndarray | pd.Series | None = None,
        x0: np.ndarray | None = None,
        P0: np.ndarray | None = None,
        smooth: bool = False,
        use_outputs: bool = True,
    ) -> xr.Dataset:
        """State-space model output prediction

        Parameters
        ----------
        df : pandas.DataFrame
            Training data
        tnew : numpy.ndarray or pandas.Series, optional
            New time instants
        x0 : numpy.ndarray, optional
            Initial column mean (nx, 1); defaults to the model's x0.
        P0 : numpy.ndarray, optional
            Upper initial covariance factor (nx, nx); defaults to model P0.
        smooth : bool, optional
            If True, the Kalman smoother is used instead of the Kalman filter
        use_outputs : bool, optional
            Condition on observations when true (the default).

        Returns
        -------
        xr.Dataset
            State estimates plus y_mean/y_std (time, outputs), including
            measurement noise. New times are merged and input values linearly
            interpolated; outputs at new times remain unobserved.

        Raises
        ------
        NotImplementedError
            If the model has more than one output.
        """

        if self.ss.ny > 1:
            raise NotImplementedError
        if tnew is not None:
            itp_df = pd.concat(
                [df, df.reindex(tnew).drop(df.index, errors="ignore")]
            ).sort_index()
            itp_df[self.inputs] = itp_df[self.inputs].interpolate(method="linear")
        else:
            itp_df = df.copy()
        ds = self.estimate_states(
            itp_df, smooth=smooth, x0=x0, P0=P0, use_outputs=use_outputs
        )
        idx_name = df.index.name or "time"
        if tnew is not None:
            ds = ds.sel({idx_name: tnew})
        mean, std = output_moments(
            self.ss.C,
            self.ss.D,
            self.ss.R,
            ds.x.values[..., None],
            ds.P.values,
            itp_df.loc[ds[idx_name].values, self.inputs].to_numpy()[..., None],
        )
        ds["y_mean"] = ((idx_name, "outputs"), mean[..., 0])
        ds["y_std"] = ((idx_name, "outputs"), std[..., 0])
        return ds

    @property
    def pymc_model(reg: Regressor) -> Callable[[pd.DataFrame], pm.Model]:
        """Return a PyMC model class with standardized priors and likelihood."""
        class PyMCModel(pm.Model):
            """PyMC likelihood model bound to this regressor's parameter priors."""

            def __init__(self, df: pd.DataFrame) -> None:
                """Create prior variables and negative-likelihood potential.

                Raises ValueError when a free parameter has no prior.
                """
                super().__init__()
                theta = []
                for name, par in zip(
                    reg.parameters.ids_free, reg.parameters.parameters_free
                ):
                    if par.prior is None:
                        raise ValueError(f"Free parameter {name!r} needs a prior")
                    theta.append(par.prior.pymc_dist(name))
                theta = pt.as_tensor_variable(theta)
                # _Loglike has exactly one scalar tensor output.
                likelihood = cast(TensorVariable, _Loglike(reg, df)(theta))
                pm.Potential("likelihood", -likelihood)

        # PyMC's inherited __new__ signature does not describe this df constructor.
        return cast(Callable[[pd.DataFrame], pm.Model], PyMCModel)

    def sample(
        self, df: pd.DataFrame, draws: int = 1000, tune: int = 500,
        chains: int = 4, **kwargs: Any,
    ) -> xr.DataTree:
        """Sample from the posterior distribution

        Parameters
        ----------
        df : pandas.DataFrame
            Dataframe containing the inputs and outputs
        draws : int, optional
            Number of samples, by default 1000
        tune : int, optional
            Number of tuning samples, by default 500
        chains : int, optional
            Number of chains, by default 4
        **kwargs : Any
            Options forwarded to pymc.sample, including cores.

        Returns
        -------
        xarray.DataTree
            Stored inference tree; posterior draws are standardized constrained
            parameter values. This mutates the regressor's stored trace.

        Notes
        -----
        The number of cores is set to the minimum between the number of cores and the
        number of chains.

        The sample method directly uses `pymc.sample`. See the PyMC
        documentation for more details.
        """
        cores = min(kwargs.pop("cores", cpu_count()), chains)
        with self.pymc_model(df):
            self._trace = pm.sample(
                draws=draws, tune=tune, chains=chains, cores=cores, **kwargs
            )

        return self.trace

    def _predictive(
        self, df: pd.DataFrame, parameters: xr.Dataset, random_seed: int | None,
    ) -> xr.DataArray:
        """Simulate unconditioned trajectories serially for standardized draws.

        parameters has chain/draw dimensions and one variable per free parameter.
        Returns (chain, draw, time, outputs); random_seed generates independent
        per-trajectory seeds. Uses a copied estimator, ignoring observed outputs.
        """
        draws = parameters[self.parameters.ids_free].to_array("parameter").transpose(
            "chain", "draw", "parameter"
        )
        data = self.prepare_data(df, with_outputs=False)[:3]
        estimator = deepcopy(self.estimator)
        rng = np.random.default_rng(random_seed)
        results = np.empty((*draws.shape[:2], len(df), self.ss.ny))
        for chain in range(draws.sizes["chain"]):
            for draw in range(draws.sizes["draw"]):
                # StateSpace's initialized parameters survive the estimator copy.
                cast(Parameters, estimator.ss.parameters).theta_sd = draws.values[chain, draw]
                seed = int(rng.integers(0, 2**31))
                results[chain, draw] = estimator.simulate(
                    *data, random_seed=seed
                ).y[..., 0]
        return xr.DataArray(
            results,
            dims=("chain", "draw", df.index.name or "time", "outputs"),
            coords={
                "chain": draws.chain,
                "draw": draws.draw,
                df.index.name or "time": df.index,
                "outputs": self.outputs,
            },
        )


    def prior_predictive(
        self, df: pd.DataFrame, samples: int = 1000, *,
        random_seed: int | None = None, **kwargs: Any,
    ) -> xr.Dataset:
        """Sample from the prior predictive distribution

        Parameters
        ----------
        df : pandas.DataFrame
            Dataframe defining observation times and exogenous inputs.
        samples : int, optional
            Number of samples, by default 1000
        random_seed : int, optional
            Seed for prior parameters and independent noisy trajectories.
        **kwargs : Any
            Options forwarded to pymc.sample_prior_predictive.

        Returns
        -------
        xarray.Dataset
            One variable per output with (draw, time) dimensions. Trajectories
            are simulated serially without conditioning on observations;
            standardized parameter draws are retained in the prior trace.

        """
        with self.pymc_model(df), warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self._prior_trace = pm.sample_prior_predictive(
                draws=samples, random_seed=random_seed, **kwargs
            )
        prediction = self._predictive(
            df, self._prior_trace["prior"].to_dataset(), random_seed
        )
        return prediction.isel(chain=0, drop=True).to_dataset("outputs")

    def posterior_predictive(
        self, df: pd.DataFrame, *, random_seed: int | None = None,
    ) -> xr.Dataset:
        """Sample from the posterior predictive distribution

        Parameters
        ----------
        df : pandas.DataFrame
            Time-indexed exogenous inputs; observations are ignored.
        random_seed : int, optional
            Seed used to derive independent trajectory seeds.

        Returns
        -------
        xarray.Dataset
            One variable per output with (chain, draw, time) dimensions.
            Simulates serially from standardized posterior draws without
            conditioning on observed outputs.

        Raises
        ------
        RuntimeError
            If the model has not been sampled yet
        """
        try:
            trace = self.trace
        except AttributeError:
            raise RuntimeError("No trace available : run `sample` first")
        prediction = self._predictive(
            df, trace["posterior"].to_dataset(), random_seed
        )
        return prediction.to_dataset("outputs")

    @property
    def trace(self) -> xr.DataTree:
        """Return stored inference data, merging missing prior groups in place.

        Raises AttributeError when neither posterior nor prior data is stored.
        """
        main_trace = getattr(self, "_trace", None)
        prior_trace = getattr(self, "_prior_trace", None)

        if main_trace is not None:
            if prior_trace is not None:
                main_trace.update(
                    {
                        name: group
                        for name, group in prior_trace.children.items()
                        if name not in main_trace.children
                    }
                )
            return main_trace
        if prior_trace is not None:
            return prior_trace
        raise AttributeError("No trace available")
