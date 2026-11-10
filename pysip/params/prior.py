"""Standardized constrained-coordinate priors backed by SciPy and PyMC."""

from collections.abc import Callable
from inspect import Parameter, Signature, get_annotations
from typing import Any, ClassVar, cast, dataclass_transform, overload

import numpy as np
import pymc as pm
from makefun import with_signature
from numpy.typing import NDArray
from pytensor.tensor.variable import TensorVariable
from scipy import stats
from scipy.stats._distn_infrastructure import rv_continuous_frozen

__ALL__ = ["Normal", "Gamma", "Beta", "InverseGamma", "LogNormal"]


class BasePrior:
    """Prior class template

    This contains both the scipy distribution (to compute standard numerical operations)
    as well as the pymc distribution factory (to be used with the bayesian inference
    engine).
    """

    @property
    def scipy_dist(self) -> rv_continuous_frozen:
        """Frozen SciPy distribution for standardized constrained values."""
        raise NotImplementedError

    @property
    def pymc_dist(self) -> Callable[[str], TensorVariable]:
        """Factory accepting a variable name and creating a PyMC random variable."""
        raise NotImplementedError

    @property
    def shape_parameters(self) -> tuple[float, ...]:
        """Distribution constructor arguments, in declaration order."""
        raise NotImplementedError

    def __eq__(self, __value: object, /) -> bool:
        """Compare distribution argument tuples without comparing family identity."""
        return self.shape_parameters == cast(BasePrior, __value).shape_parameters

    @property
    def mean(self) -> float:
        """Expected value in standardized constrained coordinates."""
        return self.scipy_dist.mean()

    @overload
    def pdf(self, x: float) -> float:
        """Evaluate density at a scalar standardized-coordinate quantile."""

    @overload
    def pdf(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluate density at an array of standardized-coordinate quantiles."""

    def pdf(self, x: float | NDArray[np.float64]) -> float | NDArray[np.float64]:
        """Evaluate standardized-coordinate density at scalar or array quantiles."""
        return self.scipy_dist.pdf(x)

    @overload
    def logpdf(self, x: float) -> float:
        """Evaluate log density at a scalar standardized-coordinate quantile."""

    @overload
    def logpdf(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluate log density at an array of standardized-coordinate quantiles."""

    def logpdf(self, x: float | NDArray[np.float64]) -> float | NDArray[np.float64]:
        """Evaluate standardized-coordinate log density at scalar or array quantiles."""
        return self.scipy_dist.logpdf(x)

    def random(self, n: int = 1, hpd: float | None = None) -> NDArray[np.float64]:
        """Draw an array of shape (n,) in standardized constrained coordinates.

        ``hpd`` retains central equal-tail probability mass, not necessarily a
        highest-density interval for asymmetric distributions. None or 1 uses
        the full distribution. Raises TypeError for nonpositive/noninteger n
        and ValueError for probability mass outside (0, 1].
        """
        if not isinstance(n, int) or n <= 0:
            raise TypeError("`n` must an integer greater or equal to 1")

        if hpd is not None and (hpd <= 0.0 or hpd > 1.0):
            raise ValueError("`hpd must be between ]0, 1]")

        if hpd is None or hpd == 1.0:
            return self.scipy_dist.rvs(n)

        low = (1.0 - hpd) / 2.0
        return self.scipy_dist.ppf(np.random.uniform(low=low, high=1.0 - low, size=n))


@dataclass_transform(eq_default=False)
class PriorMeta(type):
    """Generate typed constructors from numeric fields and named distribution factories."""

    def __new__(
        cls, name: str, bases: tuple[type, ...], attrs: dict[str, Any]
    ) -> "PriorMeta":
        """Generate a constructor from numeric annotations and named factories."""
        new_class: Any = super().__new__(cls, name, bases, attrs)
        annots = get_annotations(new_class)
        annots.pop("_scipy_factory")
        annots.pop("_pymc_factory")
        scipy_dist = cast(Callable[..., rv_continuous_frozen], attrs["_scipy_factory"])
        pymc_dist = cast(Callable[..., TensorVariable], attrs["_pymc_factory"])
        dist_args = annots
        defaults = {
            k: default
            for k in dist_args
            if (default := attrs.get(k, False)) is not False
        }

        def make_scipy_dist(self) -> rv_continuous_frozen:
            """Freeze the SciPy distribution with current constructor arguments."""
            return scipy_dist(*[getattr(self, k) for k in dist_args])

        def make_pymc_dist_factory(self) -> Callable[[str], TensorVariable]:
            """Return a named-variable factory using current prior arguments."""
            def factory(name: str) -> TensorVariable:
                """Create a named PyMC variable using the current prior arguments."""
                return pymc_dist(name, *[getattr(self, k) for k in dist_args])

            return factory

        sign = Signature(
            parameters=[
                Parameter("self", Parameter.POSITIONAL_OR_KEYWORD),
                *[
                    Parameter(
                        k,
                        Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=annot,
                        default=defaults.get(k, Parameter.empty),
                    )
                    for k, annot in dist_args.items()
                ],
            ],
            return_annotation=None,
        )

        @with_signature(sign)
        def __init__(self, **kwargs: float) -> None:
            """Store the distribution arguments exposed by the generated signature."""
            for k, v in kwargs.items():
                setattr(self, k, v)

        def shape_parameters(self) -> tuple[float, ...]:
            """Return distribution arguments in constructor declaration order."""
            return tuple(getattr(self, arg) for arg in dist_args)

        new_class.__init__ = __init__
        new_class.shape_parameters = property(shape_parameters)
        new_class.scipy_dist = property(make_scipy_dist)
        new_class.pymc_dist = property(make_pymc_dist_factory)
        return new_class

    def params_repr(self) -> str:
        """Format the values named by the prior class's params attribute."""
        return ", ".join(
            f"{param}={getattr(self, param):.3e}" for param in cast(Any, self).params
        )


class Normal(BasePrior, metaclass=PriorMeta):
    """Normal prior distribution

    Parameters
    ----------
    mu : float
        Mean of the normal distribution
    sigma : float
        Standard deviation of the normal distribution
    """

    mu: float = 0.0
    sigma: float = 1.0
    _scipy_factory: ClassVar[Callable[..., rv_continuous_frozen]] = cast(
        Callable[..., rv_continuous_frozen], stats.norm
    )
    _pymc_factory: ClassVar[Callable[..., TensorVariable]] = pm.Normal


class Gamma(BasePrior, metaclass=PriorMeta):
    """Gamma prior distribution

    Parameters
    ----------
    alpha : float
        Shape parameter of the gamma distribution
    beta : float
        Rate parameter of the gamma distribution
    """

    alpha: float = 3.0
    beta: float = 1.0
    _scipy_factory: ClassVar[Callable[..., rv_continuous_frozen]] = (
        lambda a, b: cast(rv_continuous_frozen, stats.gamma(a=a, scale=1.0 / b))
    )
    _pymc_factory: ClassVar[Callable[..., TensorVariable]] = pm.Gamma


class Beta(BasePrior, metaclass=PriorMeta):
    """Beta prior distribution

    Parameters
    ----------
    alpha : float
        Shape parameter of the beta distribution
    beta : float
        Shape parameter of the beta distribution
    """

    alpha: float = 3.0
    beta: float = 3.0
    _scipy_factory: ClassVar[Callable[..., rv_continuous_frozen]] = cast(
        Callable[..., rv_continuous_frozen], stats.beta
    )
    _pymc_factory: ClassVar[Callable[..., TensorVariable]] = pm.Beta


class InverseGamma(BasePrior, metaclass=PriorMeta):
    """Inverse Gamma prior distribution

    Parameters
    ----------
    alpha : float
        Shape parameter of the inverse gamma distribution
    beta : float
        Scale parameter of the inverse gamma distribution
    """

    alpha: float = 3.0
    beta: float = 1.0
    _scipy_factory: ClassVar[Callable[..., rv_continuous_frozen]] = (
        lambda a, b: cast(rv_continuous_frozen, stats.invgamma(a=a, scale=b))
    )
    _pymc_factory: ClassVar[Callable[..., TensorVariable]] = pm.InverseGamma


class LogNormal(BasePrior, metaclass=PriorMeta):
    """Log Normal prior distribution

    Parameters
    ----------
    mu : float
        Mean of the underlying normal logarithm.
    sigma : float
        Standard deviation of the underlying normal logarithm.
    """

    mu: float = 0.0
    sigma: float = 1.0
    _scipy_factory: ClassVar[Callable[..., rv_continuous_frozen]] = (
        lambda mu, sigma: cast(rv_continuous_frozen, stats.lognorm(scale=np.exp(mu), s=sigma))
    )
    _pymc_factory: ClassVar[Callable[..., TensorVariable]] = pm.Lognormal


class Uniform(BasePrior, metaclass=PriorMeta):
    """Uniform prior distribution

    Parameters
    ----------
    lower : float
        Lower bound of the uniform distribution
    upper : float
        Upper bound of the uniform distribution
    """

    lower: float = 0.0
    upper: float = 1.0
    _scipy_factory: ClassVar[Callable[..., rv_continuous_frozen]] = (
        lambda lower, upper: cast(rv_continuous_frozen, stats.uniform(loc=lower, scale=upper - lower))
    )
    _pymc_factory: ClassVar[Callable[..., TensorVariable]] = pm.Uniform
