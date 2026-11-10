"""Transforms between standardized constrained θ_sd and unconstrained η.

The scalar argument named θ in transform methods always denotes θ_sd, not the
physical value loc + scale * θ_sd. Bounds likewise use standardized coordinates.
Transforms expose forward/inverse derivatives and boundary penalties. Explicit
transform names do not validate compatibility with supplied bounds.
"""

from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, ClassVar, TypeVar, cast

import numpy as np

TransformType = TypeVar("TransformType", bound="ParameterTransform")


class Transforms:
    """Registry of transform classes, indexed by their public names."""
    @classmethod
    def register(
        cls, transform: type["ParameterTransform"], name: str | None = None
    ) -> None:
        """Register a class; raise ValueError if its name already exists."""
        if name is None:
            name = cast(str, getattr(transform, "name", transform.__qualname__))
        if hasattr(cls, name):
            raise ValueError("Transform with that name already exists")
        setattr(cls, name, transform)

    @classmethod
    def get(cls, name: str) -> type["ParameterTransform"]:
        """Look up a transform class; unknown names raise AttributeError."""
        return cast(type["ParameterTransform"], getattr(cls, name))


# decorator to register transforms
def register_transform(cls: type[TransformType]) -> type[TransformType]:
    """Register and return a transform class without wrapping it."""
    Transforms.register(cls)
    return cls


class ParameterTransform(metaclass=ABCMeta):
    """Contract for scalar standardized-coordinate bijections and penalties."""

    lb: float | None
    ub: float | None

    def __init__(self, bounds: tuple[float | None, float | None]) -> None:
        """Store standardized lower/upper bounds; None denotes an unbounded side."""
        self.lb, self.ub = bounds

    if TYPE_CHECKING:
        # Concrete transforms expose names as class attributes for registration.
        name: ClassVar[str]
    else:
        @property
        @abstractmethod
        def name(self) -> str:
            """Registry name identifying this transform."""

    @abstractmethod
    def transform(self, θ: float) -> float:
        """Transform standardized constrained θ_sd to unconstrained η.

        ``θ`` is the scalar standardized constrained value θ_sd.

        Returns
        -------
        η : float
            Parameter value in the unconstrained space η
        """

    @abstractmethod
    def untransform(self, η: float) -> float:
        """Transform unconstrained η to standardized constrained θ_sd.

        ``η`` is the scalar unconstrained parameter value.

        Returns
        -------
        θ : float
            Standardized constrained parameter value θ_sd.
        """

    @abstractmethod
    def grad_transform(self, θ: float) -> float:
        """Gradient of the transform function

        ``θ`` is the scalar standardized constrained value θ_sd.

        Returns
        -------
        grad : float
            Gradient of the transform function
        """

    @abstractmethod
    def grad_untransform(self, η: float) -> float:
        """Gradient of the untransform function

        ``η`` is the scalar unconstrained parameter value.

        Returns
        -------
        grad : float
            Gradient of the untransform function
        """

    @abstractmethod
    def penalty(self, θ: float) -> float:
        """Penalty for the parameter value θ

        ``θ`` is the scalar standardized constrained value θ_sd.

        Returns
        -------
        penalty : float
            Penalty for the parameter value θ
        """

    @abstractmethod
    def grad_penalty(self, θ: float) -> float:
        """Gradient of the penalty function for the parameter value θ

        ``θ`` is the scalar standardized constrained value θ_sd.

        Returns
        -------
        grad_penalty : float
            Gradient of the penalty function for the parameter value θ
        """

    def in_bounds(self, x: float) -> bool:
        """Check if the parameter value is in the bounds of the transform

        Parameters
        ----------
        x : float
            Standardized constrained value θ_sd. The base implementation accepts all.

        Returns
        -------
        in_bounds : bool
            True if the parameter value is in the bounds of the transform
        """
        return True

    def __repr__(self) -> str:
        """Represent a transform by its registered name."""
        return f"{self.name}"

    def __eq__(self, __value: object, /) -> bool:
        """Compare registered names, ignoring bound values."""
        if isinstance(__value, ParameterTransform):
            return self.name == __value.name
        return False


@register_transform
class NoneTransform(ParameterTransform):
    """Identity transform: standardized θ_sd equals η; bounds are not checked."""

    name = "none"

    def transform(self, θ: float) -> float:
        """Return θ_sd unchanged."""
        return θ

    def untransform(self, η: float) -> float:
        """Return η unchanged as θ_sd."""
        return η

    def grad_transform(self, θ: float) -> float:
        """Return the identity forward derivative."""
        return 1.0

    def grad_untransform(self, η: float) -> float:
        """Return the identity inverse derivative."""
        return 1.0

    def penalty(self, θ: float) -> float:
        """Return zero penalty regardless of supplied bounds."""
        return 0.0

    def grad_penalty(self, θ: float) -> float:
        """Return zero penalty derivative."""
        return 0.0


@register_transform
class FixedTransform(NoneTransform):
    """Identity transform marking a parameter as excluded from free vectors."""

    name = "fixed"


@register_transform
class LogTransform(ParameterTransform):
    """Log transform θ_sd = exp(η); positivity is implicit in the mapping."""

    name = "log"

    def transform(self, θ: float) -> float:
        """Return log(θ_sd)."""
        return np.log(θ)

    def untransform(self, η: float) -> float:
        """Return exp(η) in standardized coordinates."""
        return np.exp(η)

    def grad_transform(self, θ: float) -> float:
        """Return dη/dθ_sd = 1/θ_sd."""
        return 1.0 / θ

    def grad_untransform(self, η: float) -> float:
        """Return dθ_sd/dη = exp(η)."""
        return np.exp(η)

    def penalty(self, θ: float) -> float:
        """Return the logarithmic transform's near-zero boundary penalty."""
        return 1e-12 / (θ - 1e-12)

    def grad_penalty(self, θ: float) -> float:
        """Return the penalty derivative with respect to θ_sd."""
        return -1e-12 / (θ - 1e-12) ** 2


@register_transform
class LowerTransform(ParameterTransform):
    """Lower-bound transform θ_sd = exp(η) + lb; requires a finite lower bound."""

    name = "lower"

    def transform(self, θ: float) -> float:
        """Return log(θ_sd - lb)."""
        return np.log(θ - cast(float, self.lb))

    def untransform(self, η: float) -> float:
        """Return exp(η) + lb in standardized coordinates."""
        return np.exp(η) + cast(float, self.lb)

    def grad_transform(self, θ: float) -> float:
        """Return dη/dθ_sd = 1/(θ_sd - lb)."""
        return 1.0 / (θ - cast(float, self.lb))

    def grad_untransform(self, η: float) -> float:
        """Return dθ_sd/dη = exp(η)."""
        return np.exp(η)

    def penalty(self: ParameterTransform, θ: float) -> float:
        """Return the reciprocal-distance penalty above a finite lb."""
        lb = cast(float, self.lb)
        return abs(lb) / (θ - lb)

    def grad_penalty(self: ParameterTransform, θ: float) -> float:
        """Return the penalty derivative for a finite standardized lb."""
        lb = cast(float, self.lb)
        return -abs(lb) / (θ - lb) ** 2

    def in_bounds(self, x: float) -> bool:
        """Check strict standardized lower-bound membership."""
        return x > cast(float, self.lb)


@register_transform
class UpperTransform(ParameterTransform):
    """Upper-bound transform θ_sd = ub - exp(η); requires a finite upper bound."""

    name = "upper"

    def transform(self, θ: float) -> float:
        """Return log(ub - θ_sd)."""
        return np.log(cast(float, self.ub) - θ)

    def untransform(self, η: float) -> float:
        """Return ub - exp(η) in standardized coordinates."""
        return cast(float, self.ub) - np.exp(η)

    def grad_transform(self, θ: float) -> float:
        """Return dη/dθ_sd = -1/(ub - θ_sd)."""
        return -1.0 / (cast(float, self.ub) - θ)  # TODO: check

    def grad_untransform(self, η: float) -> float:
        """Return dθ_sd/dη = -exp(η)."""
        return -np.exp(η)

    def penalty(self: ParameterTransform, θ: float) -> float:
        """Return the reciprocal-distance penalty below a finite ub."""
        ub = cast(float, self.ub)
        return abs(ub) / (ub - θ)

    def grad_penalty(self: ParameterTransform, θ: float) -> float:
        """Return the penalty derivative for a finite standardized ub."""
        ub = cast(float, self.ub)
        return abs(ub) / (ub - θ) ** 2

    def in_bounds(self, x: float) -> bool:
        """Check strict standardized upper-bound membership."""
        return x < cast(float, self.ub)


@register_transform
class LogitTransform(ParameterTransform):
    """Logit mapping from finite open standardized bounds (lb, ub) to η."""

    name = "logit"

    def transform(self, θ: float) -> float:
        """Return log((θ_sd - lb)/(ub - θ_sd))."""
        return np.log((θ - cast(float, self.lb)) / (cast(float, self.ub) - θ))

    def untransform(self, η: float) -> float:
        """Return the bounded logistic inverse at η."""
        lb, ub = cast(float, self.lb), cast(float, self.ub)
        return lb + (ub - lb) / (1 + np.exp(-η))

    def grad_transform(self, θ: float) -> float:
        """Return dη/dθ_sd for the bounded logit."""
        lb, ub = cast(float, self.lb), cast(float, self.ub)
        return (ub - lb) / ((θ - lb) * (ub - θ))

    def grad_untransform(self, η: float) -> float:
        """Return dθ_sd/dη for the bounded logistic inverse."""
        x = np.exp(-η)
        lb, ub = cast(float, self.lb), cast(float, self.ub)
        return (ub - lb) * x / (1 + x) ** 2

    def penalty(self, θ: float) -> float:
        """Sum lower- and upper-bound penalties at θ_sd."""
        return UpperTransform.penalty(self, θ) + LowerTransform.penalty(self, θ)

    def grad_penalty(self, θ: float) -> float:
        """Sum bound-penalty derivatives with respect to θ_sd."""
        return UpperTransform.grad_penalty(self, θ) + LowerTransform.grad_penalty(
            self, θ
        )

    def in_bounds(self, x: float) -> bool:
        """Check membership in the open standardized interval (lb, ub)."""
        return cast(float, self.lb) < x < cast(float, self.ub)


def auto_transform(bounds: tuple[float | None, float | None]) -> ParameterTransform:
    """Automatically select a transform based on the bounds

    Parameters
    ----------
    bounds : tuple
        Standardized lower and upper bounds, of any sign; either may be None.

    Returns
    -------
    transform : ParameterTransform
        Identity for no bounds, log for (0, None), shifted lower/upper log
        for one bound, and logit for two bounds. Does not validate bound ordering.
    """
    lb, ub = bounds
    if lb is None and ub is None:
        return Transforms.get("none")(bounds)
    if lb == 0.0 and ub is None:
        return Transforms.get("log")(bounds)
    if ub is None and lb is not None:
        return Transforms.get("lower")(bounds)
    if lb is None and ub is not None:
        return Transforms.get("upper")(bounds)
    if lb is not None and ub is not None:
        return Transforms.get("logit")(bounds)
    raise ValueError(f"No transform found for bounds {bounds}")
