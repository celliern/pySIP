"""Scalar parameters in physical, standardized, and optimization coordinates."""

from typing import Annotated, Literal, cast

from pydantic import ConfigDict, Field, ValidationInfo, field_validator
from pydantic.dataclasses import dataclass

from pysip.params.prior import BasePrior

from .transforms import FixedTransform, ParameterTransform, Transforms, auto_transform


@dataclass(config=ConfigDict(arbitrary_types_allowed=True, validate_default=True))
class Parameter:
    """A scalar with physical θ = loc + scale * θ_sd and unconstrained η.

    Transforms and bounds operate on standardized constrained θ_sd, not physical
    θ. Assigning a coordinate property synchronizes the other coordinates.

    Parameters
    ----------
    name : str
        Parameter name
    value : float, optional
        Initial standardized constrained θ_sd. If omitted, initialize at η = 0.
    loc : float, default 0
        Physical location offset.
    scale : float, default 1
        Strictly positive physical scaling factor.
    transform : str or ParameterTransform, default "auto"
        Map θ_sd to η. Auto selects identity, log, shifted log, or logit from
        the bounds; "fixed" excludes the parameter from free vectors.
    bounds : tuple of float or None, default (None, None)
        Lower and upper standardized bounds; None denotes an unbounded side.
    prior : BasePrior
        Prior distribution on the standardized constrained value, not θ or η.

    Raises
    ------
    ValueError
        If bounds are reversed or equal, scale is nonpositive, or the initial
        standardized value fails the selected transform's bounds check.
    """

    name: str
    value: float | None = None
    loc: float = 0.0
    scale: Annotated[float, Field(gt=0)] = 1.0
    bounds: tuple[float | None, float | None] = (None, None)
    transform: ParameterTransform | Literal[
        "auto", "fixed", "none", "log", "lower", "upper", "logit"
    ] = "auto"
    prior: BasePrior | None = None

    @field_validator("bounds")
    @classmethod
    def _validate_bounds(
        cls, bounds: tuple[float | None, float | None]
    ) -> tuple[float | None, float | None]:
        """Reject finite bounds that do not define a nonempty interval."""
        lb, ub = bounds
        if lb is not None and ub is not None and lb >= ub:
            raise ValueError("`lb` must be lower than `ub`")
        return bounds

    @field_validator("transform")
    @classmethod
    def _validate_transform(
        cls,
        transform: ParameterTransform | Literal[
            "auto", "fixed", "none", "log", "lower", "upper", "logit"
        ],
        info: ValidationInfo,
    ) -> ParameterTransform | Literal[
        "auto", "fixed", "none", "log", "lower", "upper", "logit"
    ]:
        """Resolve a transform name using already validated standardized bounds."""
        if isinstance(transform, ParameterTransform):
            return transform
        bounds = info.data.get("bounds")
        if bounds is None:
            return transform
        if transform == "auto":
            return auto_transform(bounds)
        return Transforms.get(transform)(bounds)

    def __post_init__(self) -> None:
        """Initialize synchronized coordinates and check transform bounds."""
        if self.value is None:
            self.eta = 0.0
        else:
            self.theta = self.loc + self.scale * self.value
        if not self._transform.in_bounds(self.theta_sd):
            raise ValueError(
                f"Initial value {self.value} is out of bounds {self.bounds}"
            )

    @property
    def _transform(self) -> ParameterTransform:
        """Return the transform instance resolved by Pydantic before post-init."""
        return cast(ParameterTransform, self.transform)

    @property
    def theta(self) -> float:
        """Physical constrained θ; assignment also synchronizes θ_sd and η."""
        return self.loc + self.scale * self.theta_sd

    @theta.setter
    def theta(self, x: float) -> None:
        """Set physical θ and synchronize standardized and unconstrained values."""
        self.value = (x - self.loc) / self.scale
        self._eta = self._transform.transform(self.theta_sd)

    @property
    def theta_sd(self) -> float:
        """Standardized constrained value; assigning it also synchronizes η."""
        # Post-init initializes value through a coordinate setter when omitted.
        return cast(float, self.value)

    @theta_sd.setter
    def theta_sd(self, x: float) -> None:
        """Set standardized constrained θ_sd and synchronize η."""
        self.value = x
        self._eta = self._transform.transform(self.theta_sd)

    @property
    def eta(self) -> float:
        """Unconstrained η; assignment also synchronizes θ_sd and physical θ."""
        return self._eta

    @eta.setter
    def eta(self, x: float) -> None:
        """Set unconstrained η and synchronize constrained values."""
        self._eta = x
        self.value = self._transform.untransform(self._eta)

    @property
    def free(self) -> bool:
        """Whether this parameter participates in free parameter vectors."""
        return not isinstance(self.transform, FixedTransform)

    def get_transformed(self) -> float:
        """Apply the forward transform to the stored η (not to θ_sd)."""
        return self._transform.transform(self._eta)

    def get_transform_jacobian(self) -> float:
        """Return dη/dθ_sd at the current standardized value."""
        return self._transform.grad_transform(self.theta_sd)

    def get_inv_transformed(self) -> float:
        """Return θ_sd by applying the inverse transform to stored η."""
        return self._transform.untransform(self._eta)

    def get_inv_transform_jacobian(self) -> float:
        """Return dθ_sd/dη at the current unconstrained value."""
        return self._transform.grad_untransform(self._eta)

    def get_penalty(self) -> float:
        """Return the transform's bound penalty at current θ_sd."""
        return self._transform.penalty(self.theta_sd)

    def get_grad_penalty(self) -> float:
        """Return the derivative of the bound penalty with respect to θ_sd."""
        return self._transform.grad_penalty(self.theta_sd)
