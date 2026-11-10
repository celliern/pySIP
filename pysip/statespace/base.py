"""Continuous state-space models, metadata registry, and data preparation."""

from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import field
from enum import Enum
from itertools import chain
from typing import (
    Annotated,
    Any,
    Literal,
    NamedTuple,
    cast,
)

import numpy as np
import pandas as pd
from jinja2 import Template
from pydantic import ConfigDict, Field, field_validator
from pydantic.dataclasses import dataclass

from ..params import Parameters
from ..utils.math import nearest_cholesky
from ..utils.misc import Namespace
from . import discretization
from ._functional import (
    ContinuousMatrices,
    evaluate_matrices,
)
from ._functional import (
    discrete_covariance as _discrete_covariance,
)

model_registry = Namespace()


class Io(Enum):
    """Input/Output data type"""

    TEMPERATURE = ("Temperature", "°C")
    POWER = ("Power", "W")
    ANY = ("Any type", "any")


class Par(Enum):
    """Parameter type"""

    THERMAL_RESISTANCE = ("Thermal resistance", "°C/W")
    THERMAL_TRANSMITTANCE = ("Thermal transmittance", "W/°C")
    THERMAL_CAPACITY = ("Thermal capacity", "J/°C")
    SOLAR_APERTURE = ("Solar aperture", "m²")
    STATE_DEVIATION = ("State deviation", "any")
    MEASURE_DEVIATION = ("Measure deviation", "any")
    INITIAL_DEVIATION = ("Initial deviation", "any")
    INITIAL_MEAN = ("Initial mean", "any")
    COEFFICIENT = ("Coefficient", "any")
    MAGNITUDE_SCALE = ("Magnitude scale", "any")
    LENGTH_SCALE = ("Length scale", "any")
    PERIOD = ("Period", "any")


@dataclass
class Node:
    """Description of model variables

    Parameters
    ----------
    category : Io or Par
        Category of the node
    name : str
        Name of the node
    description : str
        Description of the node
    """

    category: Io | Par = field()
    name: str = field(default="")
    description: str = field(default="")

    @field_validator("category", mode="before")
    @classmethod
    def _validate_category(cls, category: Io | Par | str) -> Io | Par:
        """Resolve category names to enums; reject unknown names."""
        if isinstance(category, str):
            try:
                return Io[category]
            except KeyError:
                try:
                    return Par[category]
                except KeyError as exc:
                    raise ValueError(f"Unknown node category: {category}") from exc
        return category

    def unpack(self) -> tuple[str, str, str]:
        """Return category name, variable name, and description."""
        return self.category.name, self.name, self.description


def statespace(cls: str, *args: Any, **kwargs: Any) -> "type[StateSpace] | StateSpace | None":
    """Look up a registered name, instantiating it only if arguments are supplied.

    Unknown names return None. Constructor arguments are forwarded unchanged.
    """
    if cls not in model_registry:
        return None
    if args or kwargs:
        return model_registry[cls](*args, **kwargs)
    return model_registry[cls]


class MetaStateSpace(type):
    """Metaclass for the state-space models

    Mainly used to build the documentation of the models from the sections
    (inputs, outputs, states, params) available as class attributes. Will also add
    the class to the registry.
    """

    def __new__(
        cls, name: str, bases: tuple[type, ...], attrs: dict[str, Any]
    ) -> "MetaStateSpace":
        """Create a model class and register its declared or qualified name."""
        # Add the class to the registry
        new_class = super().__new__(cls, name, bases, attrs)
        if "__base__" not in attrs:
            model_registry[attrs.get("__name__") or attrs["__qualname__"]] = new_class
        return new_class

    def _format_section(self, section_name: str) -> str | None:
        """Render a metadata section when nodes are present."""
        if not (nodes := getattr(self, section_name, None)):
            return None
        nodes = [Node(*x) for x in nodes]
        return cast(Template, cast(Any, self)._section_template).render(
            section_name=section_name, nodes=nodes
        )

    @property
    def _sections(self) -> dict[str, list[Node]]:
        """Return normalized input, output, and state metadata."""
        return {
            section: [Node(*x) for x in getattr(self, section, [])]
            for section in ["inputs", "outputs", "states"]
        }

    @property
    def _parameters_cat(self) -> dict[Io | Par, list[Node]]:
        """Group parameter metadata by physical category."""
        parameters_cat = defaultdict(list)
        for par in getattr(self, "params", []):
            par = Node(*par)
            parameters_cat[par.category].append(par)
        return dict(parameters_cat)

    def _build_doc(self) -> str:
        """Append variable and parameter metadata to the model description."""
        _doc_template_str = """{{base_doc}}

Model Variables
---------------

{% for section, nodes in sections.items() %}
- {{section.capitalize()}}
{% for node in nodes %}
    - ``{{node.name}}``: {{node.description}} `({{node.category.value[1]}})`
{% endfor %}
{% endfor %}

{% if parameters_cat %}

Model Parameters
----------------

{% for category, parameters in parameters_cat.items() %}
- {{category.value[0]}}
{% for parameter in parameters %}
    - ``{{parameter.name}}``: {{parameter.description}}
      `({{parameter.category.value[1]}})`
{% endfor %}
{% endfor %}
{% endif %}
    """
        _doc_template = Template(
            _doc_template_str,
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=False,
        )

        return _doc_template.render(
            base_doc=self.__doc__,
            sections=self._sections,
            parameters_cat=self._parameters_cat,
        )

    def __init__(self, name: str, bases: tuple[type, ...], attr: dict[str, Any]) -> None:
        """Populate class documentation after model creation."""
        self.__doc__ = self._build_doc()


def prepare_data(
    df: pd.DataFrame,
    inputs: str | Sequence[str],
    outputs: str | Sequence[str] | Literal[False],
    time_scale: str = "s",
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Prepare copied data as forward intervals, inputs, derivatives, and outputs.

    Parameters
    ----------
    df : pandas.DataFrame
        Source with at least two rows. Its index supplies sample times.
    inputs : str or sequence of str
        Input column names.
    outputs : str, sequence of str, or False
        Output columns; False or an empty sequence produces one NaN column.
    time_scale : str, optional
        Unit for datetime-index intervals; numeric indexes retain their units.

    Returns
    -------
    dt : pandas.Series
        Forward intervals, shape (n,); the final interval repeats its predecessor.
    u : pandas.DataFrame
        Inputs, shape (n, nu).
    dtu : pandas.DataFrame
        Forward input derivatives, shape (n, nu); the final row is zero.
    y : pandas.DataFrame
        Outputs, shape (n, ny); NaN denotes a missing observation.

    Raises
    ------
    KeyError
        A requested column is absent.
    ValueError
        Time intervals, inputs, or derivatives contain nonfinite values.
    """
    df = df.copy()
    time = df.index.to_series()
    if not isinstance(df, pd.DataFrame):
        raise TypeError("`df` must be a dataframe")
    time_unit = cast(Callable[[int, str], pd.Timedelta], pd.to_timedelta)(1, time_scale)

    # diff and forward-fill the nan-last value
    dt = time.diff().shift(-1)
    dt.iloc[-1] = dt.iloc[-2]
    # deal with numerical approximation that lead to almost equal values
    if np.isclose(dt.to_numpy(), dt.iloc[0]).all():
        dt[:] = dt.iloc[0]
    if isinstance(df.index, pd.DatetimeIndex):
        dt = dt / time_unit
    else:
        dt = dt.astype(float)

    u = pd.DataFrame(df[inputs])

    dtu = u.diff().shift(-1) / dt.to_numpy()[:, None]
    dtu.iloc[-1, :] = 0

    for subdf in [dt, u, dtu]:
        if not np.all(np.isfinite(subdf)):
            raise ValueError(f"{subdf} contains undefinite values")

    if outputs:
        y = pd.DataFrame(df[outputs])
    else:
        y = pd.DataFrame(data=np.full(len(df), np.nan), index=df.index)
    if not np.all(np.isnan(y[~np.isfinite(y)])):
        raise TypeError("The output vector must contains numerical values or numpy.nan")
    return cast(pd.Series, dt), u, cast(pd.DataFrame, dtu), y


class ContinuousStates(NamedTuple):
    """Continuous A(nx,nx), B(nx,nu), C(ny,nx), D(ny,nu), Q(nx,nx), R(ny,ny).

    Q and R are upper noise factors: their covariances are U.T @ U.
    """

    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray
    Q: np.ndarray
    R: np.ndarray


class DiscreteStates(NamedTuple):
    """Discrete A(nx,nx), B0/B1(nx,nu), and upper process-noise factor Q(nx,nx).

    B1 multiplies the input derivative; Q.T @ Q is the process covariance.
    """

    A: np.ndarray
    B0: np.ndarray
    B1: np.ndarray
    Q: np.ndarray


@dataclass(config=ConfigDict(arbitrary_types_allowed=True), repr=False)
class StateSpace(metaclass=MetaStateSpace):
    """Linear Gaussian continuous-time model with mutable system matrices.

    A, B, C, D have shapes (nx,nx), (nx,nu), (ny,nx), (ny,nu).
    Q, R and P0 are upper factors of process diffusion, measurement noise,
    and initial-state covariance: each covariance is U.T @ U.
    x0 has shape (nx,1). Updates consume physical (constrained) parameters.
    """

    parameters: Parameters | None = field(default=None, repr=False)
    hold_order: Annotated[int, Field(ge=0, le=1)] = 0
    method: str = "mfd"
    name: str = ""

    @field_validator("parameters", mode="before")
    @classmethod
    def _validate_parameters(cls, parameters: Any) -> Parameters | None:
        """Normalize dynamic constructor specifications to Parameters."""
        if not parameters:
            return None
        if not isinstance(parameters, Parameters):
            return Parameters(parameters)
        return parameters

    def _coerce_attributes(self) -> None:
        """Normalize tuple metadata to Nodes and initialize Parameters.

        Metadata declarations remain outside dataclass fields. Their union
        describes both phases; after this method, instance views contain Nodes
        and ``parameters`` is always a Parameters container.
        """
        for attr in ["states", "params", "inputs", "outputs"]:
            setattr(self, attr, [Node(*s) for s in getattr(self, attr)])
        self.states: list[Node] | list[tuple[str, str, str]] | tuple[tuple[str, str, str], ...]
        self.params: list[Node] | list[tuple[str, str, str]] | tuple[tuple[str, str, str], ...]
        self.inputs: list[Node] | list[tuple[str, str, str]] | tuple[tuple[str, str, str], ...]
        self.outputs: list[Node] | list[tuple[str, str, str]] | tuple[tuple[str, str, str], ...]

        if self.parameters:
            if not isinstance(self.parameters, Parameters):
                self.parameters = Parameters(self.parameters)
        else:
            self.parameters = Parameters([p.name for p in cast(list[Node], self.params)])
        self.parameters.name = self.name

    def _init_states(self) -> None:
        """Allocate continuous matrices, upper factors, and initial moments."""
        self.A = np.zeros((self.nx, self.nx))
        self.B = np.zeros((self.nx, self.nu))
        self.C = np.zeros((self.ny, self.nx))
        self.D = np.zeros((self.ny, self.nu))
        self.Q = np.zeros((self.nx, self.nx))
        self.R = np.zeros((self.ny, self.ny))
        self.x0 = np.zeros((self.nx, 1))
        self.P0 = np.zeros((self.nx, self.nx))
        self._diag = np.diag_indices_from(self.A)
        self.set_constant_continuous_ssm()

    def __repr__(self) -> str:
        """Render model variables and current physical parameter values."""
        tpl = Template(
            """{{name}}
{{'-' * name|length}}
States:
{% for state in states %}
- {{state.name}}: {{state.description}}
{% endfor %}

Inputs:
{% for input in inputs %}
- {{input.name}}: {{input.description}}
{% endfor %}

Outputs:
{% for output in outputs %}
- {{output.name}}: {{output.description}}
{% endfor %}

Parameters:
{% for parameter in parameters %}
- {{parameter.name}}: {{parameter.value}}
{% endfor %}""",
            trim_blocks=True,
            lstrip_blocks=True,
        )
        return tpl.render(
            name=self.name,
            states=self.states,
            inputs=self.inputs,
            outputs=self.outputs,
            parameters=self.parameters,
        )

    def __post_init__(self) -> None:
        """Initialize the model name, metadata, parameters, and matrices."""
        if self.name == "":
            self.name = self.__class__.__name__
        self._coerce_attributes()
        self._init_states()

    @property
    def nx(self) -> int:
        """Number of latent states."""
        return len(self.states)

    @property
    def ny(self) -> int:
        """Number of observed outputs."""
        return len(self.outputs)

    @property
    def nu(self) -> int:
        """Number of deterministic inputs."""
        return len(self.inputs)

    def set_constant_continuous_ssm(self) -> None:
        """Mutate parameter-independent continuous matrices; base hook does nothing."""

    def update_continuous_ssm(self) -> None:
        """Mutate continuous matrices from physical parameters; base hook does nothing."""

    def update(self) -> None:
        """Mutate continuous matrices and initial moments from physical parameters."""
        self.update_continuous_ssm()

    def continuous_matrices(self, theta: Any, *, xp: Any = np) -> ContinuousMatrices:
        """Evaluate physical parameters without mutating this model.

        Parameters
        ----------
        theta : array_like
            All physical parameters, including fixed leaves, in flattened order.
        xp : module, optional
            NumPy or jax.numpy. Custom JAX models must use immutable array updates
            or override this method with a pure definition.

        Returns
        -------
        ContinuousMatrices
            A/B/C/D, full diffusion and measurement covariances, initial column
            mean, and full initial covariance. Covariances are not factors.
        """
        try:
            return evaluate_matrices(self, theta, xp)
        except (TypeError, ValueError) as error:
            if xp is np:
                raise
            raise TypeError(
                f"{type(self).__name__} cannot evaluate immutable model matrices; "
                "use array_set for indexed updates or implement continuous_matrices"
            ) from error

    def discrete_covariance(
        self, theta: Any, dt: Any, *, xp: Any = np,
    ) -> tuple[Any, Any, Any, Any]:
        """Return discrete A/B0/B1 and full process covariance without mutation.

        Parameters
        ----------
        theta : array_like
            All physical parameters in flattened order.
        dt : float
            Sampling interval in the model's time unit.
        xp : module, optional
            NumPy or jax.numpy.

        Returns
        -------
        tuple
            Transition, input-value and input-derivative matrices, and full
            process covariance. Custom discretization overrides must implement
            this pure counterpart explicitly.
        """
        return _discrete_covariance(
            self.continuous_matrices(theta, xp=xp), dt, self.hold_order, xp
        )

    def get_discrete_ssm(self, dt: float) -> DiscreteStates:
        """Update continuous matrices and return their discretization at ``dt``.

        ``dt`` uses the model time unit. Return a DiscreteStates named tuple:
        A(nx,nx), B0(nx,nu), B1(nx,nu), Q(nx,nx). B0 multiplies input
        values and B1 input derivatives (zero for zero-order hold).
        Q is an upper factor, with covariance Q.T @ Q. Updating mutates the
        continuous matrices and initial moments.
        """
        self.update_continuous_ssm()
        return DiscreteStates(*self.discretization(dt))

    def discretization(
        self, dt: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Discretize the current continuous matrices without updating parameters.

        ``dt`` uses the model time unit. Return the tuple (Ad, B0d, B1d, Qd)
        with shapes (nx,nx), (nx,nu), (nx,nu), (nx,nx).
        The augmented exponential supports singular A. B1d multiplies input
        derivatives and is zero for zero-order hold. Qd is an upper factor:
        the discrete process covariance is Qd.T @ Qd.
        """

        if self.nu == 0:
            Ad = discretization.state(self.A, dt)
            B0d = np.zeros((self.nx, self.nu))
            B1d = B0d
        else:
            Ad, B0d, B1d = discretization.state_input(
                self.A, self.B, dt, self.hold_order, "expm"
            )

        Qd = nearest_cholesky(
            discretization.diffusion_mfd(self.A, self.Q.T @ self.Q, dt)
        )

        # The default LDL factorization always returns an array.
        return Ad, B0d, B1d, cast(np.ndarray, Qd)

    def prepare_data(
        self,
        df: pd.DataFrame,
        inputs: str | Sequence[str] | None = None,
        outputs: str | Sequence[str] | Literal[False] | None = None,
        time_scale: str = "s",
    ) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Prepare data using model column names unless explicitly overridden.

        The source frame is copied. ``inputs`` and ``outputs`` accept a column
        name or sequence; None uses model metadata. False for ``outputs`` returns
        one NaN column, not an omitted return value. ``time_scale`` selects the
        interval unit for datetime indexes; numeric indexes retain their units.
        At least two rows are required.

        Returns
        -------
        dt : pandas.Series
            Forward intervals, shape (n,); the final interval is repeated.
        u : pandas.DataFrame
            Inputs, shape (n, nu).
        dtu : pandas.DataFrame
            Forward input derivatives, shape (n, nu), with a final zero row.
        y : pandas.DataFrame
            Outputs, shape (n, ny), allowing NaN missing observations.

        Raises
        ------
        KeyError
            A requested column is missing.
        ValueError
            Time intervals, inputs, or derivatives are nonfinite.
        """
        if inputs is None:
            inputs = [par.name for par in cast(list[Node], self.inputs)]
        if outputs is None:
            outputs = [par.name for par in cast(list[Node], self.outputs)]
        if isinstance(inputs, str):
            inputs = [inputs]
        if isinstance(outputs, str):
            outputs = [outputs]
        if outputs is False:
            outputs = []
        for key in chain(inputs, outputs):
            if key not in df.columns:
                raise KeyError(
                    f"Missing column {key}. Use `inputs` and `outputs` parameters to "
                    "specify the inputs and outputs if they diverge from the model "
                    "definition."
                )
        return prepare_data(df, inputs, outputs, time_scale=time_scale)


@dataclass(config=ConfigDict(arbitrary_types_allowed=True), repr=False)
class RCModel(StateSpace):
    """Dynamic thermal model"""

    latent_forces: str = ""

    def __le__(self, gp: "GPModel") -> StateSpace:
        """Compose copied models, replacing the named input with a GP latent force."""
        from .latent_force_model import LatentForceModel

        return LatentForceModel(self, gp, self.latent_forces)

    def discretization(
        self, dt: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (Ad, B0d, B1d, Qd) for the current thermal matrices at ``dt``.

        Shapes and upper-factor convention match StateSpace.discretization.
        ``method="analytic"`` uses linear solves and requires nonsingular A;
        other method values use the augmented exponential and MFD covariance.
        No parameter update is performed.
        """

        if self.method == "analytic":
            Ad, B0d, B1d = discretization.state_input(
                self.A, self.B, dt, self.hold_order, "analytic"
            )
            Qd = nearest_cholesky(
                discretization.diffusion_lyap(self.A, self.Q.T @ self.Q, Ad)
            )
        else:
            Ad, B0d, B1d = discretization.state_input(
                self.A, self.B, dt, self.hold_order, "expm"
            )
            Qd = nearest_cholesky(
                discretization.diffusion_mfd(self.A, self.Q.T @ self.Q, dt)
            )

        # The default LDL factorization always returns an array.
        return Ad, B0d, B1d, cast(np.ndarray, Qd)


class GPModel(StateSpace):
    """Gaussian Process"""

    def __post_init__(self) -> None:
        """Expand harmonic state blocks before initializing continuous matrices."""
        if hasattr(self, "J"):
            self.states = cast(Any, self).states_block * int(cast(Any, self).J + 1)
        super().__post_init__()

    def __mul__(self, gp: "GPModel") -> "GPModel":
        """Return a product-kernel model built from copies of both operands."""
        if not isinstance(gp, GPModel):
            raise TypeError("`gp` must be an GPModel instance")
        # TODO: refactor to avoid circular import
        from .gaussian_process import GPProduct

        return GPProduct(self, gp)

    def __add__(self, gp: "GPModel") -> "GPModel":
        """Return an independent sum-kernel model built from copied operands."""
        if not isinstance(gp, GPModel):
            raise TypeError("`gp` must be an GPModel instance")
        # TODO: refactor to avoid circular import
        from .gaussian_process import GPSum

        return GPSum(self, gp)

    def discretization(
        self, dt: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Discretize current GP matrices using the continuous diffusion integral.

        Return (Ad, B0d, B1d, Qd), with shapes and upper-factor convention as
        in StateSpace.discretization. Both input matrices are zero and share
        storage. Integrating diffusion avoids cancellation in Pinf-Ad Pinf Ad.T
        for long length scales. No parameter update is performed.
        """
        Ad = discretization.state(self.A, dt)
        B0d = np.zeros((self.nx, self.nu))
        Qd = nearest_cholesky(
            discretization.diffusion_mfd(self.A, self.Q.T @ self.Q, dt)
        )

        # The default LDL factorization always returns an array.
        return Ad, B0d, B0d, cast(np.ndarray, Qd)
