"""Hierarchical parameter collections with flattened coordinate-vector views."""

from collections.abc import Iterator, Sequence
from typing import Any, Literal, TypeAlias, Union, cast

import numpy as np
from flatten_dict import flatten
from numpy.typing import NDArray

from .parameter import Parameter

ParameterTree: TypeAlias = dict[str, Union[Parameter, "ParameterTree"]]
ParameterInput: TypeAlias = (
    Sequence[str] | Sequence[Parameter] | Sequence[dict[str, Any]] | ParameterTree
)


def _coerce_params(parameters: ParameterInput) -> ParameterTree:
    """Normalize a homogeneous nonempty sequence, or retain a tree by reference."""
    if isinstance(parameters, dict):
        return parameters
    if isinstance(parameters[0], str):
        return {name: Parameter(name) for name in cast(Sequence[str], parameters)}
    if isinstance(parameters[0], Parameter):
        return {p.name: p for p in cast(Sequence[Parameter], parameters)}
    return {
        k["name"]: Parameter(**k)
        for k in cast(Sequence[dict[str, Any]], parameters)
    }


class Parameters:
    """A named parameter tree with depth-first, insertion-ordered vector views.

    Parameters
    ----------
    parameters : sequence or dict
        A nonempty sequence of names, Parameter instances, or keyword dictionaries
        accepted by Parameter; alternatively a recursive dictionary whose leaves
        are Parameter instances. Dictionaries and Parameter leaves are shared,
        not copied. Iteration yields immediate child leaves or subtrees, whereas
        vector properties flatten all leaves.
    name : str, default ""
        Collection name used when composing trees with addition.

    Notes
    -----
    θ denotes physical values, θ_sd standardized constrained values, and η
    unconstrained optimization values. Bounds and priors apply to θ_sd.
    Array properties return new one-dimensional arrays; list properties return
    new lists, but their Parameter objects remain shared. Coordinate setters
    mutate leaves and require the corresponding full or free vector length,
    raising ValueError on a mismatch. Free vectors omit fixed parameters.
    """

    def __init__(self, parameters: ParameterInput, name: str = "") -> None:
        """Store the collection name and normalize its parameter specification."""
        self.name = name
        self._parameters = _coerce_params(parameters)

    def __repr__(self) -> str:
        """Render the tree with indented groups and scalar parameter representations."""
        # TODO: make that easier to maintain
        def _repr(d: ParameterTree, level: int = 0) -> str:
            """Render one subtree at its indentation level."""
            s = ""
            if level == 0:
                s += "Parameters " + self.name + "\n"
            for k, v in d.items():
                s += "    " * level
                if isinstance(v, Parameter):
                    s += v.__repr__() + "\n"
                else:
                    s += f"* {k}\n"
                    s += _repr(v, level + 1)
            return s

        return _repr(self._parameters)

    def __iter__(self) -> Iterator[Parameter | ParameterTree]:
        """Iterate over immediate child leaves or subtrees, without flattening."""
        return iter(self._parameters.values())

    def __eq__(self, other: object) -> bool:
        """Compare another parameter tree, preserving the Parameters-only contract."""
        return self._parameters == cast(Parameters, other)._parameters

    def __add__(self, other: "Parameters") -> "Parameters":
        """Group parameters by name, suffixing colliding names with left/right.

        The collection name is unchanged by disambiguation. Parameters remain
        shared with the operands, as with noncolliding compositions.
        """
        left_name = self.name if self.name else "left"
        right_name = other.name if other.name else "right"
        name = f"{left_name}__{right_name}"
        if left_name == right_name:
            left_name += "__left"
            right_name += "__right"
        return Parameters(
            parameters={left_name: self._parameters, right_name: other._parameters},
            name=name,
        )

    @property
    def parameters(self) -> list[Parameter]:
        """Flatten the tree into a new list of shared Parameter leaves."""
        # TODO: use a flatten dict to hold the parameters

        def _leafs(node: Parameter | ParameterTree) -> list[Parameter]:
            """Collect leaves in recursive dictionary insertion order."""
            if isinstance(node, Parameter):
                return [node]
            return [leaf for child in node.values() for leaf in _leafs(child)]

        return _leafs(self._parameters)

    def __len__(self) -> int:
        """Return the number of scalar leaves, including fixed parameters."""
        return self.n_par

    def set_parameter(self, *args: str, **kwargs: Any) -> None:
        """Change settings of Parameters after instantiation

        p_alpha = Parameters(['a', 'b', 'c'], name='alpha')
        p_alpha.set_parameter('a', value=1, transform='log')

        A single hierarchical identifier from ``ids`` is also accepted, or pass
        successive group names and a leaf name. Keyword arguments are Parameter
        constructor settings; the leaf is replaced, not edited in place.
        Unknown keys raise ``KeyError``. Invalid settings raise validation errors.
        """

        if len(args) == 1:
            args = next(
                (path for path in flatten(self._parameters) if "__".join(path) == args[0]),
                args,
            )
        node = self._parameters
        for key in args[:-1]:
            # Intermediate path entries are groups; the final entry is a leaf.
            node = cast(ParameterTree, node[key])
        parameter = cast(Parameter, node[args[-1]])
        node[args[-1]] = Parameter(**{**parameter.__dict__, **kwargs})

    @property
    def theta(self) -> NDArray[np.float64]:
        """Physical constrained θ, shape (n_par,), including fixed parameters."""
        return np.array([h.theta for h in self.parameters])

    @theta.setter
    def theta(self, x: Sequence[float] | NDArray[np.float64]) -> None:
        """Set all physical values; reject a length other than n_par."""
        if len(x) != self.n_par:
            raise ValueError(f"{len(x)} values are given but {self.n_par} are expected")

        for p, value in zip(self.parameters, x):
            p.theta = value

    @property
    def theta_free(self) -> NDArray[np.float64]:
        """Physical constrained θ for free parameters, shape (n_par_free,)."""
        return np.array([p.theta for p in self.parameters_free])

    @theta_free.setter
    def theta_free(self, x: Sequence[float] | NDArray[np.float64]) -> None:
        """Set free physical values; reject a length other than n_par_free."""
        if len(x) != self.n_par_free:
            raise ValueError(
                f"{len(x)} values are given but {self.n_par_free} are expected"
            )

        for p, value in zip(self.parameters_free, x):
            p.theta = value

    @property
    def theta_sd(self) -> NDArray[np.float64]:
        """Standardized constrained θ_sd for free leaves, shape (n_par_free,)."""
        return np.array([p.theta_sd for p in self.parameters_free])

    @theta_sd.setter
    def theta_sd(self, x: Sequence[float] | NDArray[np.float64]) -> None:
        """Set free standardized values; reject a length other than n_par_free."""
        if len(x) != self.n_par_free:
            raise ValueError(
                f"{len(x)} values are given but {self.n_par_free} are expected"
            )

        for p, value in zip(self.parameters_free, x):
            p.theta_sd = value

    @property
    def theta_jacobian(self) -> NDArray[np.float64]:
        """Inverse-transform dθ_sd/dη for free leaves, shape (n_par_free,)."""
        return np.array([p.get_inv_transform_jacobian() for p in self.parameters_free])

    @property
    def theta_log_jacobian(self) -> float:
        """Sum log absolute dθ_sd/dη over free leaves, excluding physical scale."""
        return np.sum(np.log(np.abs(self.theta_jacobian)))

    @property
    def eta(self) -> NDArray[np.float64]:
        """Unconstrained η, shape (n_par,), including fixed parameters."""
        return np.array([p.eta for p in self.parameters])

    @eta.setter
    def eta(self, x: Sequence[float] | NDArray[np.float64]) -> None:
        """Set all unconstrained values; reject a length other than n_par."""
        if len(x) != self.n_par:
            raise ValueError(f"{len(x)} values are given but {self.n_par} are expected")

        for p, value in zip(self.parameters, x):
            p.eta = value

    @property
    def eta_free(self) -> NDArray[np.float64]:
        """Unconstrained η for free parameters, shape (n_par_free,)."""
        return self.eta[self.free]

    @eta_free.setter
    def eta_free(self, x: Sequence[float] | NDArray[np.float64]) -> None:
        """Set free unconstrained values; reject a length other than n_par_free."""
        if len(x) != self.n_par_free:
            raise ValueError(
                f"{len(x)} values are given but {self.n_par_free} are expected"
            )

        for p, value in zip(self.parameters_free, x):
            p.eta = value

    @property
    def eta_jacobian(self) -> list[float]:
        """List forward-transform dη/dθ_sd for free leaves, excluding physical scale."""
        return [p.get_transform_jacobian() for p in self.parameters_free]

    @property
    def penalty(self) -> float:
        """Sum free-leaf standardized-coordinate bound penalties, scaled by 1e-4."""
        SCALING = 1e-4
        return SCALING * np.sum([p.get_penalty() for p in self.parameters_free])

    @property
    def d_penalty(self) -> list[float]:
        """List free-leaf penalty derivatives with respect to θ_sd, scaled by 1e-4."""
        SCALING = 1e-4
        return [SCALING * p.get_grad_penalty() for p in self.parameters_free]

    @property
    def prior(self) -> float:
        """Sum log prior densities on θ_sd for free leaves with an assigned prior."""
        return np.sum(
            [
                # Scalar standardized inputs produce scalar log densities.
                cast(float, p.prior.logpdf(p.theta_sd))
                for p in self.parameters_free
                if p.prior is not None
            ]
        )

    @property
    def free(self) -> list[bool]:
        """List a Boolean free-parameter mask in flattened leaf order."""
        return [p.free for p in self.parameters]

    @property
    def names(self) -> list[str]:
        """List leaf names in flattened order, including duplicate local names."""
        return [p.name for p in self.parameters]

    @property
    def ids(self) -> list[str]:
        """List flattened hierarchical identifiers with groups joined by '__'."""
        return list(
            {"__".join(path): value for path, value in flatten(self._parameters).items()}
        )

    @property
    def ids_free(self) -> list[str]:
        """Hierarchical identifiers of free parameters, in vector order."""
        return [id_ for id_, free in zip(self.ids, self.free) if free]

    @property
    def names_free(self) -> list[str]:
        """List local names of free leaves in vector order."""
        return [p.name for p in self.parameters_free]

    def prior_init(self, hpd: float | None = None) -> None:
        """Draw standardized constrained values from priors of free parameters.

        Fixed parameters and parameters without a prior are unchanged. Both the
        constrained value and its unconstrained transform are synchronized.

        Arguments
        ---------
        hpd : float or None
            Central equal-tail prior probability mass in (0, 1]; None uses
            the full distribution. Invalid mass raises ValueError.
        """
        for p in self.parameters_free:
            if p.prior is not None:
                p.theta_sd = p.prior.random(hpd=hpd)[0]

    @property
    def scale(self) -> NDArray[np.float64]:
        """Physical scaling factors for free leaves, shape (n_par_free,)."""
        return np.array([p.scale for p in self.parameters_free])

    @property
    def n_par(self) -> int:
        """Number of scalar leaves, including fixed parameters."""
        return len(self.parameters)

    @property
    def n_par_free(self) -> int:
        """Number of free scalar leaves."""
        return len(self.parameters_free)

    @property
    def parameters_free(self) -> list[Parameter]:
        """New list of shared free Parameter leaves in flattened order."""
        return [p for p in self.parameters if p.free]

    def init_parameters(
        self,
        n_init: int = 1,
        method: Literal[
            "unconstrained", "prior", "zero", "fixed", "value"
        ] = "unconstrained",
        hpd: float = 0.95,
    ) -> NDArray[np.float64]:
        """Random initialization of the parameters

        Parameters
        ----------
        n_init: int, default 1
            Number of random initialization
        method: str, default 'unconstrained'
            ``unconstrained`` draws uniformly in [-1, 1]; ``prior`` draws
            standardized constrained values from each prior; ``zero`` sets
            unconstrained values to zero; ``fixed`` uses current values.
            ``value`` draws within +/- 25% of the original standardized
            magnitude, clipped to the transform's open bounds. Zero values
            remain zero, and fixed parameters are never changed.
        hpd: float, default 0.95
            Prior probability mass retained when drawing prior samples.

        Returns
        -------
        eta0: ndarray of shape (n_par_free,) or (n_par_free, n_init)
            Unconstrained free parameters. A single initialization preserves
            the parameter axis; multiple initializations occupy columns.

        Notes
        -----
        Prior and value initialization mutate the free leaves; after multiple
        draws the tree retains the final column. Other methods leave it unchanged.

        Raises
        ------
        TypeError
            If n_init is not a positive integer.
        ValueError
            If method is unknown or hpd is outside (0, 1].
        """

        if not isinstance(n_init, int) or n_init <= 0:
            raise TypeError("`n_init` must an integer greater or equal to 1")

        available_methods = ["unconstrained", "prior", "zero", "fixed", "value"]
        if method not in available_methods:
            raise ValueError(
                f"`method` must be one of the following {available_methods}"
            )

        if not (0.0 < hpd <= 1.0):
            raise ValueError("`hpd` must be between ]0, 1]")

        n_par = len(self.eta_free)
        if method == "unconstrained":
            eta0 = np.random.uniform(-1, 1, (n_par, n_init))
        elif method == "zero":
            eta0 = np.zeros((n_par, n_init))
        else:
            eta0 = np.zeros((n_par, n_init))
            lb = ub = None
            if method == "value":
                value = self.theta_sd
                radius = 0.25 * np.abs(value)
                lb, ub = value - radius, value + radius
                for i, p in enumerate(self.parameters_free):
                    lower = p._transform.lb
                    upper = p._transform.ub
                    if p._transform.name == "log":
                        lower = 0.0
                    if lower is not None:
                        lb[i] = max(lb[i], np.nextafter(lower, np.inf))
                    if upper is not None:
                        ub[i] = min(ub[i], np.nextafter(upper, -np.inf))
            for n in range(n_init):
                if method == "prior":
                    self.prior_init(hpd=hpd)
                elif method == "value":
                    self.theta_sd = np.random.uniform(
                        cast(NDArray[np.float64], lb), cast(NDArray[np.float64], ub)
                    )

                eta0[:, n] = self.eta_free

        return eta0[:, 0] if n_init == 1 else eta0
