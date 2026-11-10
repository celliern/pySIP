"""Expose generated constructor signatures without inspecting entire packages."""

import inspect
from dataclasses import is_dataclass
from importlib import import_module
from typing import Annotated, Any, cast, get_args, get_origin

from griffe import Class, Extension, Function, Parameter, ParameterKind, Parameters

from pysip.params.prior import BasePrior


class RuntimeConstructors(Extension):
    """Supply real Pydantic, dataclass, and prior constructor signatures to Griffe."""

    def on_class(self, *, cls: Class, **kwargs: Any) -> None:
        """Augment module-level pySIP classes after static source collection."""
        if not cls.module.path.startswith("pysip.") or not cls.parent or not cls.parent.is_module:
            return
        runtime = cast(type, getattr(import_module(cls.module.path), cls.name))
        if not is_dataclass(runtime) and not (issubclass(runtime, BasePrior) and runtime is not BasePrior):
            return
        parameters = Parameters(Parameter("self", kind=ParameterKind.positional_or_keyword))
        signature = inspect.signature(runtime.__init__)
        if any(value.kind in (value.VAR_POSITIONAL, value.VAR_KEYWORD) for value in signature.parameters.values()):
            signature = inspect.signature(cast(type, runtime))
        for value in signature.parameters.values():
            if value.name == "self":
                continue
            annotation = value.annotation
            if get_origin(annotation) is Annotated:
                annotation = get_args(annotation)[0]
            parameters.add(Parameter(
                value.name,
                annotation=None if annotation is inspect.Parameter.empty else inspect.formatannotation(annotation),
                kind=ParameterKind[value.kind.name.lower()],
                default=None if value.default is inspect.Parameter.empty else repr(value.default),
            ))
        cls.set_member("__init__", Function("__init__", parameters=parameters, returns="None", parent=cls))
