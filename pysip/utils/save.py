"""Version-stamped model persistence using trusted pickle files."""

import pickle
from collections.abc import Callable
from os import PathLike
from typing import Any, ParamSpec, TypeVar, cast

from pysip import __version__

P = ParamSpec("P")
R = TypeVar("R")


def with_extension(extension: str) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Append a suffix to the first positional path of a positional-call wrapper.

    The wrapped callable's static signature is preserved. At runtime only
    positional arguments are accepted; an existing matching suffix is retained.
    """
    def _extension(f: Callable[P, R]) -> Callable[P, R]:
        """Wrap a callable without changing its result or exception behavior."""
        def wrapper(*args: Any) -> R:
            """Normalize the positional path and forward remaining arguments."""
            a = str(args[0])
            if not a.endswith("." + extension):
                a += "." + extension
            return cast(Callable[..., R], f)(a, *args[1:])

        return cast(Callable[P, R], wrapper)

    return _extension


@with_extension("pickle")
def save_model(path: str | PathLike[str], model: Any) -> None:
    """Stamp ``model.__version__`` in place and pickle it to ``path``.

    A missing ``.pickle`` suffix is appended. Existing files are overwritten.
    The object must allow version assignment and support pickle; filesystem
    and serialization errors propagate to the caller.
    """
    model.__version__ = __version__
    with open(path, "wb") as f:
        pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)


@with_extension("pickle")
def load_model(path: str | PathLike[str]) -> Any:
    """Return the object unpickled from ``path`` with a ``.pickle`` suffix.

    Only load trusted files: unpickling can execute arbitrary code. The saved
    version is not checked. Filesystem and unpickling errors propagate.
    """
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model
