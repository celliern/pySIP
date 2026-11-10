"""Dictionary-backed namespaces with attribute access and pickle support."""

from typing import Any


class Namespace(dict[str, Any]):
    """A namespace object that allows attributes to be accessed as dict keys.

    This class is a subclass of dict, so it can be used as a dict.  It
    also allows attributes to be accessed as dict keys.  For example::

        >>> ns = Namespace()
        >>> ns['foo'] = 'bar'
        >>> ns.foo
        'bar'
        >>> ns.foo = 'baz'
        >>> ns['foo']
        'baz'

    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize from dict-compatible arguments and share attribute storage."""
        super().__init__(*args, **kwargs)
        self.__dict__ = self

    def __repr__(self) -> str:
        """Return the dictionary representation wrapped in the class name."""
        return f"Namespace({super().__repr__()})"

    def __getstate__(self) -> dict[str, Any]:
        """Return the live dictionary of dynamic values as the pickle payload."""
        return self.__dict__

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Merge a pickle payload and restore shared attribute storage."""
        self.update(state)
        self.__dict__ = self

    def __delattr__(self, name: str) -> None:
        """Delete a stored attribute, raising AttributeError if it is absent."""
        if name in self:
            del self[name]
        else:
            raise AttributeError(name)

    def __dir__(self) -> list[str]:
        """Return stored keys rather than inherited dictionary attributes."""
        return list(self.keys())

    def copy(self) -> "Namespace":
        """Return a shallow namespace copy; stored values remain shared."""
        return Namespace(self)
