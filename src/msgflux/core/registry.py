from functools import partial
from typing import Any, Callable, TypeVar, overload

T = TypeVar("T", bound=Callable[..., Any])


class Registry:
    """Instance-based callable registry.

    A decorator-based registry that collects callables (tools, modules, functions)
    and exposes them as a list or a name-to-callable dict.

    Examples
    --------
    Register tools for an Agent::

        tools = Registry()

        @tools
        def search(query: str) -> str: ...

        @tools(name="web_fetch")
        def fetch(url: str) -> str: ...

        agent = Agent(tools=tools.to_list())

    Register modules for Inline::

        modules = Registry()

        @modules(name="summarizer")
        class Summarizer(Module): ...

        pipeline = Inline(modules=modules.to_items())
    """

    def __init__(self) -> None:
        self._entries: dict[str, Any] = {}

    def _resolve_name(self, obj: Any) -> str:
        """Extract a registration name from a callable-like object."""
        if hasattr(obj, "name"):
            name = obj.name
            if callable(name) and not isinstance(name, str):
                name = name()
            if isinstance(name, str) and name:
                return name

        dunder_name = getattr(obj, "__name__", None)
        if isinstance(dunder_name, str) and dunder_name:
            return dunder_name

        if isinstance(obj, partial):
            return self._resolve_name(obj.func)

        wrapped = getattr(obj, "__wrapped__", None)
        if wrapped is not None:
            return self._resolve_name(wrapped)

        class_name = getattr(obj.__class__, "__name__", None)
        if isinstance(class_name, str) and class_name:
            return class_name

        raise TypeError(
            "Unable to resolve a registry name. Provide `name=` or define `.name` "
            "or `.__name__` on the registered object."
        )

    @overload
    def __call__(self, obj: T) -> T: ...

    @overload
    def __call__(self, *, name: str) -> Callable[[T], T]: ...

    @overload
    def __call__(self, name: str) -> Callable[[T], T]: ...

    def __call__(
        self,
        obj: T | str | None = None,
        *,
        name: str | None = None,
    ) -> T | Callable[[T], T]:
        # @registry
        if obj is not None and callable(obj) and not isinstance(obj, str):
            resolved = self._resolve_name(obj)
            self._entries[resolved] = obj
            return obj

        # @registry("custom_name") or @registry(name="custom_name")
        explicit_name = obj if isinstance(obj, str) else name

        def decorator(fn: T) -> T:
            key = explicit_name if explicit_name else self._resolve_name(fn)
            self._entries[key] = fn
            return fn

        return decorator

    def to_list(self) -> list[Any]:
        """Return registered callables as a list."""
        return list(self._entries.values())

    def to_items(self) -> dict[str, Any]:
        """Return registered callables as a ``{name: callable}`` dict."""
        return dict(self._entries)

    def update(self, other: "Registry") -> "Registry":
        """Merge entries from *other* into this registry. Returns ``self`` for chaining."""
        self._entries.update(other._entries)
        return self

    def get(self, name: str) -> Any:
        """Return a single registered callable by name."""
        return self._entries[name]

    def pop(self, name: str) -> Any:
        """Remove and return the callable registered under *name*."""
        return self._entries.pop(name)

    def clear(self) -> None:
        """Remove all entries."""
        self._entries.clear()

    def __contains__(self, name: str) -> bool:
        return name in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self):
        return iter(self._entries.values())

    def __repr__(self) -> str:
        names = list(self._entries.keys())
        return f"Registry({names})"
