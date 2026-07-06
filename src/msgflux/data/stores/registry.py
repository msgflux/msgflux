from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T", bound=Callable[..., Any])

store_registry: dict[str, dict[str, Callable[..., Any]]] = {}


def register_store(store_type: str | None = None) -> Callable[[T], T]:
    """Register a store provider under a typed store namespace."""

    def decorator(factory: T) -> T:
        resolved_store_type = store_type or getattr(factory, "store_type", None)
        provider = getattr(factory, "provider", None)

        if not resolved_store_type or not provider:
            raise ValueError(
                f"{factory.__name__} must define `store_type` and `provider`."
            )

        store_registry.setdefault(resolved_store_type, {})[provider] = factory
        return factory

    return decorator
