from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T", bound=Callable[..., Any])

store_registry: dict[str, dict[str, Callable[..., Any]]] = {}


def register_store(store_type: str, provider: str) -> Callable[[T], T]:
    """Register a store provider under a typed store namespace."""

    def decorator(factory: T) -> T:
        store_registry.setdefault(store_type, {})[provider] = factory
        return factory

    return decorator
