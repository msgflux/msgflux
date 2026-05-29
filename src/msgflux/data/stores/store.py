from __future__ import annotations

from typing import Any

from msgflux.data.stores.registry import store_registry


class Store:
    """Factory for registered msgFlux stores."""

    @classmethod
    def providers(cls) -> dict[str, list[str]]:
        # Import lazily so inbox providers register without creating package
        # initialization cycles.
        import msgflux.runtime.agent_inbox  # noqa: F401, PLC0415

        return {
            store_type: list(providers)
            for store_type, providers in store_registry.items()
        }

    @classmethod
    def _create(cls, store_type: str, provider: str, **kwargs: Any) -> Any:
        providers = store_registry.get(store_type)
        if providers is None:
            raise ValueError(f"Store type `{store_type}` is not supported")
        if provider not in providers:
            raise ValueError(
                f"Provider `{provider}` not registered for store type `{store_type}`"
            )
        return providers[provider](**kwargs)

    @classmethod
    def agent_inbox(cls, provider: str, **kwargs: Any) -> Any:
        # Import lazily so inbox providers register without creating package
        # initialization cycles.
        import msgflux.runtime.agent_inbox  # noqa: F401, PLC0415

        return cls._create("agent_inbox", provider, **kwargs)

    @classmethod
    def checkpoint(cls, provider: str, **kwargs: Any) -> Any:
        return cls._create("checkpoint", provider, **kwargs)
