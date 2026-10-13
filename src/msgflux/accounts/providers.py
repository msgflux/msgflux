"""Authentication methods supported by managed accounts."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Literal

import msgspec

from msgflux.models.provider_env import ProviderEnvBase


class ProviderAuthSpec(msgspec.Struct, frozen=True):
    provider: str
    method: Literal["api_key", "oauth"]
    env_name: str | None = None


def _discover_api_key_providers() -> dict[str, ProviderAuthSpec]:
    from msgflux.models.registry import model_registry  # noqa: PLC0415

    result: dict[str, ProviderAuthSpec] = {}
    # The model registry performs its existing provider autoload once, lazily
    # on first lookup. Iterate its registered classes instead of scanning the
    # provider package independently.
    for model_types in model_registry.values():
        for candidate in model_types.values():
            if not isinstance(candidate, type) or not issubclass(
                candidate, ProviderEnvBase
            ):
                continue
            provider = getattr(candidate, "provider", "")
            env_name = getattr(candidate, "api_key_env", "")
            if provider and env_name:
                result.setdefault(
                    provider, ProviderAuthSpec(provider, "api_key", env_name)
                )
    return result


_OAUTH_PROVIDERS = {
    "openai-codex": ProviderAuthSpec("openai-codex", "oauth"),
}


class _LazyAPIKeyProviders(Mapping[str, ProviderAuthSpec]):
    def __getitem__(self, key: str) -> ProviderAuthSpec:
        return _discover_api_key_providers()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(_discover_api_key_providers())

    def __len__(self) -> int:
        return len(_discover_api_key_providers())


API_KEY_PROVIDERS = _LazyAPIKeyProviders()
OAUTH_PROVIDERS = _OAUTH_PROVIDERS


def get_provider_auth(provider: str) -> ProviderAuthSpec:
    normalized = provider.strip().lower()
    spec = _OAUTH_PROVIDERS.get(normalized) or _discover_api_key_providers().get(
        normalized
    )
    if spec is None:
        raise ValueError(f"Unsupported account provider: {provider!r}")
    return spec


def api_key_providers() -> dict[str, ProviderAuthSpec]:
    """Return discovered API key providers after the model registry loads."""
    return dict(_discover_api_key_providers())


def oauth_providers() -> dict[str, ProviderAuthSpec]:
    return dict(_OAUTH_PROVIDERS)


def get_api_key_env(provider: str) -> str:
    spec = get_provider_auth(provider)
    if spec.method != "api_key" or spec.env_name is None:
        raise ValueError(f"Provider {provider!r} does not use an API key")
    return spec.env_name
