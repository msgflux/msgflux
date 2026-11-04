"""Authentication methods supported by managed accounts.

Builtin providers keep their static specs below. Providers declared in a
:class:`~msgflux.models.catalog.ProviderCatalog` derive their auth from the
declaration instead:

- An explicit ``api_key_env`` always means API-key auth, even when the base
  provider is the OAuth-based Codex provider.
- Otherwise the base provider's auth *mechanism* is inherited, but never its
  credential account or env var: API-key aliases get their own derived env
  var (``<ALIAS>_API_KEY``), and OAuth aliases reuse the Codex adapter under
  their own provider/account namespace.
- External factory classes are imported only when auth for that provider is
  actually requested; static catalog/listing inspection never imports them.
  Factory auth follows :class:`ProviderEnvBase` metadata, except Codex
  descendants, which inherit OAuth unless they explicitly declare an API-key
  env override. Programmatically registered Codex descendants (no config)
  inherit OAuth by class identity as well.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING, Literal

import msgspec

from msgflux.models.provider_env import ProviderEnvBase

if TYPE_CHECKING:  # pragma: no cover - typing only
    from msgflux.models.catalog import ProviderCatalog


class ProviderAuthSpec(msgspec.Struct, frozen=True):
    provider: str
    method: Literal["api_key", "oauth"]
    env_name: str | None = None
    # OAuth adapter family (currently only "codex"); None for API-key specs.
    oauth_adapter: str | None = None


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
    "openai-codex": ProviderAuthSpec("openai-codex", "oauth", oauth_adapter="codex"),
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


def _active_provider_catalog(
    provider_catalog: ProviderCatalog | None,
) -> ProviderCatalog | None:
    """Return the explicit catalog, else the active catalog, else ``None``."""
    if provider_catalog is not None:
        return provider_catalog
    from msgflux.models.catalog import get_current_provider_catalog  # noqa: PLC0415

    return get_current_provider_catalog()


def _derive_api_key_env(provider: str) -> str:
    """Derive an alias's own API-key env var; never reuse the base env var."""
    stem = re.sub(r"[^0-9A-Za-z]+", "_", provider).strip("_").upper()
    return f"{stem}_API_KEY"


def _codex_model_class() -> type:
    from msgflux.models.providers.openai_codex import (  # noqa: PLC0415
        OpenAICodexChatCompletion,
    )

    return OpenAICodexChatCompletion


def _explicit_api_key_env(cls: type, *, stop_at_codex: bool = False) -> str | None:
    """API-key env explicitly declared on the class or its own bases."""
    for klass in cls.__mro__:
        if stop_at_codex and klass is _codex_model_class():
            break
        env_name = klass.__dict__.get("api_key_env")
        if env_name:
            return str(env_name)
    return None


def _class_auth(provider: str, cls: type) -> ProviderAuthSpec:
    """Auth from a model class's own metadata/inheritance (no config)."""
    declared_provider = getattr(cls, "provider", "")
    if declared_provider and declared_provider != provider:
        raise ValueError(
            f"Provider factory for {provider!r} declares provider "
            f"{declared_provider!r}; refusing to relabel its credentials"
        )
    if issubclass(cls, _codex_model_class()):
        # Codex descendants inherit OAuth unless they explicitly declare an
        # API-key env override between themselves and the Codex base class.
        env_name = _explicit_api_key_env(cls, stop_at_codex=True)
        if env_name is not None:
            return ProviderAuthSpec(provider, "api_key", env_name)
        return ProviderAuthSpec(provider, "oauth", oauth_adapter="codex")
    env_name = _explicit_api_key_env(cls)
    if env_name is not None:
        return ProviderAuthSpec(provider, "api_key", env_name)
    raise ValueError(
        f"Provider {provider!r} does not declare an API-key env; "
        "set api_key_env in its provider declaration or factory class"
    )


def _config_auth(
    provider: str,
    config,
    catalog: ProviderCatalog,
    *,
    _seen: tuple[str, ...],
) -> ProviderAuthSpec:
    if config.api_key_env:
        # An explicit env means API-key auth, even for a Codex base.
        return ProviderAuthSpec(provider, "api_key", config.api_key_env)
    if config.base is not None:
        base_spec = _resolve_auth(config.base, catalog, _seen=(*_seen, provider))
        if base_spec.method == "oauth":
            # Inherit the mechanism (and adapter), not the account: the
            # alias stores credentials under its own provider namespace.
            return ProviderAuthSpec(
                provider, "oauth", oauth_adapter=base_spec.oauth_adapter
            )
        # Inherit the API-key mechanism with the alias's own env var.
        return ProviderAuthSpec(provider, "api_key", _derive_api_key_env(provider))
    # Factory: importing the class happens only here, when its auth is
    # actually requested via the catalog's trusted resolution.
    model_class = catalog.get_model_class("chat_completion", provider)
    return _class_auth(provider, model_class)


def _registered_auth(
    provider: str, catalog: ProviderCatalog
) -> ProviderAuthSpec | None:
    """Class-declared auth for a catalog-registered (or builtin) provider."""
    try:
        model_class = catalog.get_model_class("chat_completion", provider)
    except ValueError:
        return None
    try:
        return _class_auth(provider, model_class)
    except ValueError:
        return None


def _resolve_auth(
    provider: str, catalog: ProviderCatalog | None, *, _seen: tuple[str, ...] = ()
) -> ProviderAuthSpec:
    normalized = provider.strip().lower()
    if normalized in _seen:
        raise ValueError(f"Cyclic provider derivation involving {provider!r}")
    config = catalog.get_config(normalized) if catalog is not None else None
    if config is not None:
        return _config_auth(normalized, config, catalog, _seen=_seen)
    if catalog is not None:
        registered = _registered_auth(normalized, catalog)
        if registered is not None:
            return registered
    if catalog is None:
        from msgflux.models.registry import model_registry  # noqa: PLC0415

        model_class = model_registry.get("chat_completion", {}).get(normalized)
        if model_class is not None:
            try:
                return _class_auth(normalized, model_class)
            except ValueError:
                pass
    spec = _OAUTH_PROVIDERS.get(normalized) or _discover_api_key_providers().get(
        normalized
    )
    if spec is None:
        raise ValueError(f"Unsupported account provider: {provider!r}")
    return spec


def get_provider_auth(
    provider: str, *, provider_catalog: ProviderCatalog | None = None
) -> ProviderAuthSpec:
    catalog = _active_provider_catalog(provider_catalog)
    return _resolve_auth(provider, catalog)


def _needs_factory_auth(provider: str, catalog: ProviderCatalog) -> bool:
    """Whether static auth inspection would need an external class import."""
    seen = set()
    while provider not in seen:
        seen.add(provider)
        config = catalog.get_config(provider)
        if config is None or config.api_key_env is not None:
            return False
        if config.factory is not None:
            return True
        provider = config.base
    return True


def _catalog_specs(
    catalog: ProviderCatalog, method: Literal["api_key", "oauth"]
) -> dict[str, ProviderAuthSpec]:
    """Catalog specs knowable without importing any external factory class."""
    result: dict[str, ProviderAuthSpec] = {}
    for name in catalog.providers().get("chat_completion", ()):
        normalized = name.strip().lower()
        if _needs_factory_auth(normalized, catalog):
            continue
        try:
            spec = _resolve_auth(normalized, catalog)
        except ValueError:
            continue
        if spec.method == method:
            result[spec.provider] = spec
    return result


def api_key_providers(
    *, provider_catalog: ProviderCatalog | None = None
) -> dict[str, ProviderAuthSpec]:
    """Return discovered API key providers after the model registry loads."""
    result = dict(_discover_api_key_providers())
    catalog = _active_provider_catalog(provider_catalog)
    if catalog is not None:
        result.update(_catalog_specs(catalog, "api_key"))
    return result


def oauth_providers(
    *, provider_catalog: ProviderCatalog | None = None
) -> dict[str, ProviderAuthSpec]:
    result = dict(_OAUTH_PROVIDERS)
    catalog = _active_provider_catalog(provider_catalog)
    if catalog is not None:
        result.update(_catalog_specs(catalog, "oauth"))
    return result


def get_api_key_env(
    provider: str, *, provider_catalog: ProviderCatalog | None = None
) -> str:
    spec = get_provider_auth(provider, provider_catalog=provider_catalog)
    if spec.method != "api_key" or spec.env_name is None:
        raise ValueError(f"Provider {provider!r} does not use an API key")
    return spec.env_name
