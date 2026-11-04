"""Catalog-scoped dynamic provider definitions.

A :class:`ProviderCatalog` holds trusted provider declarations without
mutating the global model registry. ``base`` declarations derive a lazily
created, named subclass of the base provider class so the base runtime
(reasoning/history codecs, capabilities and transport) is inherited in
full while the derived class carries the alias provider identity and
endpoint/env metadata. Derived classes are cached per catalog and are
never registered globally or serialized. ``factory`` declarations load a
trusted external class; the import runs under a scoped registration
context so ``@register_model`` decorators in that module register only
in this catalog.
"""

from __future__ import annotations

import copy
import inspect
import re
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Mapping
from urllib.parse import urlparse

import msgspec

from msgflux.models.base import BaseModel
from msgflux.models.model_credentials import BearerTokenCredentialResolver
from msgflux.models.provider_env import ProviderEnvBase
from msgflux.models.registry import (
    _scoped_import_catalog,
    model_registry,
)
from msgflux.utils.imports import load_factory, validate_factory_path
from msgflux.utils.msgspec import lossless_json_roundtrip
from msgflux.utils.validation import is_valid_account_name

_RESERVED_OPTION_KEYS = frozenset(
    {
        "model_id",
        "provider",
        "credential_resolver",
        "provider_catalog",
        # URL / env overrides are first-class ProviderConfig fields.
        "base_url",
        "api_key_env",
    }
)
_CREDENTIAL_OPTION_KEYS = frozenset(
    {
        "api_key",
        "authorization",
        "password",
        "access_token",
        "refresh_token",
        "auth",
        "secret_key",
        "client_secret",
        "token",
        "cookie",
        "set_cookie",
    }
)
_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _validate_provider_name(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"`{field}` must be a non-empty provider name string")
    if not is_valid_account_name(value.strip()):
        raise ValueError(f"`{field}` must be a safe managed provider name")
    return value.strip()


def _reject_credential_options(value: Any, path: str = "options") -> None:
    """Reject credential-bearing keys anywhere inside ``options``."""
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                continue
            normalized = key.strip().lower().replace("-", "_")
            if (
                normalized in _CREDENTIAL_OPTION_KEYS
                or "api_key" in normalized
                or normalized.endswith("_password")
                or normalized in {"bearer_token", "id_token", "session_token"}
            ):
                raise ValueError(
                    f"`options` must not contain credential-bearing field "
                    f"{key!r} at {path}; credentials are resolved at request "
                    "time, never from provider config options"
                )
            _reject_credential_options(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_credential_options(item, f"{path}[{index}]")


class ProviderConfig(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """Trusted declaration for one dynamic provider alias."""

    base: str | None = None
    factory: str | None = None
    base_url: str | None = None
    api_key_env: str | None = None
    options: dict[str, Any] = msgspec.field(default_factory=dict)

    def __post_init__(self) -> None:
        self._validate_source_fields()
        self._validate_options()

    def _validate_source_fields(self) -> None:
        if (self.base is None) == (self.factory is None):
            raise ValueError("exactly one of `base` and `factory` must be set")
        if self.base is not None:
            msgspec.structs.force_setattr(
                self, "base", _validate_provider_name(self.base, "base")
            )
        if self.factory is not None:
            if not isinstance(self.factory, str):
                raise ValueError("`factory` must be a 'module:attribute' string")
            validate_factory_path(self.factory)
        if self.base_url is not None:
            if not isinstance(self.base_url, str) or not self.base_url.strip():
                raise ValueError("`base_url` must be a non-empty URL string")
            parsed = urlparse(self.base_url.strip())
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ValueError("`base_url` must be an http(s) URL")
            msgspec.structs.force_setattr(self, "base_url", self.base_url.strip())
        if self.api_key_env is not None:
            if not isinstance(self.api_key_env, str) or not _ENV_NAME_RE.match(
                self.api_key_env.strip()
            ):
                raise ValueError("`api_key_env` must be an environment variable name")
            msgspec.structs.force_setattr(self, "api_key_env", self.api_key_env.strip())

    def _validate_options(self) -> None:
        if not isinstance(self.options, Mapping):
            raise TypeError("`options` must be a mapping of constructor options")
        options = dict(self.options)
        for key in options:
            if not isinstance(key, str):
                raise TypeError("`options` keys must be strings")
            if key in _RESERVED_OPTION_KEYS:
                raise ValueError(
                    f"`options` must not contain reserved key {key!r}; "
                    "use the first-class ProviderConfig field instead"
                )
        _reject_credential_options(options)
        lossless, plain_options = lossless_json_roundtrip(options)
        if not lossless:
            raise ValueError("`options` must contain only plain, non-secret JSON data")
        msgspec.structs.force_setattr(self, "options", plain_options)


_current_catalog_var: ContextVar[ProviderCatalog | None] = ContextVar(
    "msgflux_current_provider_catalog", default=None
)


def get_current_provider_catalog() -> ProviderCatalog | None:
    """Return the catalog active via :meth:`ProviderCatalog.use`, if any."""
    return _current_catalog_var.get()


def _source_is_configured(source: Any) -> bool:
    return isinstance(source, Mapping) and (
        "base" in source or "factory" in source or "chain" in source
    )


def _constructor_accepts(cls: type, name: str) -> bool:
    try:
        parameters = inspect.signature(cls).parameters
    except (TypeError, ValueError):
        return False
    return name in parameters


def _class_name(provider: str, model_type: str, base_cls: type) -> str:
    parts = re.split(r"[^0-9A-Za-z]+", provider)
    prefix = "".join(part[:1].upper() + part[1:] for part in parts if part)
    suffix_parts = re.split(r"[^0-9A-Za-z]+", model_type)
    suffix = "".join(part[:1].upper() + part[1:] for part in suffix_parts if part)
    name = f"{prefix}{suffix}Model"
    if name == base_cls.__name__:
        name = f"{prefix}{suffix}DerivedModel"
    return name


def _alias_api_key_env(provider: str) -> str:
    normalized = re.sub(r"[^0-9A-Za-z]+", "_", provider).strip("_")
    return f"{normalized.upper()}_API_KEY"


def _codex_base_class() -> type:
    from msgflux.models.providers.openai_codex import (  # noqa: PLC0415
        OpenAICodexChatCompletion,
    )

    return OpenAICodexChatCompletion


def _has_explicit_api_key_env_before_codex(cls: type) -> bool:
    """True if a Codex descendant declares a nonempty env before Codex."""
    codex = _codex_base_class()
    if not isinstance(cls, type) or not issubclass(cls, codex):
        return False
    for klass in cls.__mro__:
        if klass is codex:
            break
        env = klass.__dict__.get("api_key_env")
        if isinstance(env, str) and env.strip():
            return True
    return False


def _is_oauth_class(cls: type) -> bool:
    # Explicit inherited metadata wins over any other heuristic.
    if not isinstance(cls, type):
        return False
    method = getattr(cls, "provider_auth_method", None)
    if method is not None:
        return method == "oauth"
    codex = _codex_base_class()
    if not issubclass(cls, codex):
        return False
    if _has_explicit_api_key_env_before_codex(cls):
        return False
    return True


def _provider_env_api_key(self) -> str:
    return ProviderEnvBase._get_api_key(self)


def _codex_api_key_namespace(cls: type) -> dict:
    """Bearer/env namespace for a Codex descendant declaring its own env.

    Preserves an explicitly supplied custom resolver (or ``_get_api_key``)
    declared before the Codex base.
    """
    if not _has_explicit_api_key_env_before_codex(cls):
        return {}
    if _is_oauth_class(cls):
        return {}
    codex = _codex_base_class()
    namespace: dict[str, Any] = {"provider_auth_method": "api_key"}
    has_custom_resolver = False
    has_custom_get_api_key = False
    for klass in cls.__mro__:
        if klass is codex:
            break
        if "credential_resolver" in klass.__dict__:
            has_custom_resolver = True
        if "_get_api_key" in klass.__dict__:
            has_custom_get_api_key = True
    if not has_custom_resolver:
        namespace["credential_resolver"] = BearerTokenCredentialResolver()
    if not has_custom_get_api_key:
        namespace["_get_api_key"] = _provider_env_api_key
    return namespace


def _cleanup_instance(instance: Any) -> BaseException | None:
    """Close ``instance``; return any cleanup failure for the caller to note."""
    close = getattr(instance, "close", None)
    if not callable(close):
        return None
    try:
        close()
    except Exception as exc:  # re-reported as an error note by the caller
        return exc
    return None


def _copy_config(config: ProviderConfig) -> ProviderConfig:
    """Deep-copy a config so its options dict cannot mutate trusted state."""
    return ProviderConfig(
        base=config.base,
        factory=config.factory,
        base_url=config.base_url,
        api_key_env=config.api_key_env,
        options=copy.deepcopy(dict(config.options)),
    )


def _explicit_api_key_namespace(base_cls: type, config: ProviderConfig) -> dict:
    """Namespace for an explicit API-key override on an OAuth-shaped class.

    The Codex transport/body/codec are inherited, but request-time auth
    must use the bearer resolver and the env-based key lookup instead of
    the OAuth auth-file resolver and its empty ``_get_api_key``.
    """
    if config.api_key_env is None or not _is_oauth_class(base_cls):
        return {}
    # Only class-level resolver/key lookup: the builtin Codex
    # ``_initialize`` honors the inherited class-level resolver, and its
    # transport/body/codec are preserved via normal inheritance.
    return {
        "provider_auth_method": "api_key",
        "credential_resolver": BearerTokenCredentialResolver(),
        "_get_api_key": _provider_env_api_key,
    }


class ProviderCatalog:
    """An independent, trusted catalog of dynamic provider definitions."""

    def __init__(self, configured: Mapping[str, ProviderConfig] | None = None) -> None:
        self._configured: dict[str, ProviderConfig] = {}
        if configured is not None:
            if not isinstance(configured, Mapping):
                raise TypeError("`configured` must be a mapping of ProviderConfig")
            for provider, config in configured.items():
                name = _validate_provider_name(provider, "provider")
                if not isinstance(config, ProviderConfig):
                    raise TypeError(
                        f"Configured provider {name!r} must map to a ProviderConfig"
                    )
                if name in self._configured:
                    raise ValueError(f"Duplicate configured provider {name!r}")
                self._configured[name] = _copy_config(config)
        self._registrations: dict[str, dict[str, type[BaseModel]]] = {}
        self._derived_cache: dict[tuple[str, str], type[BaseModel]] = {}
        self._factory_cache: dict[str, type[BaseModel]] = {}

    def __repr__(self) -> str:
        return (
            f"ProviderCatalog(configured={sorted(self._configured)!r}, "
            f"registered={sorted(self._registrations)!r})"
        )

    def __contains__(self, provider: object) -> bool:
        return provider in self._configured

    @property
    def configured_names(self) -> tuple[str, ...]:
        """Configured provider names, in declaration order."""
        return tuple(self._configured)

    @property
    def configured(self) -> dict[str, ProviderConfig]:
        """Deep copies of the configured provider declarations."""
        return {name: _copy_config(config) for name, config in self._configured.items()}

    def providers(self) -> dict[str, list[str]]:
        """List known provider names without importing configured factories."""
        result = {kind: list(classes) for kind, classes in model_registry.items()}
        for kind, classes in self._registrations.items():
            result.setdefault(kind, []).extend(
                name for name in classes if name not in result.get(kind, ())
            )
        chat = result.setdefault("chat_completion", [])
        chat.extend(name for name in self._configured if name not in chat)
        return result

    @contextmanager
    def use(self):
        """Make this catalog the current catalog for nested model calls."""
        token = _current_catalog_var.set(self)
        try:
            yield self
        finally:
            _current_catalog_var.reset(token)

    def register(self, cls: type[BaseModel], *, replace: bool = False):
        """Register ``cls`` in this catalog only; returns the class."""
        model_type = getattr(cls, "model_type", None)
        provider = getattr(cls, "provider", None)
        if not model_type or not provider:
            raise ValueError(f"{cls.__name__} must define `model_type` and `provider`.")
        providers = self._registrations.setdefault(model_type, {})
        existing = providers.get(provider)
        if existing is cls:
            return cls
        if existing is not None and not replace:
            raise ValueError(
                f"Provider `{provider}` for type `{model_type}` is already "
                f"registered in this catalog as {existing.__name__!r}; pass "
                "replace=True to replace it explicitly."
            )
        providers[provider] = cls
        # Future derivations must see a newly registered/replaced base class.
        self._derived_cache.clear()
        return cls

    def get_config(self, provider: str) -> ProviderConfig | None:
        """Return a deep copy of the config for ``provider``, if configured."""
        config = self._configured.get(provider)
        return _copy_config(config) if config is not None else None

    # -- resolution ---------------------------------------------------------

    def get_model_class(self, model_type: str, provider: str) -> type[BaseModel]:
        return self._get_model_class(model_type, provider, seen=())

    def _get_model_class(
        self, model_type: str, provider: str, *, seen: tuple[str, ...]
    ) -> type[BaseModel]:
        config = self._configured.get(provider)
        if config is not None:
            if provider in seen:
                cycle = " -> ".join((*seen, provider))
                raise ValueError(f"Provider base cycle detected: {cycle}")
            if config.factory is not None:
                return self._factory_model_class(model_type, provider, config)
            return self._derived_model_class(
                model_type, provider, config, seen=(*seen, provider)
            )

        registered = self._registrations.get(model_type, {}).get(provider)
        if registered is not None:
            return registered
        return self._global_model_class(model_type, provider)

    @staticmethod
    def _global_model_class(model_type: str, provider: str) -> type[BaseModel]:
        # Accessing the registry performs the builtin autoload. This never
        # runs under a scoped factory import (autoload happens first), so
        # builtin decorators always land in the global registry.
        if model_type not in model_registry:
            raise ValueError(f"Model type `{model_type}` is not supported")
        providers = model_registry[model_type]
        if provider not in providers:
            raise ValueError(
                f"Provider `{provider}` not registered for type `{model_type}`"
            )
        return providers[provider]

    def _resolve_base_class(
        self, model_type: str, base: str, *, seen: tuple[str, ...]
    ) -> type[BaseModel]:
        if base in self._configured or base in {
            p for providers in self._registrations.values() for p in providers
        }:
            return self._get_model_class(model_type, base, seen=seen)
        try:
            return self._global_model_class(model_type, base)
        except ValueError as exc:
            raise ValueError(
                f"Unknown base provider {base!r} for type `{model_type}`: {exc}"
            ) from exc

    def _derived_model_class(
        self,
        model_type: str,
        provider: str,
        config: ProviderConfig,
        *,
        seen: tuple[str, ...],
    ) -> type[BaseModel]:
        if config.base is None:
            raise ValueError("configured provider requires `base`")
        cache_key = (model_type, provider)
        cached = self._derived_cache.get(cache_key)
        if cached is not None:
            return cached
        base_cls = self._resolve_base_class(model_type, config.base, seen=seen)
        if not (isinstance(base_cls, type) and issubclass(base_cls, BaseModel)):
            raise TypeError(
                f"Base provider {config.base!r} resolved to a non-model class"
            )
        is_oauth = _is_oauth_class(base_cls)
        namespace: dict[str, Any] = {
            "provider": provider,
            "model_type": model_type,
            "__module__": __name__,
            "__qualname__": (
                f"ProviderCatalog.{_class_name(provider, model_type, base_cls)}"
            ),
            "provider_auth_method": "oauth" if is_oauth else "api_key",
            "profile_provider": getattr(base_cls, "profile_provider", None)
            or config.base,
        }
        namespace.update(_explicit_api_key_namespace(base_cls, config))
        namespace.update(_codex_api_key_namespace(base_cls))
        if config.base_url is not None:
            namespace["base_url"] = config.base_url
            # The configured URL wins over the base provider's env override.
            namespace["base_url_env"] = None
        if config.api_key_env is not None:
            namespace["api_key_env"] = config.api_key_env
            namespace["display_name"] = provider
        elif not is_oauth:
            # A new alias gets its own account namespace; it must not
            # silently fall back to the builtin base provider's env var.
            # OAuth aliases keep the inherited mechanism (handled by auth).
            namespace["api_key_env"] = _alias_api_key_env(provider)
            namespace["display_name"] = provider
        derived = type(
            _class_name(provider, model_type, base_cls),
            (base_cls,),
            namespace,
        )
        self._derived_cache[cache_key] = derived
        return derived

    def _factory_model_class(
        self, model_type: str, provider: str, config: ProviderConfig
    ) -> type[BaseModel]:
        if config.factory is None:
            raise ValueError("configured provider requires `factory`")
        factory_cls = self._factory_cache.get(provider)
        if factory_cls is None:
            factory_cls = self._load_factory_class(provider, config)
            self._factory_cache[provider] = factory_cls
        if factory_cls.model_type != model_type:
            raise ValueError(
                f"Factory class {factory_cls.__name__!r} for provider "
                f"{provider!r} implements model type "
                f"{factory_cls.model_type!r}, not {model_type!r}"
            )
        if factory_cls.provider != provider:
            raise ValueError(
                f"Factory class {factory_cls.__name__!r} declares provider "
                f"{factory_cls.provider!r}, which does not match catalog "
                f"provider {provider!r}; refusing to relabel it"
            )
        return factory_cls

    def _load_factory_class(
        self, provider: str, config: ProviderConfig
    ) -> type[BaseModel]:
        if config.factory is None:
            raise ValueError("configured provider requires `factory`")
        # Builtin autoload must complete in the global registry before the
        # import-registration context is entered.
        _ = list(model_registry.keys())
        with _scoped_import_catalog(self):
            loaded = load_factory(config.factory)
        if not isinstance(loaded, type):
            raise TypeError(
                f"Factory {config.factory!r} for provider {provider!r} must "
                "resolve directly to a model class"
            )
        factory_cls = loaded
        if not issubclass(factory_cls, BaseModel):
            raise TypeError(
                f"Factory {config.factory!r} for provider {provider!r} must "
                "resolve to a BaseModel subclass"
            )
        if not getattr(factory_cls, "model_type", None) or not getattr(
            factory_cls, "provider", None
        ):
            raise ValueError(
                f"Factory class {factory_cls.__name__!r} must define "
                "`model_type` and `provider`."
            )
        if factory_cls.provider != provider:
            raise ValueError(
                f"Factory class {factory_cls.__name__!r} declares provider "
                f"{factory_cls.provider!r}, which does not match catalog "
                f"provider {provider!r}; refusing to relabel it"
            )
        codex_namespace = _codex_api_key_namespace(factory_cls)
        if config.base_url is None and config.api_key_env is None:
            if codex_namespace:
                # A factory Codex class declaring its own env must use the
                # env/Bearer resolver, not the inherited auth-file one.
                # Return a per-catalog subclass; never mutate the shared
                # factory class.
                namespace = {
                    "provider": provider,
                    "model_type": factory_cls.model_type,
                    "__module__": __name__,
                    "__qualname__": (
                        "ProviderCatalog."
                        f"{_class_name(provider, factory_cls.model_type, factory_cls)}"
                    ),
                }
                namespace.update(codex_namespace)
                return type(
                    _class_name(provider, factory_cls.model_type, factory_cls),
                    (factory_cls,),
                    namespace,
                )
            return factory_cls
        # Overrides must be per-catalog and survive restore without
        # mutating the shared factory class: use a named local subclass.
        is_oauth = _is_oauth_class(factory_cls)
        namespace: dict[str, Any] = {
            "provider": provider,
            "model_type": factory_cls.model_type,
            "__module__": __name__,
            "__qualname__": (
                "ProviderCatalog."
                f"{_class_name(provider, factory_cls.model_type, factory_cls)}"
            ),
            "provider_auth_method": (
                "api_key"
                if config.api_key_env is not None
                else "oauth"
                if is_oauth
                else "api_key"
            ),
        }
        namespace.update(_explicit_api_key_namespace(factory_cls, config))
        namespace.update(codex_namespace)
        if config.base_url is not None:
            namespace["base_url"] = config.base_url
            namespace["base_url_env"] = None
        if config.api_key_env is not None:
            namespace["api_key_env"] = config.api_key_env
            namespace["display_name"] = provider
        return type(
            _class_name(provider, factory_cls.model_type, factory_cls),
            (factory_cls,),
            namespace,
        )

    # -- construction / restoration -----------------------------------------

    def _chain(self, provider: str) -> list[tuple[str, ProviderConfig]]:
        """Configured chain for ``provider``, leaf first, root last."""
        chain: list[tuple[str, ProviderConfig]] = []
        seen: set[str] = set()
        current: str | None = provider
        while current is not None and current in self._configured:
            if current in seen:
                cycle = " -> ".join([name for name, _ in chain] + [current])
                raise ValueError(f"Provider base cycle detected: {cycle}")
            seen.add(current)
            config = self._configured[current]
            chain.append((current, config))
            current = config.base
        return chain

    def create_model(
        self, model_type: str, provider: str, model_id: str, **kwargs
    ) -> BaseModel:
        explicit_catalog = kwargs.pop("provider_catalog", None)
        context = explicit_catalog if explicit_catalog is not None else self
        if explicit_catalog is not None and explicit_catalog is not self:
            raise ValueError(
                "`provider_catalog` passed to create_model must be this catalog"
            )
        config = self._configured.get(provider)
        model_cls = self.get_model_class(model_type, provider)
        call_kwargs: dict[str, Any] = {}
        if config is not None:
            # Inherited base-chain options apply first; leaf options win.
            for _name, chain_config in reversed(self._chain(provider)):
                call_kwargs.update(copy.deepcopy(dict(chain_config.options)))
            if config.base_url is not None and _constructor_accepts(
                model_cls, "base_url"
            ):
                call_kwargs.setdefault("base_url", config.base_url)
            if config.api_key_env is not None and _constructor_accepts(
                model_cls, "api_key_env"
            ):
                call_kwargs.setdefault("api_key_env", config.api_key_env)
        call_kwargs.update(kwargs)
        account_auth = (
            config is not None
            and _is_oauth_class(model_cls)
            and "credential_resolver" not in call_kwargs
            and "auth_file" not in call_kwargs
        )
        if account_auth:
            call_kwargs["credential_resolver"] = self._account_resolver(provider)
        with context.use():
            instance = model_cls(model_id=model_id, **call_kwargs)
        self._validate_instance_identity(instance, provider, model_type)
        if account_auth:
            instance.__dict__["_provider_account_auth"] = True
        self._attach_source(instance, provider, config)
        return instance

    @staticmethod
    def _validate_instance_identity(
        instance: Any, provider: str, model_type: str
    ) -> None:
        actual_provider = getattr(instance, "provider", None)
        actual_type = getattr(instance, "model_type", None)
        if actual_provider != provider or actual_type != model_type:
            cleanup_error = _cleanup_instance(instance)
            error = ValueError(
                f"Constructed model for provider `{provider}` and type "
                f"`{model_type}` does not match the resolved alias "
                f"(actual provider={actual_provider!r}, "
                f"model_type={actual_type!r}); refusing to use it"
            )
            if cleanup_error is not None:
                error.add_note(f"Resource cleanup failed: {cleanup_error!r}")
            raise error

    def from_serialized(
        self, provider: str, model_type: str, state: Mapping[str, Any]
    ) -> BaseModel:
        if not isinstance(state, Mapping):
            raise TypeError(
                f"`state` must be a mapping, given `{type(state).__name__}`"
            )
        # Validate against the trusted catalog before resolving any class,
        # so a tampered factory path in state is never imported.
        self._validate_source(provider, state)
        model_cls = self.get_model_class(model_type, provider)
        runtime_state = dict(state)
        if runtime_state.get("_provider_account_auth"):
            runtime_state["credential_resolver"] = self._account_resolver(provider)
        with self.use():
            instance = model_cls.from_serialized(runtime_state)
        self._validate_instance_identity(instance, provider, model_type)
        return instance

    def _account_resolver(self, provider: str):
        from msgflux.accounts.store import Accounts  # noqa: PLC0415

        return Accounts(provider_catalog=self).resolver(provider)

    # -- source metadata ------------------------------------------------------

    def _attach_source(
        self, instance: BaseModel, provider: str, config: ProviderConfig | None
    ) -> None:
        if config is not None:
            instance.__dict__["_provider_source"] = self._expected_source(provider)

    def _expected_source(self, provider: str) -> dict[str, Any] | None:
        config = self._configured.get(provider)
        if config is None:
            return None
        chain = [
            {
                "provider": name,
                "base": chain_config.base,
                "factory": chain_config.factory,
                "base_url": chain_config.base_url,
                "api_key_env": chain_config.api_key_env,
                "options": copy.deepcopy(dict(chain_config.options)),
            }
            for name, chain_config in self._chain(provider)
        ]
        leaf = chain[0]
        return {
            "provider": provider,
            "base": leaf["base"],
            "factory": leaf["factory"],
            "base_url": leaf["base_url"],
            "api_key_env": leaf["api_key_env"],
            "options": copy.deepcopy(leaf["options"]),
            "chain": chain,
        }

    def _validate_source(self, provider: str, state: Mapping[str, Any]) -> None:
        """Validate serialized source metadata against this trusted catalog."""
        source = state.get("_provider_source")
        config = self._configured.get(provider)
        if config is not None:
            expected = self._expected_source(provider)
            if source is None:
                raise ValueError(
                    f"Serialized state for provider `{provider}` is missing "
                    "trusted provider source metadata; restore requires the "
                    "configured provider catalog that created it"
                )
            if dict(source) != expected:
                raise ValueError(
                    f"Serialized state for provider `{provider}` does not "
                    "match the trusted catalog definition; the provider "
                    "definition is missing or changed"
                )
            return
        if _source_is_configured(source):
            raise ValueError(
                f"Serialized state for provider `{provider}` requires a "
                "configured provider catalog, but this catalog has no "
                "definition for it"
            )
        if source is not None and not isinstance(source, Mapping):
            raise ValueError(
                f"Serialized state for provider `{provider}` has invalid "
                "provider source metadata"
            )
        if source is not None and source.get("provider") not in (None, provider):
            raise ValueError(
                f"Serialized state for provider `{provider}` carries source "
                "metadata for a different provider"
            )


def validate_serialized_source(
    provider: str, state: Mapping[str, Any], catalog: ProviderCatalog | None
) -> None:
    """Validate ``state`` source metadata for a restore path.

    With a catalog, validation is delegated to it. Without one, a state
    carrying configured-provider metadata cannot be trusted or resolved.
    """
    if catalog is not None:
        catalog._validate_source(provider, state)
        return
    source = state.get("_provider_source") if isinstance(state, Mapping) else None
    if _source_is_configured(source):
        raise ValueError(
            f"Serialized state for provider `{provider}` requires a trusted "
            "ProviderCatalog; none was supplied"
        )
