from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

from msgflux.utils.imports import AutoloadRegistry

if TYPE_CHECKING:
    from msgflux.models.base import BaseModel

model_registry = AutoloadRegistry("msgflux.models.providers")

# Set only around a trusted catalog factory import. This is intentionally
# separate from ``ProviderCatalog.use()``: entering a catalog for model
# resolution must not redirect unrelated ``@register_model`` decorators.
_import_catalog_var: ContextVar[Any] = ContextVar(
    "msgflux_import_catalog", default=None
)


def _current_import_catalog() -> Any:
    return _import_catalog_var.get()


@contextmanager
def _scoped_import_catalog(catalog: Any):
    token = _import_catalog_var.set(catalog)
    try:
        yield catalog
    finally:
        _import_catalog_var.reset(token)


def _validate_model_class(cls: type["BaseModel"]):
    model_type = getattr(cls, "model_type", None)
    provider = getattr(cls, "provider", None)
    if not model_type or not provider:
        raise ValueError(f"{cls.__name__} must define `model_type` and `provider`.")
    return model_type, provider


def register_model(
    cls: type["BaseModel"] | None = None,
    *,
    catalog: Any = None,
    replace: bool = False,
):
    """Register a model class globally or in one provider catalog.

    Supports ``@register_model``, ``@register_model()`` and
    ``@register_model(catalog=catalog)``. Re-registering the same class is
    idempotent; a different class for the same type/provider requires
    ``replace=True``.
    """
    if cls is None:

        def decorator(inner):
            return register_model(inner, catalog=catalog, replace=replace)

        return decorator

    model_type, provider = _validate_model_class(cls)
    target = catalog if catalog is not None else _current_import_catalog()
    if target is not None:
        target.register(cls, replace=replace)
        return cls

    # Avoid triggering AutoloadRegistry._ensure_loaded here: builtin provider
    # modules call this function while that autoload is in progress, and a
    # re-entrant autoload would loop. ``dict.setdefault`` does not trigger it.
    providers = model_registry.setdefault(model_type, {})
    existing = providers.get(provider)
    if existing is cls:
        return cls
    if existing is not None and not replace:
        raise ValueError(
            f"Provider `{provider}` for type `{model_type}` is already "
            f"registered as {existing.__name__!r}; pass replace=True to "
            "replace it explicitly."
        )
    providers[provider] = cls
    return cls
