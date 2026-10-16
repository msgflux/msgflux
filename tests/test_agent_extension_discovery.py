"""Installed extension metadata stays separate from plugin construction."""

from inspect import CORO_CLOSED, getcoroutinestate
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import msgspec
import pytest

from msgflux.nn.extensions.base import AgentExtension
from msgflux.nn.extensions.discovery import (
    GROUP,
    AgentExtensionCatalog,
    ExtensionConfig,
    ExtensionInfo,
)
from msgflux.utils.imports import load_factory, validate_factory_path


def _entry(name, factory, *, distribution="test-extensions", version="1.0"):
    module, _, attribute = factory.partition(":")
    return SimpleNamespace(
        name=name,
        module=module,
        attr=attribute,
        dist=SimpleNamespace(metadata={"Name": distribution}, version=version),
    )


@pytest.fixture
def plugins(monkeypatch):
    entries = []
    imports = []
    factories = {}

    def discover(*, group):
        assert group == GROUP
        return tuple(entries)

    def import_module(name):
        imports.append(name)
        if name not in factories:
            raise ModuleNotFoundError(name)
        module = ModuleType(name)
        module.build = factories[name]
        return module

    monkeypatch.setattr(
        "msgflux.nn.extensions.discovery.metadata.entry_points", discover
    )
    monkeypatch.setattr("msgflux.utils.imports.importlib.import_module", import_module)
    return entries, imports, factories


class _TestExtension(AgentExtension):
    def __init__(self, *, label, settings=None):
        super().__init__("internal-name")
        self.label = label
        self.settings = settings


def test_config_uses_strict_schema_and_isolated_option_defaults():
    first = ExtensionConfig()
    second = ExtensionConfig()
    first.options["limit"] = 1
    assert second.options == {}
    with pytest.raises(msgspec.ValidationError, match="unknown field"):
        msgspec.convert({"unexpected": True}, type=ExtensionConfig)


def test_listing_resolves_metadata_without_importing_plugins(plugins):
    entries, imports, _factories = plugins
    entries.extend(
        [
            _entry("zeta", "plugins.zeta:build"),
            _entry("alpha", "plugins.alpha:build"),
        ]
    )
    catalog = AgentExtensionCatalog(
        {"custom": ExtensionConfig(factory="missing.custom:build")}
    )
    assert catalog.list() == (
        ExtensionInfo("alpha", "plugins.alpha:build", "test-extensions", "1.0"),
        ExtensionInfo("custom", "missing.custom:build"),
        ExtensionInfo("zeta", "plugins.zeta:build", "test-extensions", "1.0"),
    )
    assert catalog.resolve("alpha").distribution == "test-extensions"
    assert imports == []


def test_validation_loads_only_selected_factory_without_constructing(plugins):
    entries, imports, factories = plugins
    constructed = []

    def build(*, label):
        constructed.append(label)
        return _TestExtension(label=label)

    entries.extend(
        [
            _entry("selected", "plugins.selected:build"),
            _entry("broken", "missing:build"),
        ]
    )
    factories["plugins.selected"] = build
    catalog = AgentExtensionCatalog(
        {"selected": ExtensionConfig(options={"label": "hello"})}
    )
    catalog.validate(["selected"])
    assert imports == ["plugins.selected"]
    assert constructed == []
    extension = catalog.create("selected")
    assert extension.label == "hello"
    assert extension.name == "selected"
    assert constructed == ["hello"]


@pytest.mark.parametrize(
    ("options", "message"),
    [({}, "missing a required"), ({"label": "x", "unknown": 1}, "unexpected")],
)
def test_validation_rejects_invalid_options_without_invoking_factory(
    plugins, options, message
):
    _entries, _imports, factories = plugins
    factories["plugins.selected"] = _TestExtension
    catalog = AgentExtensionCatalog(
        {"selected": ExtensionConfig(factory="plugins.selected:build", options=options)}
    )
    with pytest.raises(ValueError, match=message) as failure:
        catalog.validate(["selected"])
    assert "selected" in str(failure.value)
    assert "plugins.selected:build" in str(failure.value)
    assert isinstance(failure.value.__cause__, TypeError)


def test_create_uses_fresh_instances_options_and_configured_aliases(plugins):
    _entries, _imports, factories = plugins
    factories["plugins.selected"] = _TestExtension
    config = ExtensionConfig(
        factory="plugins.selected:build",
        options={"label": "before", "settings": {"values": [1]}},
    )
    catalog = AgentExtensionCatalog({"first": config, "second": config})
    config.options["label"] = "after"
    first = catalog.create("first")
    first.settings["values"].append(2)
    second = catalog.create("second")
    again = catalog.create("first")
    assert first is not again and first is not second
    assert (first.name, second.name, again.name) == ("first", "second", "first")
    assert second.label == "before"
    assert second.settings == again.settings == {"values": [1]}
    assert config.options["settings"] == {"values": [1]}


@pytest.mark.parametrize("operation", ["resolve", "list"])
def test_duplicate_entry_points_require_explicit_factory(plugins, operation):
    entries, imports, _factories = plugins
    first = _entry("shared", "plugins.first:build", distribution="first")
    second = _entry("shared", "plugins.second:build", distribution="second")
    entries[:] = [second, first]
    catalog = AgentExtensionCatalog()
    with pytest.raises(ValueError, match="multiple installed factories") as failure:
        catalog.resolve("shared") if operation == "resolve" else catalog.list()
    message = str(failure.value)
    assert message.index("plugins.first:build") < message.index("plugins.second:build")
    assert "configure an explicit factory" in message
    assert imports == []
    configured = AgentExtensionCatalog(
        {"shared": ExtensionConfig(factory="chosen.custom:build")}
    )
    assert configured.list() == (ExtensionInfo("shared", "chosen.custom:build"),)
    assert imports == []


def test_unknown_extension_has_concrete_error(plugins):
    catalog = AgentExtensionCatalog({"missing": ExtensionConfig()})
    with pytest.raises(ValueError, match="Unknown Agent extension 'missing'"):
        catalog.resolve("missing")


def test_new_catalog_observes_new_metadata_without_global_cache(plugins):
    entries, _imports, _factories = plugins
    first = AgentExtensionCatalog()
    entries.append(_entry("added", "plugins.added:build"))
    assert first.list() == ()
    assert AgentExtensionCatalog().resolve("added").factory == "plugins.added:build"


@pytest.mark.parametrize("result", [None, object()])
def test_create_rejects_non_extension_results(plugins, result):
    _entries, _imports, factories = plugins
    factories["plugins.bad"] = lambda: result
    catalog = AgentExtensionCatalog(
        {"bad": ExtensionConfig(factory="plugins.bad:build")}
    )
    with pytest.raises(ValueError, match="must return an AgentExtension") as failure:
        catalog.create("bad")
    assert isinstance(failure.value.__cause__, TypeError)


def test_factory_failure_preserves_name_path_and_original_cause(plugins):
    _entries, _imports, factories = plugins
    original = RuntimeError("invalid plugin settings")

    def fail():
        raise original

    factories["plugins.fail"] = fail
    catalog = AgentExtensionCatalog(
        {"failing": ExtensionConfig(factory="plugins.fail:build")}
    )
    catalog.validate(["failing"])
    with pytest.raises(ValueError, match=r"failing.*plugins.fail:build") as failure:
        catalog.create("failing")
    assert failure.value.__cause__ is original


def test_import_failure_preserves_name_and_factory(plugins):
    catalog = AgentExtensionCatalog(
        {"unavailable": ExtensionConfig(factory="plugins.missing:build")}
    )
    with pytest.raises(
        ValueError, match=r"unavailable.*plugins.missing:build"
    ) as failure:
        catalog.validate(["unavailable"])
    assert isinstance(failure.value.__cause__, ModuleNotFoundError)


@pytest.mark.parametrize(
    "factory", ["missing_separator", ":build", "module:", "bad :f"]
)
def test_factory_path_validation_does_not_import(plugins, factory):
    _entries, imports, _factories = plugins
    with pytest.raises(ValueError, match="factory must use"):
        AgentExtensionCatalog({"invalid": ExtensionConfig(factory=factory)})
    assert imports == []


def test_shared_factory_loader_accepts_nested_attributes(monkeypatch):
    module = ModuleType("nested")
    module.container = SimpleNamespace(build=_TestExtension)
    monkeypatch.setattr(
        "msgflux.utils.imports.importlib.import_module", lambda name: module
    )
    validate_factory_path("nested:container.build")
    assert load_factory("nested:container.build") is _TestExtension


def test_factory_cannot_rename_an_extension_bound_to_an_agent(plugins):
    _entries, _imports, factories = plugins
    extension = _TestExtension(label="bound")
    owner = Mock()
    extension._bind_agent(owner)
    factories["plugins.singleton"] = lambda: extension
    catalog = AgentExtensionCatalog(
        {"alias": ExtensionConfig(factory="plugins.singleton:build")}
    )
    with pytest.raises(ValueError, match="already bound to an Agent"):
        catalog.create("alias")
    assert extension.name == "internal-name"
    assert extension.agent is owner


@pytest.mark.parametrize("callable_object", [False, True])
def test_validation_rejects_async_factories_without_invoking_them(
    plugins, callable_object
):
    _entries, _imports, factories = plugins

    async def build():
        return _TestExtension(label="async")

    class AsyncFactory:
        __call__ = staticmethod(build)

    factories["plugins.async"] = AsyncFactory() if callable_object else build
    catalog = AgentExtensionCatalog(
        {"async": ExtensionConfig(factory="plugins.async:build")}
    )
    with pytest.raises(ValueError, match="factories must be synchronous"):
        catalog.validate(["async"])


def test_sync_factory_returning_coroutine_is_closed_before_rejection(plugins):
    _entries, _imports, factories = plugins

    async def build():
        return _TestExtension(label="async")

    pending = build()
    factories["plugins.async"] = lambda: pending
    catalog = AgentExtensionCatalog(
        {"async": ExtensionConfig(factory="plugins.async:build")}
    )
    with pytest.raises(ValueError, match="factories must be synchronous"):
        catalog.create("async")
    assert getcoroutinestate(pending) == CORO_CLOSED
