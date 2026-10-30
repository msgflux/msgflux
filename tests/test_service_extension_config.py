"""Service extension selection stays lazy and isolated across coding agents."""

import sys

import pytest

from msgflux.coding.config import CodingAgentCatalog, coding_agent_options
from msgflux.coding.factory import create_coding_session_factory
from msgflux.nn.extensions.base import AgentExtension
from msgflux.nn.extensions.discovery import AgentExtensionCatalog, ExtensionConfig
from msgflux.runtime.service import ServiceThread
from msgflux.runtime.service.local.manifest import load_service_manifest

_FACTORY = "tests.test_service_extension_config:RecordingExtension"


class RecordingExtension(AgentExtension):
    created = []
    events = []

    def __init__(self, label: str, fail: bool = False):
        super().__init__("factory-default-name")
        self.label = label
        self.fail = fail
        self.removed = False
        self.created.append(self)

    def on_register(self, agent):
        self.events.append((agent.get_module_name(), self.name, self.label))
        if self.fail:
            raise RuntimeError("extension registration failed")

    def on_remove(self, _agent):
        self.removed = True


class _Model:
    model_type = "chat_completion"

    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


@pytest.fixture(autouse=True)
def reset_extensions(monkeypatch):
    monkeypatch.setattr(RecordingExtension, "created", [])
    monkeypatch.setattr(RecordingExtension, "events", [])


def test_manifest_extension_selection_does_not_import_factories(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text(
        "[defaults]\nagent='coding'\nmodel='openai/model'\n"
        "[extensions.audit]\nfactory='never_imported_extension:Audit'\n"
        "[extensions.audit.options]\nlabel='configured'\n"
        "[agents.coding]\npreset='coding'\nextensions=['audit']\n"
        "[agents.custom]\nfactory='another_unimported_factory:make'\n"
        "extensions=['audit']\n"
    )
    manifest = load_service_manifest(config)
    assert manifest.coding_catalog.agents["coding"].extensions == ("audit",)
    assert manifest.agents["custom"].extensions == ["audit"]
    assert (
        manifest.extension_catalog.resolve("audit").factory
        == "never_imported_extension:Audit"
    )
    assert "never_imported_extension" not in sys.modules
    assert "another_unimported_factory" not in sys.modules


@pytest.mark.parametrize(
    "names", ["['missing']", "['audit','audit']", "['']", "['   ']"]
)
def test_manifest_rejects_unknown_duplicate_empty_selectors_without_imports(
    tmp_path, names
):
    config = tmp_path / "config.toml"
    config.write_text(
        "[defaults]\nagent='coding'\nmodel='openai/model'\n"
        "[extensions.audit]\nfactory='never_imported_extension:Audit'\n"
        f"[agents.coding]\npreset='coding'\nextensions={names}\n"
    )
    with pytest.raises(ValueError):
        load_service_manifest(config)
    assert "never_imported_extension" not in sys.modules


@pytest.mark.parametrize("names", [("audit", "audit"), ("",), ("   ",)])
def test_python_coding_options_validate_extension_selectors(names):
    with pytest.raises(ValueError, match="extension"):
        coding_agent_options("coding", "openai/model", extensions=names)


@pytest.mark.asyncio
async def test_coding_factory_extensions_are_ordered_fresh_and_not_inherited(
    tmp_path, monkeypatch
):
    from msgflux.coding import factory

    models = []

    def create_model(_options):
        model = _Model()
        models.append(model)
        return model

    monkeypatch.setattr(factory, "_create_model", create_model)
    parent = coding_agent_options(
        "coding",
        "openai/model",
        active=("read",),
        subagents=("child",),
        extensions=("first", "second"),
    )
    child = coding_agent_options(
        "child", "openai/model", active=("read",), extensions=("child-only",)
    )
    catalog = AgentExtensionCatalog(
        {
            "first": ExtensionConfig(factory=_FACTORY, options={"label": "one"}),
            "second": ExtensionConfig(factory=_FACTORY, options={"label": "two"}),
            "child-only": ExtensionConfig(factory=_FACTORY, options={"label": "child"}),
        }
    )
    make_session = create_coding_session_factory(
        options=parent,
        catalog=CodingAgentCatalog({"coding": parent, "child": child}),
        config_path=tmp_path / "config.toml",
        runtime_dir=tmp_path / "runtime",
        agent_dir=tmp_path / "agent-state",
        extension_catalog=catalog,
    )
    assert RecordingExtension.created == []
    assert models == []
    sessions = []
    try:
        for number in range(2):
            sessions.append(
                await make_session(
                    ServiceThread(f"thread-{number}", "coding", str(tmp_path))
                )
            )
        assert (
            RecordingExtension.events
            == [
                ("child", "child-only", "child"),
                ("coding", "first", "one"),
                ("coding", "second", "two"),
            ]
            * 2
        )
        assert len({id(extension) for extension in RecordingExtension.created}) == 6
        assert (
            sessions[0].agent.extensions["first"]
            is not sessions[1].agent.extensions["first"]
        )
        assert not sessions[0].agent.has_extension("child-only")
    finally:
        for session in sessions:
            await session.on_close()
    assert all(extension.removed for extension in RecordingExtension.created)
    assert all(model.closed for model in models)


@pytest.mark.asyncio
async def test_coding_extension_registration_failure_cleans_prior_instances_and_models(
    tmp_path, monkeypatch
):
    from msgflux.coding import factory

    model = _Model()
    monkeypatch.setattr(factory, "_create_model", lambda _options: model)
    options = coding_agent_options(
        "coding", "openai/model", active=("read",), extensions=("good", "bad")
    )
    catalog = AgentExtensionCatalog(
        {
            "good": ExtensionConfig(factory=_FACTORY, options={"label": "good"}),
            "bad": ExtensionConfig(
                factory=_FACTORY, options={"label": "bad", "fail": True}
            ),
        }
    )
    make_session = create_coding_session_factory(
        options=options,
        catalog=CodingAgentCatalog({"coding": options}),
        config_path=tmp_path / "config.toml",
        runtime_dir=tmp_path / "runtime",
        agent_dir=tmp_path / "agent-state",
        extension_catalog=catalog,
    )
    with pytest.raises(RuntimeError, match="registration failed"):
        await make_session(ServiceThread("thread-1", "coding", str(tmp_path)))
    assert len(RecordingExtension.created) == 2
    assert all(extension.removed for extension in RecordingExtension.created)
    assert model.closed
