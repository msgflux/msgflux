"""Safe, immutable registration metadata without constructing Agent sessions."""

import msgspec
import pytest

from msgflux.runtime.service import AgentConfiguration, AgentService, SQLiteServiceStore
from msgflux.runtime.service.local.agent_inspection import manifest_agent_configuration
from msgflux.runtime.service.local.manifest import load_service_manifest


@pytest.fixture
def service():
    result = AgentService(
        store=SQLiteServiceStore(), default_agent="coding", default_account="work"
    )
    yield result
    result.store.close()


def unexpected_factory(_thread):
    raise AssertionError("Inspection must not execute the session factory")


def test_inspection_preserves_unknown_custom_configuration(service):
    service.register("coding", unexpected_factory)
    result = service.inspect_agent("coding")
    assert result.agent_id == "coding" and result.is_default
    assert result.account_name == "work" and result.configuration is None
    assert service.threads() == ()
    with pytest.raises(KeyError):
        service.inspect_agent("missing")


def test_registration_freezes_metadata_and_validates_before_mutating(service):
    permissions = ["read"]
    metadata = AgentConfiguration(model="provider/model", permissions=permissions)
    service.register("coding", unexpected_factory, configuration=metadata)
    permissions.append("write")
    assert service.inspect_agent("coding").configuration.permissions == ("read",)
    with pytest.raises(AttributeError):
        service.inspect_agent("coding").configuration.model = "changed/model"
    with pytest.raises(TypeError, match="AgentConfiguration"):
        service.register("invalid", unexpected_factory, configuration={"model": "x"})
    with pytest.raises(msgspec.ValidationError):
        service.register(
            "invalid", unexpected_factory, configuration=AgentConfiguration(model=42)
        )
    assert service.agents() == ("coding",)


@pytest.mark.parametrize(
    "model, preset, expected",
    [
        ("openai-codex/example", "coding", ("bash", "read", "apply_patch")),
        ("other/example", "coding", ("bash", "read", "edit", "write", "delete")),
        (
            "openai-codex/example",
            "explorer",
            ("read", "ls", "glob", "grep", "web_fetch", "web_search"),
        ),
    ],
)
def test_manifest_metadata_reuses_provider_permission_and_preset_resolution(
    tmp_path, model, preset, expected
):
    config = tmp_path / "config.toml"
    content = f"[defaults]\nagent='coding'\nmodel='{model}'\n[agents.coding]\npreset='{preset}'\n"
    config.write_text(content)
    manifest = load_service_manifest(config)
    result = manifest_agent_configuration(manifest, "coding")
    assert result.model == model and result.reasoning_effort == "medium"
    assert result.tools_active == expected
    assert result.tools_deferred == ()
    assert result.prompt_source == ("default" if preset == "coding" else "preset")
    assert result.preset == preset and result.prompt_file is None
    assert result.config_digest == manifest.digest
    assert result.permissions == (
        ("read",) if preset == "explorer" else ("read", "write", "execute")
    )
    # The projection reads the loaded manifest, not mutable files on disk.
    config.write_text("not valid toml")
    assert manifest_agent_configuration(manifest, "coding") == result


def test_custom_manifest_metadata_does_not_invent_coding_defaults(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text(
        "[defaults]\nagent='custom'\n[agents.custom]\nfactory='unknown_module:create'\n"
    )
    result = manifest_agent_configuration(load_service_manifest(config), "custom")
    assert result.factory == "unknown_module:create"
    assert result.model is None and result.permissions is None
    assert result.tools_active is None and result.prompt_source is None
