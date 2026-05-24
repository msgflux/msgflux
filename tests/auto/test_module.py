import json
import sys
import types
from pathlib import Path

import pytest

import msgflux as mf
from msgflux import nn
from msgflux.auto import AutoModule
from msgflux.auto.exceptions import (
    AutoModuleConfigurationError,
    AutoModuleSecurityError,
)
from msgflux.models.providers.fake import FakeChatCompletion, FakeModelExecutionError


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _github_cache_path(cache_dir: Path, repo_id: str, revision: str = "main") -> Path:
    return cache_dir / "github" / repo_id.replace("/", "--") / revision


def _write_auto_module(
    root: Path,
    *,
    manifest: dict | None = None,
    module_source: str | None = None,
    state: dict | None = None,
) -> None:
    _write_json(
        root / "module.json",
        manifest
        or {
            "schema_version": 1,
            "msgflux_version": ">=0.0.0",
            "factory": "module.py:create",
            "state": "state.json",
            "models": {"default": "generator.model"},
        },
    )
    (root / "module.py").write_text(
        module_source
        or """
import msgflux as mf
import msgflux.nn as nn


def create(config=None, module_path=None, **kwargs):
    agent = nn.Agent(
        name="factory-agent",
        model=mf.Model.chat_completion("fake/placeholder"),
        instructions="Factory instructions",
        **kwargs,
    )
    agent.factory_module_path = module_path
    agent.factory_config = config
    return agent
""",
        encoding="utf-8",
    )
    if state is not None:
        _write_json(root / "state.json", state)


def _agent_state() -> dict:
    agent = nn.Agent(
        name="exported-agent",
        model=mf.Model.chat_completion("fake/exported", custom_param="from-state"),
        instructions="State instructions",
    )
    return agent.state_dict()


def test_fake_chat_completion_serializes_and_never_executes():
    model = mf.Model.chat_completion("fake/placeholder", reason="test")

    serialized = model.serialize()

    assert serialized["msgflux_type"] == "model"
    assert serialized["provider"] == "fake"
    assert serialized["model_type"] == "chat_completion"
    assert serialized["state"]["model_id"] == "placeholder"
    assert serialized["state"]["kwargs"] == {"reason": "test"}
    with pytest.raises(FakeModelExecutionError):
        model()


def test_auto_module_create_loads_state_and_applies_model_overrides(tmp_path):
    module_root = _github_cache_path(tmp_path, "owner/repo")
    _write_auto_module(module_root, state=_agent_state())

    agent = AutoModule.create(
        "owner/repo",
        cache_dir=tmp_path,
        local_files_only=True,
        trust_remote_code=True,
        models={"default": "fake/override"},
    )

    assert agent.name == "exported-agent"
    assert str(agent.instructions) == "State instructions"
    assert agent.model.model_id == "override"
    assert agent.factory_module_path == module_root
    assert agent.factory_config.models == {"default": "generator.model"}


def test_auto_module_instance_create_supports_model_object_override(tmp_path):
    module_root = _github_cache_path(tmp_path, "owner/repo")
    _write_auto_module(module_root, state=_agent_state())
    ref = AutoModule("owner/repo", cache_dir=tmp_path, local_files_only=True)
    override = FakeChatCompletion(model_id="object-override")

    agent = ref.create(trust_remote_code=True, models={"default": override})

    assert agent.model is override


def test_auto_module_create_requires_trust_for_remote_factory(tmp_path):
    module_root = _github_cache_path(tmp_path, "owner/repo")
    _write_auto_module(module_root, state=_agent_state())

    with pytest.raises(AutoModuleSecurityError):
        AutoModule.create("owner/repo", cache_dir=tmp_path, local_files_only=True)


def test_auto_module_get_class_loads_remote_class_when_trusted(tmp_path):
    module_root = _github_cache_path(tmp_path, "owner/repo")
    _write_auto_module(
        module_root,
        manifest={
            "schema_version": 1,
            "msgflux_version": ">=0.0.0",
            "class": "module.py:RemoteAgent",
        },
        module_source="""
class RemoteAgent:
    pass
""",
    )
    ref = AutoModule("owner/repo", cache_dir=tmp_path, local_files_only=True)

    with pytest.raises(AutoModuleSecurityError):
        ref.get_class()

    cls = ref.get_class(trust_remote_code=True)
    assert cls.__name__ == "RemoteAgent"


def test_auto_module_get_class_allows_local_import_without_remote_trust(tmp_path):
    module_root = _github_cache_path(tmp_path, "owner/repo")
    _write_auto_module(
        module_root,
        manifest={
            "schema_version": 1,
            "msgflux_version": ">=0.0.0",
            "class": "msgflux.nn:Agent",
        },
    )
    ref = AutoModule("owner/repo", cache_dir=tmp_path, local_files_only=True)

    cls = ref.get_class()

    assert cls is nn.Agent


def test_auto_module_get_class_rejects_factory_only_manifest(tmp_path):
    module_root = _github_cache_path(tmp_path, "owner/repo")
    _write_auto_module(module_root, state=_agent_state())
    ref = AutoModule("owner/repo", cache_dir=tmp_path, local_files_only=True)

    with pytest.raises(AutoModuleConfigurationError, match="does not define `class`"):
        ref.get_class(trust_remote_code=True)


def test_auto_module_factory_can_import_sibling_files(tmp_path):
    module_root = _github_cache_path(tmp_path, "owner/repo")
    _write_auto_module(
        module_root,
        manifest={
            "schema_version": 1,
            "msgflux_version": ">=0.0.0",
            "factory": "module.py:create",
            "files": ["helper.py"],
        },
        state=None,
        module_source="""
from helper import marker


def create(config=None, module_path=None):
    return {"marker": marker}
""",
    )
    (module_root / "helper.py").write_text('marker = "from-helper"', encoding="utf-8")

    result = AutoModule.create(
        "owner/repo",
        cache_dir=tmp_path,
        local_files_only=True,
        trust_remote_code=True,
    )

    assert result == {"marker": "from-helper"}


def test_auto_module_supports_huggingface_source(monkeypatch, tmp_path):
    remote_root = tmp_path / "remote"
    _write_auto_module(remote_root, state=_agent_state())

    def hf_hub_download(**kwargs):
        return str(remote_root / kwargs["filename"])

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(hf_hub_download=hf_hub_download),
    )

    info = AutoModule("hf://owner/repo", cache_dir=tmp_path).check_requirements()

    assert info["source"] == "huggingface"
    assert info["config"].factory == "module.py:create"
