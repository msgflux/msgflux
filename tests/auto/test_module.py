import json
import sys
import types
from pathlib import Path

import pytest

import msgflux as mf
from msgflux import nn
from msgflux.auto import AutoModule
from msgflux.auto.cache import AutoModuleCache
from msgflux.auto.exceptions import (
    AutoModuleConfigurationError,
    AutoModuleSecurityError,
)
from msgflux.models.base import BaseModel
from msgflux.models.providers.fake import FakeChatCompletion, FakeModelExecutionError
from msgflux.models.registry import model_registry


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _github_cache_path(cache_dir: Path, repo_id: str, revision: str = "main") -> Path:
    return AutoModuleCache(cache_dir).module_path("github", repo_id, revision)


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


class ExplodingChatCompletion(BaseModel):
    model_type = "chat_completion"
    provider = "exploding_auto_module_test"

    def __init__(self, model_id: str = "boom") -> None:
        self.model_id = model_id

    def _initialize(self):
        raise RuntimeError("original provider should not initialize")

    def __call__(self, *args, **kwargs):
        raise RuntimeError("not expected")


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
    assert agent.factory_config.models["default"].path == "generator.model"


def test_auto_module_instance_create_supports_model_object_override(tmp_path):
    module_root = _github_cache_path(tmp_path, "owner/repo")
    _write_auto_module(module_root, state=_agent_state())
    ref = AutoModule("owner/repo", cache_dir=tmp_path, local_files_only=True)
    override = FakeChatCompletion(model_id="object-override")

    agent = ref.create(trust_remote_code=True, models={"default": override})

    assert agent.model is override


def test_auto_module_model_replacement_skips_original_provider_init(tmp_path):
    original_chat_completion = model_registry.get("chat_completion")
    model_registry["chat_completion"] = dict(original_chat_completion or {})
    model_registry["chat_completion"][
        "exploding_auto_module_test"
    ] = ExplodingChatCompletion
    try:
        state = _agent_state()
        state["generator.model"] = {
            "msgflux_type": "model",
            "provider": "exploding_auto_module_test",
            "model_type": "chat_completion",
            "state": {"model_id": "would-explode"},
        }
        module_root = _github_cache_path(tmp_path, "owner/repo")
        _write_auto_module(module_root, state=state)

        agent = AutoModule.create(
            "owner/repo",
            cache_dir=tmp_path,
            local_files_only=True,
            trust_remote_code=True,
            models={"default": "fake/replacement"},
        )

        assert agent.model.provider == "fake"
        assert agent.model.model_id == "replacement"
    finally:
        if original_chat_completion is None:
            model_registry.pop("chat_completion", None)
        else:
            model_registry["chat_completion"] = original_chat_completion


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


def test_auto_module_get_class_uses_standard_module_hook(tmp_path):
    module_root = _github_cache_path(tmp_path, "owner/repo")
    _write_auto_module(
        module_root,
        manifest={
            "schema_version": 1,
            "msgflux_version": ">=0.0.0",
            "entrypoint": "module.py",
            "state": "state.json",
        },
        module_source="""
class DefaultWorkflow:
    pass


class PlannerWorkflow:
    pass


def get_class(name=None):
    if name == "planner":
        return PlannerWorkflow
    return DefaultWorkflow
""",
        state={},
    )
    ref = AutoModule("owner/repo", cache_dir=tmp_path, local_files_only=True)

    assert ref.get_class(trust_remote_code=True).__name__ == "DefaultWorkflow"
    assert ref.get_class("planner", trust_remote_code=True).__name__ == (
        "PlannerWorkflow"
    )


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


def test_auto_module_sibling_imports_are_isolated_between_modules(tmp_path):
    one_root = _github_cache_path(tmp_path, "owner/one")
    two_root = _github_cache_path(tmp_path, "owner/two")
    for root, marker in [(one_root, "one"), (two_root, "two")]:
        _write_auto_module(
            root,
            manifest={
                "schema_version": 1,
                "msgflux_version": ">=0.0.0",
                "factory": "module.py:create",
                "files": ["helper.py"],
            },
            state=None,
            module_source="""
from helper import marker


def create():
    return marker
""",
        )
        (root / "helper.py").write_text(f'marker = "{marker}"', encoding="utf-8")

    assert AutoModule.create(
        "owner/one",
        cache_dir=tmp_path,
        local_files_only=True,
        trust_remote_code=True,
    ) == "one"
    assert AutoModule.create(
        "owner/two",
        cache_dir=tmp_path,
        local_files_only=True,
        trust_remote_code=True,
    ) == "two"
    assert "helper" not in sys.modules


def test_auto_module_load_into_applies_state_and_replacements(tmp_path):
    module_root = _github_cache_path(tmp_path, "owner/repo")
    _write_auto_module(module_root, state=_agent_state())
    target = nn.Agent(
        name="target",
        model=mf.Model.chat_completion("fake/placeholder"),
        instructions="Target instructions",
    )

    result = AutoModule.load_into(
        "owner/repo",
        target,
        cache_dir=tmp_path,
        local_files_only=True,
        models={"default": "fake/load-into"},
    )

    assert result is target
    assert target.name == "exported-agent"
    assert str(target.instructions) == "State instructions"
    assert target.model.model_id == "load-into"


def test_auto_module_export_writes_state_manifest_and_stub(tmp_path):
    workflow = nn.Agent(
        name="exported",
        model=mf.Model.chat_completion("fake/default"),
        instructions="Export me",
    )

    output_dir = AutoModule.export(workflow, tmp_path / "dist")

    state = json.loads((output_dir / "state.json").read_text(encoding="utf-8"))
    manifest = json.loads((output_dir / "module.json").read_text(encoding="utf-8"))

    assert (output_dir / "module.py").read_text(encoding="utf-8") == ""
    assert state["instructions"] == "Export me"
    assert state["generator.model"]["provider"] == "fake"
    assert manifest["entrypoint"] == "module.py"
    assert manifest["state"] == "state.json"
    assert manifest["models"]["generator-model"]["path"] == "generator.model"
    assert manifest["models"]["generator-model"]["provider"] == "fake"
    assert manifest["models"]["generator-model"]["model_type"] == "chat_completion"
    assert manifest["models"]["generator-model"]["model_id"] == "default"


def test_auto_module_loads_exported_bundle_from_local_source(tmp_path):
    workflow = nn.Agent(
        name="exported",
        model=mf.Model.chat_completion("fake/default"),
        instructions="Export me",
    )
    output_dir = AutoModule.export(workflow, tmp_path / "dist")
    (output_dir / "module.py").write_text(
        """
import msgflux as mf
from msgflux import nn


def create():
    return nn.Agent(
        name="placeholder",
        model=mf.Model.chat_completion("fake/placeholder"),
        instructions="Placeholder",
    )
""",
        encoding="utf-8",
    )

    agent = AutoModule.create(
        f"local://{output_dir}",
        trust_remote_code=True,
        models={"generator-model": "fake/local-user"},
    )

    assert agent.name == "exported"
    assert str(agent.instructions) == "Export me"
    assert agent.model.model_id == "local-user"


def test_auto_module_load_into_exported_bundle_without_module_py_code(tmp_path):
    workflow = nn.Agent(
        name="exported",
        model=mf.Model.chat_completion("fake/default"),
        instructions="Export me",
    )
    output_dir = AutoModule.export(workflow, tmp_path / "dist")
    target = nn.Agent(
        name="target",
        model=mf.Model.chat_completion("fake/placeholder"),
        instructions="Target",
    )

    AutoModule.load_into(
        f"local://{output_dir}",
        target,
        models={"generator-model": "fake/manual-load"},
    )

    assert target.name == "exported"
    assert str(target.instructions) == "Export me"
    assert target.model.model_id == "manual-load"


def test_auto_module_cache_sanitizes_revision_paths(tmp_path):
    cache_root = tmp_path / "cache"
    module_path = AutoModuleCache(cache_root).module_path(
        "github",
        "owner/repo",
        "../../../../outside",
    )

    assert module_path.resolve().is_relative_to(cache_root.resolve())


def test_auto_module_check_requirements_does_not_download_declared_files(tmp_path):
    module_root = _github_cache_path(tmp_path, "owner/repo")
    _write_auto_module(
        module_root,
        manifest={
            "schema_version": 1,
            "msgflux_version": ">=0.0.0",
            "factory": "module.py:create",
            "files": ["missing-helper.py"],
        },
        state=None,
        module_source="""
def create():
    return None
""",
    )
    ref = AutoModule("owner/repo", cache_dir=tmp_path, local_files_only=True)

    info = ref.check_requirements()

    assert info["config"].files == ["missing-helper.py"]


def test_auto_module_factory_type_error_is_not_retried(tmp_path):
    module_root = _github_cache_path(tmp_path, "owner/repo")
    _write_auto_module(
        module_root,
        state=None,
        module_source="""
def create(config=None, module_path=None):
    calls_path = module_path / "calls.txt"
    calls = int(calls_path.read_text()) if calls_path.exists() else 0
    calls_path.write_text(str(calls + 1))
    raise TypeError("unexpected keyword argument raised inside factory")
""",
    )

    with pytest.raises(
        TypeError, match="unexpected keyword argument raised inside factory"
    ):
        AutoModule.create(
            "owner/repo",
            cache_dir=tmp_path,
            local_files_only=True,
            trust_remote_code=True,
        )

    assert (module_root / "calls.txt").read_text() == "1"


def test_auto_module_import_failure_cleans_sys_modules(tmp_path):
    module_root = _github_cache_path(tmp_path, "owner/repo")
    _write_auto_module(
        module_root,
        state=None,
        module_source='raise RuntimeError("boom")',
    )

    with pytest.raises(AutoModuleConfigurationError, match="boom"):
        AutoModule.create(
            "owner/repo",
            cache_dir=tmp_path,
            local_files_only=True,
            trust_remote_code=True,
        )

    assert "msgflux_auto_owner_repo_main.module" not in sys.modules


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
