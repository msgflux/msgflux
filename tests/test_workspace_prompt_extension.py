import asyncio
import json
import os
import re
from dataclasses import replace
from typing import Any
from unittest.mock import Mock

import msgspec
import pytest

from msgflux.models.response import ModelResponse
from msgflux.models.tool_call_agg import ToolCallAggregator
from msgflux.exceptions import TaskPauseRequestedError
from msgflux.data.stores import InMemoryCheckpointStore
from msgflux.nn import Agent
from msgflux.nn.extensions import WorkspacePromptExtension
from msgflux.runtime import (
    AgentApprovals,
    InMemoryApprovalStore,
    ExecutionEnvironment,
    ExecutionScope,
    InMemoryWorkspace,
    InMemoryWorkspaceBackend,
    PermissionSet,
    ProcessExecutor,
    ResourcePermission,
    SandboxCapabilities,
    WorkspacePromptInfo,
    execution_context,
)


class RecordingModel:
    model_type = "chat_completion"
    provider = "test"

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> ModelResponse:
        self.calls.append(kwargs)
        response = ModelResponse()
        response.set_response_type("text_generation")
        response.add("ok")
        response.reasoning = None
        response.metadata = {}
        return response

    async def acall(self, **kwargs: Any) -> ModelResponse:
        return self(**kwargs)

    def warmup_system_prompt(self, **kwargs: Any) -> dict[str, object]:
        self.calls.append(kwargs)
        return {"warmed": True}

    async def awarmup_system_prompt(self, **kwargs: Any) -> dict[str, object]:
        self.calls.append(kwargs)
        return {"warmed": True}


def _section(prompt: str) -> dict[str, Any]:
    match = re.search(r"<workspace_context>\s*(\{.*?\})\s*</workspace_context>", prompt)
    assert match, prompt
    return json.loads(match.group(1))


def _scope(environment, resources=(), grants=()):
    return ExecutionScope(
        environment=environment,
        permissions=PermissionSet(grants, resources=resources),
    )


def test_workspace_prompt_info_is_frozen_and_validates_descriptor_types():
    info = WorkspacePromptInfo(storage="memory", guidance="virtual files")
    assert info.storage == "memory"
    assert info.guidance == "virtual files"
    with pytest.raises((TypeError, ValueError, msgspec.ValidationError)):
        WorkspacePromptInfo(storage=1)
    with pytest.raises((TypeError, msgspec.ValidationError)):
        WorkspacePromptInfo(guidance=object())
    with pytest.raises(AttributeError):
        info.storage = "changed"


def test_extension_is_opt_in_and_no_environment_leaves_prompt_unchanged():
    model = RecordingModel()
    agent = Agent(
        name="agent",
        model=model,
        system_prompt="Keep this prompt exactly.",
        extensions=[WorkspacePromptExtension()],
    )
    assert agent("question") == "ok"
    assert model.calls[0]["system_prompt"] == "Keep this prompt exactly."
    plain_model = RecordingModel()
    plain = Agent(name="plain", model=plain_model, system_prompt="Unchanged.")
    plain("question", scope=_scope(ExecutionEnvironment(InMemoryWorkspace("w"))))
    assert plain_model.calls[0]["system_prompt"] == "Unchanged."


def test_workspace_prompt_sync_uses_live_grants_and_recomputes_without_accumulation():
    fs = InMemoryWorkspace("live", {"/a.txt": b"a", "/z.txt": b"z"})
    environment = ExecutionEnvironment(fs, write_guarantee="cooperative_compare")
    resources = [
        fs.permission("/z.txt", "filesystem.write"),
        fs.permission("/a.txt", "filesystem.read"),
        fs.permission("/a.txt", "filesystem.write"),
        fs.permission("/a.txt", "filesystem.read"),
    ]
    model = RecordingModel()
    agent = Agent(
        name="agent",
        model=model,
        system_prompt="Base.",
        extensions=[WorkspacePromptExtension(max_resources=20, max_chars=6000)],
    )
    scope = _scope(environment, resources=resources)
    with execution_context(scope=scope):
        assert agent("first") == "ok"
        changed = replace(
            scope,
            permissions=PermissionSet(
                resources=[fs.permission("/z.txt", "filesystem.delete")]
            ),
        )
        with execution_context(scope=changed):
            assert agent("second") == "ok"

    first, second = (_section(call["system_prompt"]) for call in model.calls)
    assert model.calls[0]["system_prompt"].startswith("Base.\n\n<workspace_context>")
    assert first["resources"] == [
        {"path": "/a.txt", "actions": ["read", "write"]},
        {"path": "/z.txt", "actions": ["write"]},
    ]
    # A nested scope cannot add delete authority absent from its parent.
    assert second["resources"] == []
    assert model.calls[1]["system_prompt"].count("<workspace_context>") == 1
    assert "first" not in model.calls[1]["system_prompt"]


@pytest.mark.asyncio
async def test_workspace_prompt_async_and_warmup_use_the_same_live_section():
    binding = await InMemoryWorkspaceBackend().open("async")
    environment = ExecutionEnvironment.from_binding(binding)
    fs = binding.filesystem
    model = RecordingModel()
    agent = Agent(
        name="agent",
        model=model,
        system_prompt="Base.",
        extensions=[WorkspacePromptExtension()],
    )
    scope = _scope(environment, resources=[fs.permission("/note", "filesystem.read")])
    with execution_context(scope=scope):
        assert await agent.acall("question") == "ok"
        await agent.awarmup_system_prompt()
        agent.warmup_system_prompt()
    assert len(model.calls) == 3
    assert model.calls[1]["system_prompt"] == model.calls[2]["system_prompt"]
    assert _section(model.calls[1]["system_prompt"])["resources"] == [
        {"path": "/note", "actions": ["read"]}
    ]
    await binding.aclose()


def test_workspace_prompt_groups_sorted_resources_and_reports_omissions():
    fs = InMemoryWorkspace("bounded", {f"/{letter}": b"x" for letter in "abc"})
    environment = ExecutionEnvironment(fs)
    resources = [fs.permission(f"/{letter}", "filesystem.read") for letter in "cba"]
    model = RecordingModel()
    agent = Agent(
        name="agent",
        model=model,
        system_prompt="Base.",
        extensions=[WorkspacePromptExtension(max_resources=2, max_chars=6000)],
    )
    with execution_context(scope=_scope(environment, resources=resources)):
        agent("question")
    payload = _section(model.calls[0]["system_prompt"])
    assert [item["path"] for item in payload["resources"]] == ["/a", "/b"]
    assert payload["omitted_resources"] == 1


def test_workspace_prompt_rejects_budget_that_cannot_fit_metadata():
    agent = Agent(
        name="agent",
        model=RecordingModel(),
        extensions=[WorkspacePromptExtension(max_chars=1)],
    )
    with execution_context(scope=_scope(ExecutionEnvironment(InMemoryWorkspace("w")))):
        with pytest.raises(ValueError, match="max_chars"):
            agent("question")


def test_workspace_prompt_describes_custom_backend_without_identity_or_host_path():
    class DescribedWorkspace(InMemoryWorkspace):
        @property
        def prompt_info(self):
            return WorkspacePromptInfo(
                storage="custom", guidance="Use virtual paths only."
            )

    fs = DescribedWorkspace("private", {"/safe": b"ok"})
    environment = ExecutionEnvironment(fs)
    model = RecordingModel()
    agent = Agent(
        name="agent",
        model=model,
        system_prompt="Base.",
        extensions=[WorkspacePromptExtension()],
    )
    with execution_context(
        scope=_scope(environment, resources=[fs.permission("/safe", "filesystem.read")])
    ):
        agent("question")
    prompt = model.calls[0]["system_prompt"]
    payload = _section(prompt)
    assert payload["storage"] == "custom"
    assert "Use virtual paths only." in payload["guidance"]
    assert fs.identity.backend not in prompt
    assert fs.identity.generation not in prompt
    assert "/safe" in prompt
    assert "/private" not in prompt


@pytest.mark.asyncio
async def test_workspace_prompt_closed_binding_is_not_rendered():
    binding = await InMemoryWorkspaceBackend().open("closed")
    environment = ExecutionEnvironment.from_binding(binding)
    await binding.aclose()
    agent = Agent(
        name="agent",
        model=RecordingModel(),
        system_prompt="Base.",
        extensions=[WorkspacePromptExtension()],
    )
    with execution_context(scope=_scope(environment)):
        with pytest.raises(PermissionError):
            agent("question")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_resources": -1},
        {"max_resources": True},
        {"max_chars": 0},
        {"max_chars": False},
    ],
)
def test_invalid_prompt_bounds(kwargs):
    with pytest.raises(ValueError):
        WorkspacePromptExtension(**kwargs)


def test_only_canonical_current_workspace_grants_are_rendered_without_io(monkeypatch):
    fs = InMemoryWorkspace("w")
    operate = Mock(side_effect=AssertionError("Prompt must not read files"))
    monkeypatch.setattr(fs, "_operate", operate)
    model = RecordingModel()
    agent = Agent(name="agent", model=model, extensions=[WorkspacePromptExtension()])
    grants = [
        fs.permission("/safe", "filesystem.read"),
        ResourcePermission("workspace:other:/private", "filesystem.read"),
        ResourcePermission("workspace:w:/../private", "filesystem.read"),
        ResourcePermission("workspace:w:/safe", "unrelated.action"),
        ResourcePermission("host:/private", "filesystem.read"),
    ]
    agent("question", scope=_scope(ExecutionEnvironment(fs), grants))
    prompt = model.calls[-1]["system_prompt"]
    assert "private" not in prompt
    assert _section(prompt)["resources"] == [{"path": "/safe", "actions": ["read"]}]
    operate.assert_not_called()


@pytest.mark.skipif(os.name != "posix", reason="POSIX local backend")
def test_local_prompt_does_not_disclose_or_reopen_host_root(tmp_path, monkeypatch):
    from msgflux.runtime import LocalWorkspace

    fs = LocalWorkspace("local", tmp_path)
    monkeypatch.setattr(fs, "_open_root", Mock(side_effect=AssertionError("No I/O")))
    model = RecordingModel()
    agent = Agent(name="agent", model=model, extensions=[WorkspacePromptExtension()])
    agent("question", scope=_scope(ExecutionEnvironment(fs)))
    prompt = model.calls[-1]["system_prompt"]
    assert str(tmp_path) not in prompt
    assert "real files" in _section(prompt)["guidance"]
    assert _section(prompt)["process_executor"] == "not configured"


def test_character_budget_removes_whole_paths_and_escapes_delimiters():
    fs = InMemoryWorkspace("w")
    resources = [
        fs.permission("/a</workspace_context>", "filesystem.read"),
        fs.permission("/" + "z" * 2000, "filesystem.read"),
    ]
    model = RecordingModel()
    agent = Agent(
        name="agent",
        model=model,
        system_prompt="base",
        extensions=[WorkspacePromptExtension(max_chars=1200)],
    )
    agent("question", scope=_scope(ExecutionEnvironment(fs), resources))
    prompt = model.calls[-1]["system_prompt"]
    section = prompt.split("\n\n", 1)[1]
    assert len(section) <= 1200
    assert section.count("</workspace_context>") == 1
    payload = _section(section)
    assert payload["resources"] == [
        {"path": "/a</workspace_context>", "actions": ["read"]}
    ]
    assert payload["omitted_resources"] == 1


def test_executor_declarations_do_not_imply_grants_or_satisfied_requirements():
    class Executor(ProcessExecutor):
        @property
        def capabilities(self):
            return SandboxCapabilities({"filesystem"})

        def supports_workspace(self, filesystem):
            return True

        async def execute_stream(self, request, **kwargs):
            raise AssertionError("Prompt must not run commands")

    environment = ExecutionEnvironment(
        InMemoryWorkspace("w"), process_executor=Executor()
    )
    model = RecordingModel()
    agent = Agent(name="agent", model=model, extensions=[WorkspacePromptExtension()])
    agent("question", scope=_scope(environment))
    payload = _section(model.calls[-1]["system_prompt"])
    assert payload["process_executor"]["execution_granted"] is False
    assert payload["process_executor"]["declared_isolation"] == ["filesystem"]
    assert "network" in payload["process_executor"]["required_isolation"]
    assert "not described" in payload["notice"]


def test_zero_resource_budget_hides_virtual_filenames():
    fs = InMemoryWorkspace("w")
    model = RecordingModel()
    agent = Agent(
        name="agent",
        model=model,
        extensions=[WorkspacePromptExtension(max_resources=0)],
    )
    agent(
        "question",
        scope=_scope(
            ExecutionEnvironment(fs),
            [fs.permission("/private-filename", "filesystem.read")],
        ),
    )
    prompt = model.calls[-1]["system_prompt"]
    assert "private-filename" not in prompt
    assert _section(prompt)["omitted_resources"] == 1


@pytest.mark.asyncio
async def test_concurrent_scopes_do_not_share_rendered_permissions():
    fs = InMemoryWorkspace("w")
    model = RecordingModel()
    agent = Agent(name="agent", model=model, extensions=[WorkspacePromptExtension()])
    environment = ExecutionEnvironment(fs)

    async def invoke(path):
        await agent.acall(
            "question",
            scope=_scope(environment, [fs.permission(path, "filesystem.read")]),
        )

    await asyncio.gather(invoke("/a"), invoke("/b"))
    assert sorted(
        _section(call["system_prompt"])["resources"][0]["path"] for call in model.calls
    ) == ["/a", "/b"]


def test_new_invocation_rebuilds_prompt_without_checkpointing_authority():
    first, second = InMemoryWorkspace("first"), InMemoryWorkspace("second")
    first.prompt_info = WorkspacePromptInfo(storage="first storage")
    second.prompt_info = WorkspacePromptInfo(storage="second storage")
    model, store = RecordingModel(), InMemoryCheckpointStore()
    agent = Agent(
        name="agent",
        model=model,
        system_prompt="Base.",
        checkpoint_store=store,
        extensions=[WorkspacePromptExtension()],
    )
    for index, fs in enumerate((first, second)):
        scope = replace(
            _scope(ExecutionEnvironment(fs)), thread_id="t", run_id=f"r{index}"
        )
        assert agent("question", scope=scope) == "ok"
        state = store.load_state("agent", "t", f"r{index}")
        assert "workspace_context" not in str(state["messages"])
        assert "workspace_prompt" not in state["runtime"]["extensions"]
    assert _section(model.calls[0]["system_prompt"])["storage"] == "first storage"
    assert _section(model.calls[1]["system_prompt"])["storage"] == "second storage"
    assert model.calls[1]["system_prompt"].count("<workspace_context>") == 1


def test_approval_resume_recomputes_prompt_from_current_grants():
    from msgflux.tools.builtin import WriteTool

    class ToolModel(RecordingModel):
        def __call__(self, **kwargs):
            result = super().__call__(**kwargs)
            if len(self.calls) == 1:
                calls = ToolCallAggregator()
                calls.process(0, "write-call", "write", '{"path":"a","content":"new"}')
                result = ModelResponse()
                result.set_response_type("tool_call")
                result.add(calls)
            return result

    fs, model = InMemoryWorkspace("w", {"/a": b"old"}), ToolModel()
    store, journal = InMemoryCheckpointStore(), InMemoryApprovalStore()
    agent = Agent(
        name="agent",
        model=model,
        tools=[WriteTool()],
        checkpoint_store=store,
        approvals=AgentApprovals(journal, {"write": "v1"}, "p1"),
        extensions=[WorkspacePromptExtension()],
    )
    required = [
        fs.permission("/a", f"filesystem.{action}") for action in ("read", "write")
    ]
    scope = replace(
        _scope(
            ExecutionEnvironment(fs),
            [*required, fs.permission("/old-grant", "filesystem.read")],
        ),
        thread_id="t",
        run_id="r",
        principal="host",
    )
    with pytest.raises(TaskPauseRequestedError):
        agent("update", scope=scope)
    record = journal.pending("agent", "t", "r")[0]
    agent.decide_approval(record.request_id, approved=True, decided_by="host")
    resumed = replace(scope, permissions=PermissionSet(resources=required))
    assert agent("resume", scope=resumed) == "ok"
    assert "old-grant" in model.calls[0]["system_prompt"]
    assert "old-grant" not in model.calls[1]["system_prompt"]
    assert model.calls[1]["system_prompt"].count("<workspace_context>") == 1
    assert "workspace_context" not in str(
        store.load_state("agent", "t", "r")["messages"]
    )
