from __future__ import annotations

from pathlib import Path

import pytest
import asyncio
from unittest.mock import AsyncMock

from msgflux.coding.factory import (
    CodingAgentCatalog,
    CodingAgentOptions,
    _close_resources,
    create_coding_session_factory,
)
from msgflux.coding.config import (
    _openai_tool_family,
    resolve_coding_agent_options,
    resolve_coding_tools,
    validate_coding_agent_options,
)
from msgflux.runtime.service import ServiceThread
from msgflux.models.response import ModelResponse, ModelStreamResponse
from msgflux.models.tool_call_agg import ToolCallAggregator
from msgflux.runtime.service import AgentService, AgentSession, SQLiteServiceStore
from msgflux.runtime.service.serialization import event_record, snapshot_record


class _FakeModel:
    model_type = "chat_completion"

    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_workspace_group_expands_by_provider_and_explicit_tools_deduplicate():
    options = CodingAgentOptions(
        agent_id="coding",
        model="openai-codex/gpt-6-luna",
        reasoning_effort="medium",
        tools_active=("workspace", "read"),
        subagents=("lite-coding",),
    )

    active, deferred = resolve_coding_tools(options)

    assert active == ("bash", "read", "apply_patch")
    assert deferred == ()
    assert _openai_tool_family("openrouter/openai/gpt-4.1")
    assert _openai_tool_family("vendor/openai/model")
    assert not _openai_tool_family("anthropic/claude")


def test_selector_validation_catches_expanded_overlap_and_missing_children():
    overlap = CodingAgentOptions(
        agent_id="coding",
        model="openai/gpt-4.1",
        reasoning_effort="medium",
        tools_active=("workspace",),
        tools_deferred=("bash",),
    )
    with pytest.raises(ValueError, match="both active and deferred"):
        validate_coding_agent_options(overlap)

    unknown = CodingAgentOptions(
        agent_id="coding",
        model="openai/gpt-4.1",
        reasoning_effort="medium",
        tools_active=("agent",),
    )
    with pytest.raises(ValueError, match="Unknown active tool selector"):
        resolve_coding_tools(unknown)


def test_factory_builds_named_child_agents_lazily_and_closes_models(
    tmp_path: Path, monkeypatch
):
    import msgflux.coding.factory as factory_module

    created_models: list[_FakeModel] = []

    def create_model(_options):
        model = _FakeModel()
        created_models.append(model)
        return model

    monkeypatch.setattr(factory_module, "_create_model", create_model)
    parent = resolve_coding_agent_options(
        CodingAgentOptions(
            agent_id="coding",
            model="openai-codex/gpt-6-luna",
            reasoning_effort="medium",
            tools_active=("workspace",),
            subagents=("lite-coding",),
        )
    )
    child = resolve_coding_agent_options(
        CodingAgentOptions(
            agent_id="lite-coding",
            model="openai-codex/gpt-6-luna",
            reasoning_effort="medium",
            tools_active=("read",),
        )
    )
    catalog = CodingAgentCatalog(agents={"coding": parent, "lite-coding": child})
    session_factory = create_coding_session_factory(
        options=parent,
        catalog=catalog,
        config_path=tmp_path / "config.toml",
        runtime_dir=tmp_path / "runtime",
    )

    assert created_models == []
    session = asyncio.run(
        session_factory(ServiceThread("thread-1", "coding", str(tmp_path)))
    )

    assert len(created_models) == 2
    assert session.agent.get_module_name() == "coding"
    assert session.agent.agent_dir == tmp_path
    assert session.agent.has_extension("coding_checkpoints")
    agent_tool = session.agent.tool_library.library["agent"].impl
    assert [
        entry.name for entry in session.agent.tool_library.get_bucket_entries("agent")
    ] == ["lite-coding"]
    assert agent_tool.name == "agent"
    assert (
        session.agent.tool_library.get_tool_definition("bash").loading.deferred is False
    )

    asyncio.run(session.on_close())
    assert all(model.closed for model in created_models)


def test_custom_prompt_keeps_harness_context(tmp_path: Path, monkeypatch):
    import msgflux.coding.factory as factory_module

    monkeypatch.setattr(factory_module, "_create_model", lambda _options: _FakeModel())
    prompt = tmp_path / "custom.md"
    prompt.write_text("Do the requested task.", encoding="utf-8")
    options = CodingAgentOptions(
        agent_id="coding",
        model="openai/gpt-4.1",
        reasoning_effort="medium",
        tools_active=("read",),
        prompt_file=prompt,
    )
    session = asyncio.run(
        create_coding_session_factory(
            options=options,
            catalog=CodingAgentCatalog(agents={"coding": options}),
            config_path=tmp_path / "config.toml",
            runtime_dir=tmp_path / "runtime",
        )(ServiceThread("thread-1", "coding", str(tmp_path)))
    )

    system_prompt = session.agent.system_prompt.data
    assert "You are coding, the coding agent" in system_prompt
    assert f"The project directory is {tmp_path}" in system_prompt
    assert "Use only the editing tools actually supplied." in system_prompt
    assert "Do the requested task." in system_prompt


def test_partial_tool_construction_closes_prior_owned_resources(tmp_path, monkeypatch):
    import msgflux.coding.factory as factory_module

    created_models = []

    def create_model(_options):
        model = _FakeModel()
        created_models.append(model)
        return model

    monkeypatch.setattr(factory_module, "_create_model", create_model)

    class FakeWorkspace:
        closed = False

        from msgflux.coding.config import coding_permission_set

        permissions = coding_permission_set(("read", "write", "execute"))

        def with_permissions(self, _permissions):
            return self

        async def aclose(self):
            self.closed = True

    workspace = FakeWorkspace()
    monkeypatch.setattr(
        factory_module.AgentWorkspace, "local", lambda _root, **_kwargs: workspace
    )

    class OwnedTool:
        tool_config = {}

        def __init__(self):
            self.model = _FakeModel()

        def __call__(self):
            pass

        def close(self):
            self.model.close()
            raise RuntimeError("tool cleanup failed")

    owned_tool = OwnedTool()
    factories = dict(factory_module._TOOL_FACTORIES)
    factories["web_search"] = lambda: owned_tool

    def fail_construction():
        raise RuntimeError("tool constructor failed")

    factories["web_fetch"] = fail_construction
    monkeypatch.setattr(factory_module, "_TOOL_FACTORIES", factories)
    options = CodingAgentOptions(
        agent_id="coding",
        model="openai/gpt-4.1",
        reasoning_effort="medium",
        tools_active=("web_search", "web_fetch"),
    )
    session_factory = create_coding_session_factory(
        options=options,
        catalog=CodingAgentCatalog(agents={"coding": options}),
        config_path=tmp_path / "config.toml",
        runtime_dir=tmp_path / "runtime",
    )

    with pytest.raises(RuntimeError, match="tool constructor failed"):
        asyncio.run(session_factory(ServiceThread("thread-1", "coding", str(tmp_path))))

    assert owned_tool.model.closed
    assert created_models[0].closed
    assert workspace.closed


@pytest.mark.asyncio
async def test_child_agent_tool_result_is_checkpoint_and_watch_serializable(
    tmp_path, monkeypatch
):
    import msgflux.coding.factory as factory_module

    monkeypatch.setattr(factory_module, "_create_model", lambda _options: _FakeModel())
    parent = resolve_coding_agent_options(
        CodingAgentOptions(
            agent_id="coding",
            model="openai/gpt-4.1",
            reasoning_effort="medium",
            tools_active=(),
            subagents=("lite-coding",),
        )
    )
    child = resolve_coding_agent_options(
        CodingAgentOptions(
            agent_id="lite-coding",
            model="openai/gpt-4.1",
            reasoning_effort="medium",
        )
    )
    catalog = CodingAgentCatalog(agents={"coding": parent, "lite-coding": child})
    session_factory = create_coding_session_factory(
        options=parent,
        catalog=catalog,
        config_path=tmp_path / "config.toml",
        runtime_dir=tmp_path / "runtime",
    )
    session = await session_factory(
        ServiceThread("stream-child-thread", "coding", str(tmp_path))
    )
    parent_agent = session.agent
    child_agent = parent_agent.tool_library.get_tool_definition(
        "lite-coding"
    ).executor.impl

    child_agent.generator.aforward = AsyncMock(
        side_effect=lambda **_kwargs: (
            _stream_response("child answer")
            if child_agent.config["stream"]
            else _text_response("child answer")
        )
    )
    calls = ToolCallAggregator()
    calls.process(
        0,
        "call_child",
        "agent",
        '{"name":"lite-coding","message":"solve this"}',
    )
    tool_call = ModelResponse()
    tool_call.set_response_type("tool_call")
    tool_call.add(calls)
    parent_agent.generator.aforward = AsyncMock(
        side_effect=[tool_call, _text_response("parent complete")]
    )

    service = AgentService(store=SQLiteServiceStore(tmp_path / "service.sqlite3"))
    service.register("coding", lambda _thread: session)
    thread = await service.open_thread(
        "coding", thread_id="stream-child-thread", cwd=tmp_path
    )
    try:
        async with service.watch(thread.thread_id) as watcher:
            receipt = await service.prompt(
                thread.thread_id, "delegate", request_id="child-dispatch"
            )
            settled = await service.wait(thread.thread_id, "child-dispatch")
            assert settled.status == "completed", settled.error

            records = []
            async for event in watcher:
                records.append(event_record(event))
                if event.type == "run.end" and event.run_id == receipt.run_id:
                    break

        snapshot = snapshot_record(await service.snapshot(thread.thread_id))
        assert any(record.type == "run.end" for record in records)
        assert snapshot.messages is not None
        assert child_agent.config["stream"] is False
    finally:
        await service.aclose()


def test_factory_enables_read_vision_from_model_input_modalities(tmp_path, monkeypatch):
    import msgflux.coding.factory as factory_module
    from types import SimpleNamespace

    model = _FakeModel()
    model.profile = SimpleNamespace(modalities=SimpleNamespace(input=["text", "image"]))
    monkeypatch.setattr(factory_module, "_create_model", lambda _options: model)
    options = CodingAgentOptions(
        agent_id="coding",
        model="openai/gpt-4.1",
        reasoning_effort="medium",
        tools_active=("read",),
    )
    session = asyncio.run(
        create_coding_session_factory(
            options=options,
            catalog=CodingAgentCatalog(agents={"coding": options}),
            config_path=tmp_path / "config.toml",
            runtime_dir=tmp_path / "runtime",
        )(ServiceThread("vision-thread", "coding", str(tmp_path)))
    )

    assert session.agent.tool_library.library["read"].impl.supports_vision is True


def test_factory_adds_progress_tool_only_to_interactive_parent(tmp_path, monkeypatch):
    import msgflux.coding.factory as factory_module
    from types import SimpleNamespace
    from msgflux.tools.builtin import SendUserMessageTool

    def create_model(_options):
        model = _FakeModel()
        model.api_mode_capabilities = SimpleNamespace(assistant_commentary=False)
        model.supports_unphased_tool_commentary = lambda: False
        return model

    monkeypatch.setattr(factory_module, "_create_model", create_model)
    parent = resolve_coding_agent_options(
        CodingAgentOptions(
            agent_id="coding",
            model="openai/gpt-4.1",
            reasoning_effort="medium",
            tools_active=(),
            subagents=("lite-coding",),
        )
    )
    child = resolve_coding_agent_options(
        CodingAgentOptions(
            agent_id="lite-coding",
            model="openai/gpt-4.1",
            reasoning_effort="medium",
            tools_active=("read",),
        )
    )
    session = asyncio.run(
        create_coding_session_factory(
            options=parent,
            catalog=CodingAgentCatalog(agents={"coding": parent, "lite-coding": child}),
            config_path=tmp_path / "config.toml",
            runtime_dir=tmp_path / "runtime",
        )(ServiceThread("interactive-thread", "coding", str(tmp_path)))
    )
    parent_tools = session.agent.tool_library.library
    child_agent = session.agent.tool_library.get_tool_definition(
        "lite-coding"
    ).executor.impl
    child_tools = child_agent.tool_library.library

    assert isinstance(parent_tools["send_user_message"].impl, SendUserMessageTool)
    assert "send_user_message" not in child_tools
    assert "This is a noninteractive print run." in child_agent.system_prompt.data


def _text_response(text: str) -> ModelResponse:
    response = ModelResponse()
    response.set_response_type("text_generation")
    response.add(text)
    return response


def _stream_response(text: str) -> ModelStreamResponse:
    response = ModelStreamResponse(mode="async")
    response.set_response_type("text_generation")
    response.add(text)
    response.finish()
    return response


@pytest.mark.parametrize("model", ["openai/gpt-4.1", "anthropic/claude"])
def test_read_only_workspace_has_navigation_and_no_process_or_edit_tools(model):
    options = CodingAgentOptions(
        agent_id="explorer",
        model=model,
        reasoning_effort="medium",
        tools_active=("workspace",),
        permissions=("read",),
    )
    assert resolve_coding_tools(options) == (("read", "ls", "glob", "grep"), ())


@pytest.mark.parametrize("tool", ["bash", "write", "edit", "delete", "apply_patch"])
@pytest.mark.parametrize("deferred", [False, True])
def test_read_only_rejects_explicit_mutating_tools(tool, deferred):
    options = CodingAgentOptions(
        agent_id="explorer",
        model="openai/gpt-4.1",
        reasoning_effort="medium",
        tools_active=() if deferred else (tool,),
        tools_deferred=(tool,) if deferred else (),
        permissions=("read",),
    )
    with pytest.raises(ValueError, match="requires permissions"):
        resolve_coding_tools(options)


@pytest.mark.asyncio
async def test_read_only_delegation_blocks_mutations_and_descendant_escalation(
    tmp_path, monkeypatch
):
    import msgflux.coding.factory as factory_module
    from msgflux.runtime.context import get_execution_scope

    monkeypatch.setattr(factory_module, "_create_model", lambda _options: _FakeModel())
    (tmp_path / "marker.txt").write_text("readable marker", encoding="utf-8")
    parent = resolve_coding_agent_options(
        CodingAgentOptions(
            agent_id="coding",
            model="openai/gpt-4.1",
            reasoning_effort="medium",
            tools_active=("workspace",),
            subagents=("explorer",),
        )
    )
    explorer = resolve_coding_agent_options(
        CodingAgentOptions(
            agent_id="explorer",
            model="openai/gpt-4.1",
            reasoning_effort="medium",
            tools_active=("workspace",),
            permissions=("read",),
            subagents=("helper",),
        )
    )
    helper = resolve_coding_agent_options(
        CodingAgentOptions(
            agent_id="helper",
            model="openai/gpt-4.1",
            reasoning_effort="medium",
            tools_active=("workspace",),
        )
    )
    catalog = CodingAgentCatalog(
        agents={"coding": parent, "explorer": explorer, "helper": helper}
    )
    session = await create_coding_session_factory(
        options=parent,
        catalog=catalog,
        config_path=tmp_path / "config.toml",
        runtime_dir=tmp_path / "runtime",
    )(ServiceThread("readonly-delegation", "coding", str(tmp_path)))
    main = session.agent
    child = main.tool_library.get_tool_definition("explorer").executor.impl
    grandchild = child.tool_library.get_tool_definition("helper").executor.impl
    for agent in (child, grandchild):
        assert (
            not {"bash", "write", "edit", "delete", "apply_patch"}
            & agent.tool_library.library.keys()
        )
        assert agent.workspace.shares_environment(main.workspace)
        assert "Workspace permissions: read." in agent.system_prompt.data
        assert not agent.workspace.permissions.allows(("filesystem.write",))

    async def verify_child_authority(**_kwargs):
        scope = get_execution_scope()
        assert scope.workspace.read_text("marker.txt") == "readable marker"
        with pytest.raises(PermissionError):
            scope.workspace.write_text("forbidden.txt", "must not exist")
        with pytest.raises(PermissionError):
            scope.workspace.run(["touch", "forbidden.txt"], timeout=5)
        # Even explicit broad scope grants cannot bypass the child's ceiling.
        broad = scope.with_overrides(permissions=main.workspace.permissions)
        delegated = grandchild._workspace_scope(broad)
        assert not delegated.permissions.allows(("filesystem.write",))
        assert not delegated.permissions.allows(("process.execute",))
        return _text_response("readable marker")

    child.generator.aforward = AsyncMock(side_effect=verify_child_authority)
    calls = ToolCallAggregator()
    calls.process(
        0, "call_explorer", "agent", '{"name":"explorer","message":"inspect marker"}'
    )
    call = ModelResponse()
    call.set_response_type("tool_call")
    call.add(calls)
    main.generator.aforward = AsyncMock(
        side_effect=[call, _stream_response("exploration complete")]
    )
    store = SQLiteServiceStore(tmp_path / "service.sqlite3")
    service = AgentService(store=store)
    service.register("coding", lambda _thread: session)
    try:
        thread = await service.open_thread(
            "coding", thread_id="readonly-delegation", cwd=str(tmp_path)
        )
        receipt = await service.prompt(
            thread.thread_id, "Explore the repository", request_id="explore"
        )
        settled = await service.wait(thread.thread_id, "explore")
        assert settled.status == "completed", settled.error
        assert receipt.run_id is not None
        child.generator.aforward.assert_awaited()
        assert not (tmp_path / "forbidden.txt").exists()
        # Restricting the child does not reduce the writable parent's authority.
        main.workspace.write_text("parent.txt", "allowed")
        assert (tmp_path / "parent.txt").read_text() == "allowed"
        assert "agent" not in grandchild.tool_library.library
    finally:
        await service.aclose()
        store.close()


@pytest.mark.parametrize(
    ("permissions", "expected"),
    [
        (("read", "write"), ("read", "apply_patch", "ls", "glob", "grep")),
        (("read", "execute"), ("bash", "read")),
        (("execute",), ("bash",)),
        (("write",), ()),
        ((), ()),
    ],
)
def test_workspace_selection_respects_independent_permissions(permissions, expected):
    options = CodingAgentOptions(
        agent_id="coding",
        model="openai/gpt-4.1",
        reasoning_effort="medium",
        tools_active=("workspace",),
        permissions=permissions,
    )
    assert resolve_coding_tools(options) == (expected, ())


@pytest.mark.asyncio
@pytest.mark.parametrize("permissions", [("read", "write"), ("execute",), ()])
async def test_factory_enforces_independent_workspace_permissions(
    tmp_path, monkeypatch, permissions
):
    import msgflux.coding.factory as factory_module

    monkeypatch.setattr(factory_module, "_create_model", lambda _options: _FakeModel())
    (tmp_path / "marker.txt").write_text("original", encoding="utf-8")
    options = resolve_coding_agent_options(
        CodingAgentOptions(
            agent_id="coding",
            model="openai/gpt-4.1",
            reasoning_effort="medium",
            tools_active=("workspace",),
            permissions=permissions,
        )
    )
    session = await create_coding_session_factory(
        options=options,
        catalog=CodingAgentCatalog(agents={"coding": options}),
        config_path=tmp_path / "config.toml",
        runtime_dir=tmp_path / "runtime",
    )(ServiceThread("permission-matrix", "coding", str(tmp_path)))
    workspace = session.agent.workspace
    try:
        if "read" in permissions:
            assert workspace.read_text("marker.txt") == "original"
        else:
            with pytest.raises(PermissionError):
                workspace.read_text("marker.txt")
        if "write" in permissions:
            workspace.write_text("created.txt", "allowed")
            assert (tmp_path / "created.txt").read_text() == "allowed"
        else:
            with pytest.raises(PermissionError):
                workspace.write_text("created.txt", "denied")
            assert not (tmp_path / "created.txt").exists()
        if "execute" in permissions:
            assert workspace.run(["true"], timeout=5).returncode == 0
        else:
            with pytest.raises(PermissionError):
                workspace.run(["true"], timeout=5)
        assert (
            f"Workspace permissions: {', '.join(permissions) or 'none'}."
            in session.agent.system_prompt.data
        )
    finally:
        await session._close()
