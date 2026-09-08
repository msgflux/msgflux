from types import SimpleNamespace
from unittest.mock import Mock

import msgspec
import pytest

from msgflux.nn import ToolLibrary
from msgflux.runtime import ExecutionScope, PermissionSet, execution_context
from msgflux.runtime.background import BackgroundTaskDispatcher
from msgflux.runtime.events import EventType, _capture_events, _EventSink
from msgflux.tools.config import tool_config
from msgflux.nn.modules.tool.runtime import ToolExecutionPlan, ToolRuntimeContext
from msgflux.tools.runtime import ToolIntent
from msgflux.tools.types import ToolBucket, ToolLibraryOperator


def make_library(*, background=False):
    calls = []

    @tool_config(
        required_permissions=["filesystem.read"], background=background, retry=False
    )
    def read(value: str) -> str:
        """Read a value."""
        calls.append(value)
        return value

    return ToolLibrary("files", [read]), calls


def authority():
    return execution_context(
        scope=ExecutionScope(
            principal="user:42", permissions=PermissionSet(["filesystem.read"])
        )
    )


def test_permission_denial_is_structured_and_does_not_leak_arguments():
    library, calls = make_library()
    events = []
    with _capture_events(_EventSink(events.append)):
        response = library([("call_1", "read", {"value": "secret"})])
    assert calls == []
    assert "Missing tool permissions" in response.tool_calls[0].error
    denied = [
        event
        for event in events
        if event.type in {EventType.TOOL_PERMISSION_DENIED, EventType.TOOL_BLOCKED}
    ]
    assert len(denied) == 2
    assert "secret" not in repr([event.data for event in denied])


def test_live_grants_allow_execution_but_vars_do_not_grant_authority():
    library, calls = make_library()
    with pytest.raises(RuntimeError, match="Missing tool permissions"):
        library.run(
            "read",
            {"value": "denied"},
            vars={"permissions": PermissionSet(["filesystem.read"])},
        )
    with authority():
        assert library.run("read", {"value": "allowed"}) == "allowed"
    assert calls == ["allowed"]


@pytest.mark.asyncio
async def test_async_grants_and_nested_sync_tools_keep_authority():
    library, calls = make_library()

    def nested(value: str) -> str:
        """Delegate to a protected tool from an executor thread."""
        return library.run("read", {"value": value})

    outer = ToolLibrary("outer", [nested])
    with pytest.raises(RuntimeError, match="Missing tool permissions"):
        await library.arun("read", {"value": "denied"})
    with authority():
        assert await outer.arun("nested", {"value": "nested"}) == "nested"
        assert await library.arun("read", {"value": "allowed"}) == "allowed"
    assert calls == ["nested", "allowed"]


def test_requirements_are_frozen_and_not_part_of_model_schema():
    requirements = ["filesystem.read"]
    decorator = tool_config(required_permissions=requirements)
    requirements.clear()

    @decorator
    def read(value: str) -> str:
        """Read a value."""
        return value

    library = ToolLibrary("files", [read])
    definition = library.get_tool_definition("read")
    assert definition.required_permissions == ("filesystem.read",)
    assert "required_permissions" not in repr(definition.input_schema)
    with pytest.raises(AttributeError):
        definition.required_permissions = ()


def test_background_denial_does_not_create_or_resume_task():
    library, calls = make_library(background=True)
    dispatcher = library.get_background_dispatcher()
    original = dispatcher.dispatch
    dispatcher.dispatch = Mock(side_effect=original)
    with pytest.raises(RuntimeError, match="Missing tool permissions"):
        library.run("read", {"value": "denied"})
    dispatcher.dispatch.assert_not_called()
    handle = Mock()
    handle.get_tool_definition.return_value = library.get_tool_definition("read")
    background = BackgroundTaskDispatcher(handle)
    with pytest.raises(PermissionError, match="Missing tool permissions"):
        background.resume_agent_task(
            task=SimpleNamespace(tool_name="read"), message="resume"
        )
    handle.get_task_store.return_value.requeue.assert_not_called()
    assert calls == []


@pytest.mark.asyncio
async def test_direct_adapters_enforce_requirements():
    library, calls = make_library()
    tool = library.get_tool_definition("read").executor
    with pytest.raises(PermissionError):
        tool(value="denied")
    with pytest.raises(PermissionError):
        await tool.acall(value="denied")
    with authority():
        assert tool(value="sync") == "sync"
        assert await tool.acall(value="async") == "async"
    assert calls == ["sync", "async"]


def test_denial_precedes_policies_and_keeps_batch_siblings_independent(monkeypatch):
    library, calls = make_library()
    policy = Mock(side_effect=AssertionError("Denied tools must not reach policies"))
    monkeypatch.setattr(library, "_abefore_tool_policy", policy)
    with pytest.raises(RuntimeError, match="Missing tool permissions"):
        library.run("read", {"value": "denied"})
    policy.assert_not_called()
    monkeypatch.undo()

    def public(value: str) -> str:
        """Return a public value."""
        return value

    library.add(public)
    response = library(
        [
            ("a", "read", {"value": "denied"}),
            ("b", "public", {"value": "allowed"}),
        ]
    )
    assert response.tool_calls[0].error
    assert response.tool_calls[1].result == "allowed"
    assert calls == []


@pytest.mark.asyncio
async def test_transformed_plan_cannot_remove_registered_requirements():
    library, calls = make_library()
    plan = ToolExecutionPlan(
        intent=ToolIntent(id="a", name="read", arguments={"value": "denied"}),
        definition=msgspec.structs.replace(
            library.get_tool_definition("read"), required_permissions=()
        ),
        visible_arguments={"value": "denied"},
    )
    outcome = await library._adispatch_runtime_plan(plan, ToolRuntimeContext())
    assert outcome.status == "blocked"
    assert outcome.error.code == "tool_permission_denied"
    assert calls == []


@pytest.mark.asyncio
async def test_dispatch_callback_rechecks_authority_and_denial_is_terminal(monkeypatch):
    library, calls = make_library()
    plan = ToolExecutionPlan(
        intent=ToolIntent(id="a", name="read", arguments={"value": "denied"}),
        definition=library.get_tool_definition("read"),
        visible_arguments={"value": "denied"},
    )

    async def restricted_dispatch(request):
        with execution_context(scope=ExecutionScope(permissions=PermissionSet())):
            denied = await request.execute()
            assert await request.execute() is denied
        # A dispatcher must not accidentally turn a permission denial into success.
        return msgspec.structs.replace(denied, status="completed", error=None)

    monkeypatch.setattr(library.runtime_extensions, "dispatch", restricted_dispatch)
    with authority():
        outcome = await library._adispatch_runtime_plan(plan, ToolRuntimeContext())
    assert outcome.status == "blocked"
    assert outcome.error.code == "tool_permission_denied"
    assert calls == []


def test_bucket_captured_tool_still_requires_live_authority():
    class Files(ToolBucket, ToolLibraryOperator):
        """Group file tools."""

        name = "files"
        capture = {"tool_kind": "files", "defer_loading": False}

        def __call__(self) -> str:
            return "files"

    @tool_config(tool_kind="files", required_permissions=["filesystem.read"])
    def read(value: str) -> str:
        """Read a value."""
        return value

    library = ToolLibrary("captured", [Files(), read])
    handle = library.get_handle().for_tool(tool_name="files")
    ref = library.get_tool_ref("read")
    with pytest.raises(RuntimeError, match="Missing tool permissions"):
        handle(ref, value="denied")
    with authority():
        assert handle(ref, value="allowed") == "allowed"


def test_background_worker_inherits_live_grants():
    library, calls = make_library(background=True)
    with authority():
        library.run("read", {"value": "allowed"})
        tasks = library.get_task_store().list()
        task = tasks[0]
        future = library.get_background_dispatcher().get_task_future(task.task_id)
        if future is not None:
            future.result(timeout=5)
        assert library.get_task_store().get(task.task_id).status == "completed"
    assert calls == ["allowed"]
