from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import Mock

import msgspec
import pytest

from msgflux.nn import ToolLibrary
from msgflux.runtime import (
    AgentApprovals,
    InMemoryApprovalStore,
    ExecutionEnvironment,
    InMemoryWorkspace,
    ExecutionScope,
    PermissionSet,
    ResourcePermission,
    SandboxCapabilities,
    SandboxRequirements,
    execution_context,
    get_execution_scope,
)
from msgflux.exceptions import TaskPauseRequestedError
from msgflux.tools.runtime import ToolIntent
from msgflux.runtime.background import BackgroundTaskDispatcher
from msgflux.tools.config import tool_config

READ = ResourcePermission("file:report", "filesystem.read")
WRITE = ResourcePermission("file:report", "filesystem.write")


def test_resource_authority_is_exact_restrictive_and_not_restored():
    parent = PermissionSet(resources=[READ])
    child = PermissionSet(resources=[READ, WRITE])
    with execution_context(scope=ExecutionScope(principal="user", permissions=parent)):
        with execution_context(scope=ExecutionScope(permissions=child)):
            assert get_execution_scope().permissions.resources == frozenset([READ])
            serialized = get_execution_scope().to_dict()
            assert "permissions" not in serialized
    with execution_context(scope=ExecutionScope(**serialized)):
        assert get_execution_scope().permissions.resources == frozenset()
    assert parent.missing_resources(
        [ResourcePermission("file:report/child", "filesystem.read")]
    )
    assert PermissionSet(resources=[asdict(READ)]) == parent


@pytest.mark.parametrize("background", [False, True])
@pytest.mark.asyncio
async def test_required_resources_gate_sync_async_and_background(background):
    calls = []

    @tool_config(required_resources=[READ], background=background, retry=False)
    def read() -> str:
        """Return the report."""
        calls.append("read")
        return "report"

    library = ToolLibrary("files", [read])
    definition = library.get_tool_definition("read")
    assert definition.required_resources == (READ,)
    assert "file:report" not in repr(definition.input_schema)
    with pytest.raises(RuntimeError, match="resource permissions"):
        library.run("read", {})
    with pytest.raises(RuntimeError, match="resource permissions"):
        await library.arun("read", {})
    assert calls == []
    if not background:
        with execution_context(
            scope=ExecutionScope(permissions=PermissionSet(resources=[READ]))
        ):
            assert library.run("read", {}) == "report"
            assert await library.arun("read", {}) == "report"
        assert calls == ["read", "read"]
    stripped = msgspec.structs.replace(definition, required_resources=())
    denied = library._permission_outcome(ToolIntent(id="call", name="read"), stripped)
    assert denied.status == "blocked"


@pytest.mark.parametrize("mechanisms", [[], ["filesystem"], ["filesystem", "network"]])
def test_unsupported_isolation_fails_closed(mechanisms):
    requested = SandboxRequirements(["filesystem", "network", "process"])
    with pytest.raises(PermissionError, match="Unsupported isolation"):
        SandboxCapabilities(mechanisms).require(requested)
    SandboxCapabilities(requested.mechanisms).require(requested)


def test_security_contract_validation():
    with pytest.raises(ValueError):
        ResourcePermission("", "filesystem.read")
    with pytest.raises(ValueError):
        ResourcePermission("file:report", "*")
    with pytest.raises(TypeError):
        PermissionSet(resources="file:report")
    with pytest.raises(ValueError):
        SandboxCapabilities(["pretend"])
    with pytest.raises(TypeError):
        SandboxRequirements("filesystem")


def test_approval_binds_workspace_and_static_resources():
    @tool_config(required_resources=[READ])
    def read() -> str:
        """Protected read."""
        return "read"

    library = ToolLibrary("files", [read])
    policy = AgentApprovals(InMemoryApprovalStore(), {"read": "v1"}, "p1")
    intent = ToolIntent(id="call", name="read")
    first = ExecutionScope(
        namespace="files",
        thread_id="t",
        run_id="r",
        principal="user",
        permissions=PermissionSet(resources=[READ]),
        environment=ExecutionEnvironment(InMemoryWorkspace("one")),
    )
    with execution_context(scope=first):
        batch = policy.prepare(library, [intent], {"requests": {}})
        binding = batch.records["call"].binding
    from dataclasses import replace

    second = replace(first, environment=ExecutionEnvironment(InMemoryWorkspace("two")))
    with execution_context(scope=second):
        assert (
            policy.binding(library, intent).resources_digest != binding.resources_digest
        )
        with pytest.raises(TaskPauseRequestedError, match="binding changed"):
            policy.prepare(library, [intent], {"requests": {}})


def test_background_resource_denial_precedes_resume_and_execution():
    @tool_config(required_resources=[READ], retry=False)
    def read() -> str:
        """Protected read."""
        return "read"

    library = ToolLibrary("files", [read])
    handle = Mock()
    handle.get_tool_definition.return_value = library.get_tool_definition("read")
    dispatcher = BackgroundTaskDispatcher(handle)
    with pytest.raises(PermissionError, match="resource permissions"):
        dispatcher.resume_agent_task(
            task=SimpleNamespace(tool_name="read"), message="resume"
        )
    handle.get_task_store.return_value.requeue.assert_not_called()
    tool, task_handle = Mock(), Mock()
    with pytest.raises(PermissionError, match="resource permissions"):
        dispatcher.run_tool(
            tool=tool,
            task_handle=task_handle,
            tool_name="read",
            call_params={},
            required_resources=(READ,),
        )
    tool.assert_not_called()
