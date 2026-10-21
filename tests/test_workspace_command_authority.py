"""Host command controls require fresh, live workspace authority."""

from __future__ import annotations

import pytest

from msgflux.runtime import (
    AgentWorkspace,
    ExecutionScope,
    PermissionSet,
    execution_context,
)
from msgflux.runtime.workspace.backend import InMemoryWorkspaceBackend, WorkspaceBinding
from msgflux.runtime.workspace.environment import ProcessExecutor
from msgflux.runtime.workspace.filesystem import InMemoryWorkspace
from msgflux.runtime.workspace.receipts import CommandReceipt
from msgflux.runtime.workspace.references import encode_workspace_reference


class _CountingExecutor(ProcessExecutor):
    def __init__(self):
        self.inspect_calls = 0
        self.terminate_calls = 0

    @property
    def capabilities(self):
        from msgflux.runtime.isolation import SandboxCapabilities

        return SandboxCapabilities(
            {"filesystem", "network", "process", "resource_limits"}
        )

    def supports_workspace(self, filesystem):
        return isinstance(filesystem, InMemoryWorkspace)

    async def execute_stream(self, *args, **kwargs):
        raise AssertionError("execution is not part of this test")

    async def inspect_command(self, receipt):
        self.inspect_calls += 1
        raise AssertionError("denied inspection reached executor")

    async def terminate_command(self, receipt):
        self.terminate_calls += 1
        raise AssertionError("denied termination reached executor")


class _Backend(InMemoryWorkspaceBackend):
    def __init__(self):
        super().__init__()
        self.executor = _CountingExecutor()

    async def open(self, workspace_id, *, abort_signal=None):
        filesystem = InMemoryWorkspace(workspace_id)
        binding = WorkspaceBinding(self, filesystem, self.executor, ownership="owned")
        self._resources[filesystem.identity] = filesystem
        return binding


async def _workspace(backend=None, *, permissions=None):
    backend = backend or _Backend()
    workspace = await AgentWorkspace.open(
        backend,
        "workspace",
        permissions=PermissionSet({"process.execute"})
        if permissions is None
        else permissions,
    )
    receipt = CommandReceipt(
        version=1,
        execution_id="execution-test",
        state="launched",
        workspace_reference=encode_workspace_reference(workspace),
        backend=workspace._environment.filesystem.identity.backend,
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:00:00Z",
    )
    return workspace, backend.executor, receipt


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["ainspect_command", "aterminate_command"])
async def test_command_host_controls_require_live_process_permission(method):
    workspace, executor, receipt = await _workspace()
    try:
        with (
            execution_context(
                scope=ExecutionScope(
                    workspace=workspace,
                    permissions=PermissionSet(),
                )
            ),
            pytest.raises(PermissionError),
        ):
            await getattr(workspace, method)(receipt)
        assert executor.inspect_calls == executor.terminate_calls == 0
    finally:
        await workspace.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["ainspect_command", "aterminate_command"])
async def test_command_host_controls_reject_a_foreign_live_workspace(method):
    workspace, executor, receipt = await _workspace()
    foreign, _foreign_executor, _ = await _workspace()
    try:
        with (
            execution_context(
                scope=ExecutionScope(
                    workspace=foreign,
                    permissions=PermissionSet({"process.execute"}),
                )
            ),
            pytest.raises(PermissionError),
        ):
            await getattr(workspace, method)(receipt)
        assert executor.inspect_calls == executor.terminate_calls == 0
    finally:
        await foreign.aclose()
        await workspace.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["ainspect_command", "aterminate_command"])
async def test_command_host_controls_reject_closed_binding(method):
    workspace, executor, receipt = await _workspace()
    await workspace.aclose()
    with (
        execution_context(
            scope=ExecutionScope(
                workspace=workspace,
                permissions=PermissionSet({"process.execute"}),
            )
        ),
        pytest.raises(PermissionError),
    ):
        await getattr(workspace, method)(receipt)
    assert executor.inspect_calls == executor.terminate_calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["ainspect_command", "aterminate_command"])
async def test_saved_receipt_does_not_grant_workspace_process_permission(method):
    workspace, executor, receipt = await _workspace(permissions=PermissionSet())
    try:
        with pytest.raises(PermissionError):
            await getattr(workspace, method)(receipt)
        assert executor.inspect_calls == executor.terminate_calls == 0
    finally:
        await workspace.aclose()
