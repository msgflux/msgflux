import asyncio

import msgspec
import pytest

from msgflux.exceptions import AbortRequestedError
from msgflux.runtime import (
    AbortSignal,
    ExecutionEnvironment,
    ExecutionScope,
    InMemoryWorkspace,
    InMemoryWorkspaceBackend,
    PermissionSet,
    ProcessExecutor,
    ProcessRequest,
    ProcessResult,
    SandboxCapabilities,
    SandboxRequirements,
    WorkspaceBackend,
    WorkspaceBinding,
    execution_context,
)


ALL_MECHANISMS = {"filesystem", "network", "process", "resource_limits"}


class FakeExecutor(ProcessExecutor):
    def __init__(self, supported=True, mechanisms=ALL_MECHANISMS):
        self.supported = supported
        self._capabilities = SandboxCapabilities(mechanisms)

    @property
    def capabilities(self):
        return self._capabilities

    def supports_workspace(self, filesystem):
        return self.supported

    async def execute_stream(self, request, **kwargs):
        return ProcessResult(0)


class TrackingBackend(WorkspaceBackend):
    def __init__(self, *, fail=False, block=False):
        self.fail = fail
        self.block = block
        self.releases = 0
        self.release_started = asyncio.Event()
        self.release_gate = asyncio.Event()

    async def open(self, workspace_id, *, abort_signal=None):
        if abort_signal is not None:
            abort_signal.raise_if_aborted()
        return WorkspaceBinding(self, InMemoryWorkspace(workspace_id))

    async def _release(self, binding):
        self.releases += 1
        self.release_started.set()
        if self.block:
            await self.release_gate.wait()
        if self.fail:
            raise RuntimeError("release failed")


@pytest.mark.asyncio
async def test_memory_backend_opens_independent_resources_and_reconnects_after_close():
    backend = InMemoryWorkspaceBackend({"/a": b"initial"})
    first = await backend.open("project")
    second = await backend.open("project")
    assert first.filesystem is not second.filesystem
    assert first.identity != second.identity

    await first.aclose()
    resumed = await backend.reconnect("project", first.identity)
    assert resumed.filesystem is first.filesystem
    assert resumed.identity == first.identity
    await resumed.aclose()


@pytest.mark.asyncio
async def test_memory_backend_reconnect_rejects_workspace_or_identity_changes():
    backend = InMemoryWorkspaceBackend()
    binding = await backend.open("project")
    identity = binding.identity
    for workspace_id, changed in (
        ("other", identity),
        ("project", msgspec.structs.replace(identity, generation="different")),
        ("project", msgspec.structs.replace(identity, config_revision="2")),
        ("project", msgspec.structs.replace(identity, backend="other.backend")),
    ):
        with pytest.raises(FileNotFoundError):
            await backend.reconnect(workspace_id, changed)
    await binding.aclose()


@pytest.mark.asyncio
async def test_binding_close_is_successful_and_idempotent():
    backend = TrackingBackend()
    binding = await backend.open("project")
    assert binding.state == "open"
    await binding.aclose()
    await binding.aclose()
    assert binding.state == "closed"
    assert backend.releases == 1
    with pytest.raises(PermissionError):
        binding.require_active()


@pytest.mark.asyncio
async def test_failed_release_is_fail_closed_and_not_retried():
    backend = TrackingBackend(fail=True)
    binding = await backend.open("project")
    with pytest.raises(RuntimeError, match="release failed"):
        await binding.aclose()
    assert binding.state == "release_failed"
    with pytest.raises(RuntimeError, match="reconciliation"):
        await binding.aclose()
    assert backend.releases == 1


@pytest.mark.asyncio
async def test_cancelled_release_is_fail_closed_and_not_retried():
    backend = TrackingBackend(block=True)
    binding = await backend.open("project")
    closing = asyncio.create_task(binding.aclose())
    await asyncio.wait_for(backend.release_started.wait(), timeout=2)
    closing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await closing
    assert binding.state == "release_failed"
    with pytest.raises(RuntimeError, match="reconciliation"):
        await binding.aclose()
    assert backend.releases == 1


@pytest.mark.asyncio
async def test_borrowed_release_retains_resource_contents():
    backend = TrackingBackend()
    filesystem = InMemoryWorkspace("project", {"/a": b"kept"})
    binding = WorkspaceBinding(backend, filesystem, ownership="borrowed")
    environment = ExecutionEnvironment.from_binding(binding)
    permissions = PermissionSet(
        resources=[filesystem.permission("/a", "filesystem.read")]
    )
    with execution_context(
        scope=ExecutionScope(environment=environment, permissions=permissions)
    ):
        assert filesystem.read_bytes("/a") == b"kept"
    await binding.aclose()
    assert filesystem._files["/a"] == b"kept"


def test_binding_rejects_incompatible_executor():
    filesystem = InMemoryWorkspace("project")
    with pytest.raises(PermissionError, match="cannot use"):
        WorkspaceBinding(TrackingBackend(), filesystem, FakeExecutor(supported=False))


def test_environment_from_binding_rejects_mismatched_services_and_capabilities():
    backend = TrackingBackend()
    filesystem = InMemoryWorkspace("project")
    executor = FakeExecutor()
    binding = WorkspaceBinding(backend, filesystem, executor)
    with pytest.raises(ValueError, match="belong to its binding"):
        ExecutionEnvironment(filesystem=InMemoryWorkspace("other"), binding=binding)
    limited_binding = WorkspaceBinding(
        backend, InMemoryWorkspace("limited"), FakeExecutor(mechanisms={"filesystem"})
    )
    with pytest.raises(PermissionError, match="Unsupported isolation"):
        ExecutionEnvironment.from_binding(
            limited_binding,
            requirements=SandboxRequirements({"network", "process"}),
        )


@pytest.mark.asyncio
async def test_closed_environment_blocks_filesystem_access():
    backend = TrackingBackend()
    binding = await backend.open("project")
    environment = ExecutionEnvironment.from_binding(binding)
    filesystem = binding.filesystem
    permissions = PermissionSet(
        resources=[filesystem.permission("/a", "filesystem.read")]
    )
    with execution_context(
        scope=ExecutionScope(environment=environment, permissions=permissions)
    ):
        await binding.aclose()
        with pytest.raises(PermissionError, match="not open"):
            filesystem.read_bytes("/a")


@pytest.mark.asyncio
async def test_scope_serialization_excludes_live_binding_handles():
    binding = await TrackingBackend().open("project")
    environment = ExecutionEnvironment.from_binding(binding)
    scope = ExecutionScope(environment=environment, principal="user")
    serialized = scope.to_dict()
    assert "environment" not in serialized
    assert "binding" not in serialized
    with pytest.raises(TypeError):
        msgspec.json.encode(binding)
    await binding.aclose()


@pytest.mark.asyncio
async def test_abort_before_open_and_reconnect_is_checked():
    backend = InMemoryWorkspaceBackend()
    signal = AbortSignal()
    signal.abort("stop")
    with pytest.raises(AbortRequestedError):
        await backend.open("project", abort_signal=signal)
    binding = await backend.open("project")
    with pytest.raises(AbortRequestedError):
        await backend.reconnect("project", binding.identity, abort_signal=signal)
    await binding.aclose()


@pytest.mark.asyncio
async def test_nested_execution_cannot_replace_live_environment():
    first = await TrackingBackend().open("first")
    second = await TrackingBackend().open("second")
    first_environment = ExecutionEnvironment.from_binding(first)
    second_environment = ExecutionEnvironment.from_binding(second)
    with execution_context(scope=ExecutionScope(environment=first_environment)):
        with pytest.raises(ValueError, match="replace its environment"):
            with execution_context(
                scope=ExecutionScope(environment=second_environment)
            ):
                pass
    await first.aclose()
    await second.aclose()


@pytest.mark.asyncio
async def test_closing_gates_new_operations_and_concurrent_close_releases_once():
    from types import SimpleNamespace

    from msgflux.runtime import AgentApprovals

    backend = TrackingBackend(block=True)
    fs = InMemoryWorkspace("project", {"/a": b"original"})
    binding = WorkspaceBinding(backend, fs, FakeExecutor())
    environment = ExecutionEnvironment.from_binding(binding)
    scope = ExecutionScope(environment=environment)
    first = asyncio.create_task(binding.aclose())
    await asyncio.wait_for(backend.release_started.wait(), timeout=2)
    try:
        assert binding.state == "closing"
        with execution_context(scope=scope):
            with pytest.raises(PermissionError, match="not open"):
                fs.read_bytes("/a")
            with pytest.raises(PermissionError, match="not open"):
                await environment.arun(ProcessRequest(("echo", "hello")))
            with pytest.raises(PermissionError, match="not open"):
                AgentApprovals._resource_binding(
                    SimpleNamespace(required_resources=()), scope
                )
        second = asyncio.create_task(binding.aclose())
        backend.release_gate.set()
        await asyncio.wait_for(asyncio.gather(first, second), timeout=2)
        assert backend.releases == 1
        with pytest.raises(PermissionError, match="not open"):
            ExecutionEnvironment.from_binding(binding)
    finally:
        backend.release_gate.set()
        await first


@pytest.mark.asyncio
async def test_reconnected_binding_survives_other_binding_close():
    backend = InMemoryWorkspaceBackend({"/a": b"shared"})
    first = await backend.open("project")
    second = await backend.reconnect("project", first.identity)
    await first.aclose()
    fs = second.filesystem
    environment = ExecutionEnvironment.from_binding(second)
    with execution_context(
        scope=ExecutionScope(
            environment=environment,
            permissions=PermissionSet(
                resources=[fs.permission("/a", "filesystem.read")]
            ),
        )
    ):
        assert fs.read_bytes("/a") == b"shared"
    await second.aclose()
    with pytest.raises(FileNotFoundError):
        await InMemoryWorkspaceBackend().reconnect("project", first.identity)


@pytest.mark.asyncio
async def test_managed_filesystem_cannot_escape_binding_lifecycle():
    from dataclasses import replace

    fs = InMemoryWorkspace("project", {"/a": b"original"})
    old_environment = ExecutionEnvironment(fs)
    binding = WorkspaceBinding(TrackingBackend(), fs)
    environment = ExecutionEnvironment.from_binding(binding)
    permissions = PermissionSet(resources=[fs.permission("/a", "filesystem.read")])
    for closed in (False, True):
        if closed:
            await binding.aclose()
        with pytest.raises(ValueError, match="requires a workspace binding"):
            ExecutionEnvironment(fs)
        with pytest.raises(ValueError, match="requires a workspace binding"):
            replace(environment, binding=None)
        with execution_context(
            scope=ExecutionScope(
                environment=old_environment,
                permissions=permissions,
            )
        ):
            with pytest.raises(PermissionError, match="requires a workspace binding"):
                fs.read_bytes("/a")
