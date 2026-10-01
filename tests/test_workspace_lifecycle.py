"""AgentWorkspace owns backend bindings opened through its explicit factory."""

import pytest

from msgflux.runtime import (
    AgentWorkspace,
    ExecutionEnvironment,
    InMemoryWorkspace,
    ProcessExecutor,
    ProcessResult,
    SandboxCapabilities,
    SandboxRequirements,
    WorkspaceBackend,
    WorkspaceBinding,
)


class _FakeExecutor(ProcessExecutor):
    @property
    def capabilities(self):
        return SandboxCapabilities({"filesystem"})

    def supports_workspace(self, filesystem):
        return True

    async def execute_stream(self, request, **kwargs):
        return ProcessResult(0)


class _TrackingBackend(WorkspaceBackend):
    def __init__(self, *, executor=None, fail_release=False):
        self.executor = executor
        self.fail_release = fail_release
        self.releases = 0
        self.bindings = []

    async def open(self, workspace_id, *, abort_signal=None):
        if abort_signal is not None:
            abort_signal.raise_if_aborted()
        binding = WorkspaceBinding(self, InMemoryWorkspace(workspace_id), self.executor)
        self.bindings.append(binding)
        return binding

    async def _release(self, binding):
        assert binding in self.bindings
        self.releases += 1
        if self.fail_release:
            raise RuntimeError("release failed")


@pytest.mark.asyncio
async def test_opened_agent_workspace_owns_binding_and_close_is_idempotent():
    backend = _TrackingBackend()
    workspace = await AgentWorkspace.open(backend, "project")

    assert workspace.identity == backend.bindings[0].identity
    workspace.require_active()
    await workspace.aclose()
    await workspace.aclose()

    assert backend.releases == 1
    with pytest.raises(PermissionError, match="not open"):
        workspace.require_active()


@pytest.mark.asyncio
async def test_cwd_views_share_binding_and_are_invalidated_by_owner_close():
    backend = _TrackingBackend()
    workspace = await AgentWorkspace.open(backend, "project")
    view = workspace.with_cwd("src")

    assert view.cwd == "/src"
    assert view.shares_environment(workspace)
    await view.aclose()
    assert backend.releases == 0
    view.require_active()

    await workspace.aclose()
    with pytest.raises(PermissionError, match="not open"):
        view.require_active()
    assert backend.releases == 1


@pytest.mark.asyncio
async def test_from_environment_is_borrowed_and_does_not_close_binding():
    backend = _TrackingBackend()
    binding = await backend.open("project")
    wrapper = AgentWorkspace.from_environment(
        ExecutionEnvironment.from_binding(binding)
    )
    await wrapper.aclose()

    assert binding.state == "open"
    assert backend.releases == 0
    await binding.aclose()
    assert backend.releases == 1


@pytest.mark.asyncio
async def test_open_releases_binding_when_environment_construction_fails():
    backend = _TrackingBackend(executor=_FakeExecutor())

    with pytest.raises(PermissionError, match="Unsupported isolation"):
        await AgentWorkspace.open(
            backend,
            "project",
            requirements=SandboxRequirements({"network"}),
        )

    assert backend.releases == 1
    assert backend.bindings[0].state == "closed"


@pytest.mark.asyncio
async def test_open_releases_binding_when_workspace_construction_fails():
    backend = _TrackingBackend()

    with pytest.raises(ValueError, match="absolute virtual POSIX path"):
        await AgentWorkspace.open(backend, "project", cwd="../escape")

    assert backend.releases == 1
    assert backend.bindings[0].state == "closed"


@pytest.mark.asyncio
async def test_workspace_close_surfaces_failed_release_without_retrying():
    backend = _TrackingBackend(fail_release=True)
    workspace = await AgentWorkspace.open(backend, "project")

    with pytest.raises(RuntimeError, match="release failed"):
        await workspace.aclose()
    with pytest.raises(RuntimeError, match="reconciliation"):
        await workspace.aclose()

    assert backend.releases == 1
