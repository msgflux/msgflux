from __future__ import annotations

import asyncio

import pytest

from msgflux.runtime import InMemoryWorkspace, WorkspaceBackend, WorkspaceBinding


class _ReleaseBackend(WorkspaceBackend):
    def __init__(self, cleanup_error=None):
        self.cleanup_error = cleanup_error
        self.release_calls = 0

    async def open(self, workspace_id, *, abort_signal=None):
        return WorkspaceBinding(self, InMemoryWorkspace(workspace_id))

    async def _release(self, binding):
        self.release_calls += 1
        if self.cleanup_error is not None:
            raise self.cleanup_error


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body_error",
    [
        ValueError("workspace operation failed"),
        asyncio.CancelledError("workspace operation cancelled"),
    ],
    ids=["ordinary-error", "body-cancellation"],
)
async def test_context_exit_preserves_body_error_when_cleanup_fails(body_error):
    cleanup_error = OSError("binding close failed")
    backend = _ReleaseBackend(cleanup_error)
    binding = await backend.open("project")

    with pytest.raises(type(body_error)) as exc_info:
        async with binding:
            raise body_error

    assert exc_info.value is body_error
    assert any("binding close failed" in note for note in body_error.__notes__)
    assert binding.state == "release_failed"
    assert backend.release_calls == 1


@pytest.mark.asyncio
async def test_context_exit_without_body_error_propagates_cleanup_failure():
    cleanup_error = OSError("binding close failed")
    backend = _ReleaseBackend(cleanup_error)
    binding = await backend.open("project")

    with pytest.raises(OSError) as exc_info:
        async with binding:
            pass

    assert exc_info.value is cleanup_error
    assert binding.state == "release_failed"
    assert backend.release_calls == 1


@pytest.mark.asyncio
async def test_context_exit_does_not_suppress_cleanup_cancellation():
    cleanup_error = asyncio.CancelledError("cleanup cancelled")
    backend = _ReleaseBackend(cleanup_error)
    binding = await backend.open("project")
    body_error = ValueError("workspace operation failed")

    with pytest.raises(asyncio.CancelledError) as exc_info:
        async with binding:
            raise body_error

    assert exc_info.value is cleanup_error
    assert exc_info.value.__context__ is body_error
    assert binding.state == "release_failed"
    assert backend.release_calls == 1
