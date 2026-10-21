"""Crash/restart coverage for persistent local workspace bindings."""

import asyncio
import multiprocessing
import os

import pytest

from msgflux.runtime import AgentWorkspace, PermissionSet, SandboxRequirements
from msgflux.runtime.workspace.local import LocalWorkspaceBackend
from msgflux.runtime.workspace.registry import SQLiteWorkspaceRegistry


WORKSPACE_ID = "restart-project"
STATE_PATH = "/state.txt"


def _permissions(*actions):
    from msgflux.runtime.permissions import ResourcePermission

    return PermissionSet(
        {f"filesystem.{action}" for action in actions},
        {
            ResourcePermission(
                f"workspace:{WORKSPACE_ID}:{STATE_PATH}", f"filesystem.{action}"
            )
            for action in actions
        },
    )


def _open_settings(registry_path, root, permissions):
    backend = LocalWorkspaceBackend(
        root, registry=SQLiteWorkspaceRegistry(registry_path)
    )
    return backend, {
        "permissions": permissions,
        "requirements": SandboxRequirements(),
        "write_guarantee": "cooperative_compare",
    }


def _crash_after_write(registry_path, root, identity_pipe):
    """Register and write, then die without running Python cleanup handlers."""
    backend, options = _open_settings(
        registry_path, root, _permissions("read", "write")
    )
    workspace = asyncio.run(AgentWorkspace.open(backend, WORKSPACE_ID, **options))
    workspace.write_text(STATE_PATH, "committed before process exit")
    identity_pipe.send(
        (
            workspace.identity.backend,
            workspace.identity.resource_id,
            workspace.identity.generation,
            workspace.identity.config_revision,
        )
    )
    identity_pipe.close()
    os._exit(0)


def _reconnect_after_restart(registry_path, root, fields, result_pipe):
    """A second process reconnects with a read-only grant and reports evidence."""
    backend, options = _open_settings(registry_path, root, _permissions("read"))
    workspace = asyncio.run(
        AgentWorkspace.reconnect(backend, WORKSPACE_ID, _identity(fields), **options)
    )
    content = workspace.read_text(STATE_PATH)
    try:
        workspace.write_text(STATE_PATH, "must not be allowed")
    except PermissionError:
        denied = True
    else:
        denied = False
    result_pipe.send((content, denied, workspace.identity.generation))
    result_pipe.close()
    os._exit(0)


def _concurrent_first_open(registry_path, root, start_gate, ready_pipe, result_pipe):
    """Wait for a peer process, then race initial registry creation and open."""
    ready_pipe.send("ready")
    ready_pipe.close()
    if not start_gate.wait(10):
        os._exit(2)
    backend, options = _open_settings(
        registry_path, root, _permissions("read", "write")
    )
    workspace = asyncio.run(AgentWorkspace.open(backend, WORKSPACE_ID, **options))
    result_pipe.send(
        (
            workspace.identity.backend,
            workspace.identity.resource_id,
            workspace.identity.generation,
            workspace.identity.config_revision,
        )
    )
    result_pipe.close()
    os._exit(0)


def _identity(fields):
    from msgflux.runtime.workspace.contracts import WorkspaceIdentity

    return WorkspaceIdentity(
        backend=fields[0],
        resource_id=fields[1],
        generation=fields[2],
        config_revision=fields[3],
    )


def _run_abrupt_child(target, args):
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=target, args=(*args, sender))
    process.start()
    sender.close()
    try:
        assert receiver.poll(10), "child did not return reconnection evidence"
        result = receiver.recv()
        process.join(10)
        assert not process.is_alive(), "reconnection child did not terminate"
        assert process.exitcode == 0
        return result
    finally:
        receiver.close()
        if process.is_alive():
            process.terminate()
            process.join(5)
        if process.is_alive():
            process.kill()
            process.join(5)


def _abrupt_child(registry_path, root):
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(
        target=_crash_after_write, args=(str(registry_path), str(root), sender)
    )
    process.start()
    sender.close()
    try:
        assert receiver.poll(10), "child did not publish the registered identity"
        identity = _identity(receiver.recv())
        process.join(10)
        assert not process.is_alive(), "abrupt-exit child did not terminate"
        assert process.exitcode == 0
        return identity
    finally:
        receiver.close()
        if process.is_alive():
            process.terminate()
            process.join(5)
        if process.is_alive():
            process.kill()
            process.join(5)


@pytest.mark.skipif(os.name != "posix", reason="POSIX local backend")
def test_concurrent_first_registration_across_processes_has_one_generation(tmp_path):
    context = multiprocessing.get_context("spawn")
    root = tmp_path / "workspace"
    root.mkdir()
    registry_path = tmp_path / "host-state" / "workspace.sqlite"
    start_gate = context.Event()
    processes = []
    channels = []

    try:
        for _ in range(2):
            ready_receiver, ready_sender = context.Pipe(duplex=False)
            result_receiver, result_sender = context.Pipe(duplex=False)
            process = context.Process(
                target=_concurrent_first_open,
                args=(
                    str(registry_path),
                    str(root),
                    start_gate,
                    ready_sender,
                    result_sender,
                ),
            )
            process.start()
            ready_sender.close()
            result_sender.close()
            processes.append(process)
            channels.append((ready_receiver, result_receiver))

        # Release both independent processes together at schema creation and
        # first registration after each has reached the same synchronization point.
        for ready_receiver, _ in channels:
            assert ready_receiver.poll(10), "registration worker did not become ready"
            assert ready_receiver.recv() == "ready"
        start_gate.set()

        identities = []
        for _, result_receiver in channels:
            assert result_receiver.poll(10), "registration worker returned no identity"
            identities.append(_identity(result_receiver.recv()))
        for process in processes:
            process.join(10)
            assert not process.is_alive(), "registration worker did not terminate"
            assert process.exitcode == 0

        registry = SQLiteWorkspaceRegistry(registry_path)
        assert registry.get(WORKSPACE_ID) == identities[0]
        registry.close()
        assert identities[0] == identities[1]
    finally:
        start_gate.set()
        for ready_receiver, result_receiver in channels:
            ready_receiver.close()
            result_receiver.close()
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(5)
            if process.is_alive():
                process.kill()
                process.join(5)


@pytest.mark.skipif(os.name != "posix", reason="POSIX local backend")
def test_process_restart_reconnects_identity_content_and_fresh_permissions(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    registry_path = tmp_path / "host-state" / "workspace.sqlite"

    identity = _abrupt_child(registry_path, root)

    observed, write_was_denied, generation = _run_abrupt_child(
        _reconnect_after_restart,
        (
            str(registry_path),
            str(root),
            (
                identity.backend,
                identity.resource_id,
                identity.generation,
                identity.config_revision,
            ),
        ),
    )
    assert observed == "committed before process exit"
    assert write_was_denied
    assert generation == identity.generation

    # A later host can supply a broader current grant without replacing the resource.
    backend, options = _open_settings(registry_path, root, _permissions("read"))
    resumed = asyncio.run(
        AgentWorkspace.reconnect(backend, WORKSPACE_ID, identity, **options)
    )
    assert resumed.identity == identity
    assert resumed.read_text(STATE_PATH) == "committed before process exit"
    with pytest.raises(PermissionError):
        resumed.write_text(STATE_PATH, "must not be allowed")
    asyncio.run(resumed.aclose())

    backend2, options2 = _open_settings(
        registry_path, root, _permissions("read", "write")
    )
    resumed2 = asyncio.run(
        AgentWorkspace.reconnect(backend2, WORKSPACE_ID, identity, **options2)
    )
    resumed2.write_text(STATE_PATH, "new host grant")
    assert resumed2.read_text(STATE_PATH) == "new host grant"
    asyncio.run(resumed2.aclose())


@pytest.mark.skipif(os.name != "posix", reason="POSIX local backend")
def test_reconnect_rejects_replaced_root_without_adopting_it(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    registry_path = tmp_path / "host-state" / "workspace.sqlite"
    identity = _abrupt_child(registry_path, root)

    retired = tmp_path / "workspace-retired"
    root.rename(retired)
    root.mkdir()
    (root / "state.txt").write_text("replacement root")

    backend, options = _open_settings(
        registry_path, root, _permissions("read", "write")
    )
    with pytest.raises((PermissionError, FileNotFoundError)):
        asyncio.run(
            AgentWorkspace.reconnect(backend, WORKSPACE_ID, identity, **options)
        )
    assert (retired / "state.txt").read_text() == "committed before process exit"
    assert (root / "state.txt").read_text() == "replacement root"


@pytest.mark.skipif(os.name != "posix", reason="POSIX local backend")
def test_reconnect_requires_existing_registry_entry(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    registry_path = tmp_path / "host-state" / "workspace.sqlite"
    backend, options = _open_settings(registry_path, root, _permissions("read"))
    with pytest.raises(FileNotFoundError):
        asyncio.run(
            AgentWorkspace.reconnect(
                backend,
                WORKSPACE_ID,
                _identity(("local-posix", WORKSPACE_ID, "missing", "1")),
                **options,
            )
        )
