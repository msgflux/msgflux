"""Small host identity invariants used by executor recovery inspection."""

import asyncio
import os
from types import SimpleNamespace

import pytest

from msgflux.runtime import AgentWorkspace, ExecutionEnvironment, PermissionSet
from msgflux.runtime.workspace.command_inspection import (
    local_identity_matches,
    local_process_identity,
)
from msgflux.runtime.workspace.docker_executor import DockerProcessExecutor
from msgflux.runtime.workspace.local import LocalWorkspace
from msgflux.runtime.workspace.receipts import CommandReceipt
from msgflux.runtime.workspace.references import encode_workspace_reference
from msgflux.runtime.isolation import SandboxRequirements


def test_local_inspection_requires_boot_and_process_start_identity():
    # This process is a stable live identity for the duration of the assertion.
    identity = local_process_identity(os.getpid())
    assert identity is not None
    assert identity["state"] not in {"Z", "X"}

    matches, status, _reason = local_identity_matches(identity)
    assert (matches, status) == (True, "present")

    changed = {**identity, "start_ticks": identity["start_ticks"] + 1}
    matches, status, _reason = local_identity_matches(changed)
    assert (matches, status) == (False, "mismatch")


def test_local_inspection_never_treats_pid_alone_as_authority():
    matches, status, reason = local_identity_matches({"pid": os.getpid()})
    assert matches is None
    assert status == "unavailable"
    assert "boot/PID-start" in reason


@pytest.mark.asyncio
async def test_docker_terminate_blocks_replaced_socket_before_cli(
    tmp_path, monkeypatch
):
    from msgflux.runtime.workspace import docker_executor as docker_module

    socket_path = tmp_path / "docker.sock"
    socket_path.touch()
    filesystem = LocalWorkspace("socket-test", tmp_path)
    monkeypatch.setattr(docker_module.shutil, "which", lambda _name: "/usr/bin/docker")
    executor = DockerProcessExecutor(
        filesystem, image="trusted", socket_path=str(socket_path)
    )
    workspace = AgentWorkspace.from_environment(
        ExecutionEnvironment(
            filesystem,
            executor,
            requirements=SandboxRequirements(),
        ),
        permissions=PermissionSet({"process.execute"}),
    )
    recorded_socket = executor._daemon_identity()
    receipt = CommandReceipt(
        version=1,
        execution_id="execution-socket-test",
        state="launched",
        workspace_reference=encode_workspace_reference(workspace),
        backend=filesystem.identity.backend,
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:00:00Z",
        resource={
            "container_id": "a" * 64,
            "image_id": "sha256:" + "b" * 64,
            "image": "trusted",
            "name": "msgflux-test",
            "daemon_config_revision": filesystem.identity.config_revision,
            "daemon_identity": recorded_socket,
        },
    )

    original_stat = os.stat
    calls = []

    def replaced_stat(path, *args, **kwargs):
        if os.fspath(path) == str(socket_path):
            return SimpleNamespace(st_dev=recorded_socket["device"] + 1, st_ino=1)
        return original_stat(path, *args, **kwargs)

    async def fake_create_subprocess_exec(*args, **kwargs):
        calls.append(args)
        raise AssertionError(
            "Docker CLI must not be contacted through a replaced socket"
        )

    monkeypatch.setattr(docker_module.os, "stat", replaced_stat)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    inspection = await executor.terminate_command(receipt)
    assert inspection.classification == "blocked"
    assert inspection.resource_status == "mismatch"
    assert calls == []
