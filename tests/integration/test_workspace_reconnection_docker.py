"""Opt-in real-daemon reconnection coverage; never pulls an image."""

import asyncio
import os
import shutil
import subprocess

import pytest

from msgflux.runtime import AgentWorkspace, DockerWorkspaceBackend, PermissionSet
from msgflux.runtime.permissions import ResourcePermission
from msgflux.runtime.workspace.registry import SQLiteWorkspaceRegistry


WORKSPACE_ID = "docker-restart-project"


@pytest.fixture
def docker_restart_setup(tmp_path):
    if os.environ.get("MSGFLUX_TEST_DOCKER") != "1":
        pytest.skip("Set MSGFLUX_TEST_DOCKER=1 for real Docker reconnection tests")
    docker = shutil.which("docker")
    if docker is None:
        pytest.skip("Docker CLI is not installed")
    try:
        image_id = subprocess.check_output(  # noqa: S603 -- fixed Docker CLI and host-owned image lookup
            [docker, "image", "inspect", "python:3.12-slim", "--format", "{{.Id}}"],
            text=True,
            timeout=10,
        ).strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        pytest.skip(
            f"python:3.12-slim is not present locally; no pull attempted: {exc}"
        )
    if not image_id.startswith("sha256:"):
        pytest.skip("Docker did not return an immutable image ID")
    root = tmp_path / "workspace"
    root.mkdir()
    registry_path = tmp_path / "host-state" / "workspace.sqlite"
    return root, registry_path, image_id


def _permissions(*, execute):
    grants = {"filesystem.read", "filesystem.write"}
    resources = {
        ResourcePermission(f"workspace:{WORKSPACE_ID}:/state.txt", "filesystem.read"),
        ResourcePermission(f"workspace:{WORKSPACE_ID}:/state.txt", "filesystem.write"),
        ResourcePermission(
            f"workspace:{WORKSPACE_ID}:/from-bash.txt", "filesystem.read"
        ),
    }
    if execute:
        grants.update({"process.execute", "process.workspace"})
        resources.add(
            ResourcePermission(f"workspace:{WORKSPACE_ID}:/", "process.workspace")
        )
    return PermissionSet(grants, resources)


def _backend(root, registry_path, image_id):
    return DockerWorkspaceBackend(
        root,
        image=image_id,
        registry=SQLiteWorkspaceRegistry(registry_path),
    )


@pytest.mark.asyncio
async def test_real_docker_binding_reconnects_after_application_restart(
    docker_restart_setup,
):
    root, registry_path, image_id = docker_restart_setup
    first_backend = _backend(root, registry_path, image_id)
    first = await AgentWorkspace.open(
        first_backend,
        WORKSPACE_ID,
        permissions=_permissions(execute=True),
        write_guarantee="cooperative_compare",
    )
    first.write_text("/state.txt", "written before restart")
    result = await first.arun("printf 'written by bash' > from-bash.txt")
    # Establish that the command ran in the requested immutable image.
    version = await first.arun("python -c 'import sys; print(sys.version_info.major)'")
    assert result.returncode == 0, result.stderr
    assert version.returncode == 0, version.stderr
    assert version.stdout.strip() == b"3"
    identity = first.identity
    await first.aclose()

    # Fresh registry and backend instances model a newly started application.
    second_backend = _backend(root, registry_path, image_id)
    resumed = await AgentWorkspace.reconnect(
        second_backend,
        WORKSPACE_ID,
        identity,
        permissions=_permissions(execute=True),
        write_guarantee="cooperative_compare",
    )
    assert resumed.identity == identity
    assert resumed.read_text("/state.txt") == "written before restart"
    assert resumed.read_text("/from-bash.txt") == "written by bash"
    result = await resumed.arun("cat state.txt")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == b"written before restart"
    await resumed.aclose()

    denied_backend = _backend(root, registry_path, image_id)
    denied = await AgentWorkspace.reconnect(
        denied_backend,
        WORKSPACE_ID,
        identity,
        permissions=_permissions(execute=False),
        write_guarantee="cooperative_compare",
    )
    with pytest.raises(PermissionError):
        await denied.arun("true")
    await denied.aclose()
