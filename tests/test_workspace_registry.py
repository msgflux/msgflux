import os
from concurrent.futures import ThreadPoolExecutor

import pytest

from msgflux.runtime.workspace.local import LocalWorkspaceBackend
from msgflux.runtime.workspace.registry import SQLiteWorkspaceRegistry
from msgflux.runtime.workspace.docker_executor import (
    DockerLimits,
    DockerWorkspaceBackend,
)


@pytest.mark.asyncio
async def test_local_identity_and_content_survive_independent_backend(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    registry_path = tmp_path / "state" / "workspaces.sqlite3"
    first_registry = SQLiteWorkspaceRegistry(registry_path)
    first_backend = LocalWorkspaceBackend(root, registry=first_registry)
    first = await first_backend.open("project")
    identity = first.identity
    assert identity.backend == "local-posix"
    (root / "data").write_bytes(b"saved")
    await first.aclose()
    first_registry.close()

    second_registry = SQLiteWorkspaceRegistry(registry_path)
    second_backend = LocalWorkspaceBackend(root, registry=second_registry)
    resumed = await second_backend.reconnect("project", identity)
    assert resumed.identity == identity
    assert (root / "data").read_bytes() == b"saved"
    await resumed.aclose()
    second_registry.close()


@pytest.mark.asyncio
async def test_registry_open_mismatch_and_missing_entry_do_not_adopt(tmp_path):
    root = tmp_path / "root"
    other = tmp_path / "other"
    root.mkdir()
    other.mkdir()
    registry = SQLiteWorkspaceRegistry(tmp_path / "registry.sqlite3")
    backend = LocalWorkspaceBackend(root, registry=registry)
    first = await backend.open("project")
    identity = first.identity
    await first.aclose()

    with pytest.raises(PermissionError, match="mismatch"):
        await LocalWorkspaceBackend(other, registry=registry).open("project")
    with pytest.raises(FileNotFoundError, match="not registered"):
        await backend.reconnect("missing", identity)
    registry.close()


@pytest.mark.asyncio
async def test_registry_reconnect_rejects_replaced_root(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    registry = SQLiteWorkspaceRegistry(tmp_path / "registry.sqlite3")
    backend = LocalWorkspaceBackend(root, registry=registry)
    first = await backend.open("project")
    identity = first.identity
    await first.aclose()
    root.rename(tmp_path / "old")
    root.mkdir()

    with pytest.raises(PermissionError, match="mismatch"):
        await LocalWorkspaceBackend(root, registry=registry).reconnect(
            "project", identity
        )
    registry.close()


@pytest.mark.asyncio
async def test_open_rejects_root_swap_after_registration_before_bind(
    tmp_path, monkeypatch
):
    root = tmp_path / "root"
    root.mkdir()
    registry = SQLiteWorkspaceRegistry(tmp_path / "registry.sqlite3")
    backend = LocalWorkspaceBackend(root, registry=registry)
    register = registry.register_or_verify

    def register_then_replace(workspace_id, **kwargs):
        identity = register(workspace_id, **kwargs)
        root.rename(tmp_path / "old-root")
        root.mkdir()
        return identity

    monkeypatch.setattr(registry, "register_or_verify", register_then_replace)
    with pytest.raises(PermissionError, match="root was replaced"):
        await backend.open("project")
    assert "project" not in backend._resources
    registry.close()


def test_registry_rejects_unknown_schema(tmp_path):
    import sqlite3

    path = tmp_path / "registry.sqlite3"
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA user_version=99")
    connection.close()
    with pytest.raises(ValueError, match="Unsupported workspace registry schema"):
        SQLiteWorkspaceRegistry(path)


def test_persistent_docker_config_requires_and_fingerprints_pinned_image(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    socket_path = tmp_path / "docker.sock"
    socket_path.touch()
    registry = SQLiteWorkspaceRegistry(tmp_path / "registry.sqlite3")
    pinned = "example@sha256:" + "a" * 64
    backend = DockerWorkspaceBackend(
        root,
        image=pinned,
        socket_path=str(socket_path),
        registry=registry,
    )
    changed = DockerWorkspaceBackend(
        root,
        image=pinned,
        limits=DockerLimits(memory_bytes=512 * 1024 * 1024),
        socket_path=str(socket_path),
        registry=registry,
    )
    assert backend._backend_kind() == "docker-local-posix"
    assert backend._configuration_revision() != changed._configuration_revision()
    mutable = DockerWorkspaceBackend(
        root, image="example:latest", socket_path=str(socket_path), registry=registry
    )
    with pytest.raises(ValueError, match="pinned image digest"):
        mutable._configuration_revision()
    registry.close()


def test_concurrent_initial_registration_keeps_one_generation(tmp_path):
    path = tmp_path / "registry.sqlite3"
    registries = [SQLiteWorkspaceRegistry(path) for _ in range(2)]
    values = {
        "backend": "local-posix",
        "resource_id": "project",
        "config_revision": "config",
        "root": "/workspace",
        "root_device": 1,
        "root_inode": 2,
    }
    with ThreadPoolExecutor(max_workers=2) as pool:
        identities = list(
            pool.map(
                lambda registry: registry.register_or_verify("project", **values),
                registries,
            )
        )
    assert identities[0] == identities[1]
    for registry in registries:
        registry.close()


def test_registry_rejects_invalid_record_before_persist_and_replaces_by_revision(
    tmp_path,
):
    registry = SQLiteWorkspaceRegistry(tmp_path / "registry.sqlite3")
    with pytest.raises(ValueError):
        registry.register_or_verify(
            "bad id",
            backend="local-posix",
            resource_id="bad id",
            config_revision="config",
            root="/workspace",
            root_device=True,
            root_inode=2,
        )
    with pytest.raises(FileNotFoundError):
        registry.get("bad id")

    identity = registry.register_or_verify(
        "project",
        backend="local-posix",
        resource_id="project",
        config_revision="config",
        root="/workspace",
        root_device=1,
        root_inode=2,
    )
    first = registry.get_record("project")
    replacement = registry.replace(
        "project",
        expected_revision=first.revision,
        backend="local-posix",
        resource_id="project",
        config_revision="new-config",
        root="/new-workspace",
        root_device=1,
        root_inode=3,
    )
    assert replacement.revision == first.revision + 1
    assert replacement.identity.generation != identity.generation
    assert registry.get("project") == replacement.identity
    with pytest.raises(PermissionError, match="revision changed"):
        registry.replace(
            "project",
            expected_revision=first.revision,
            backend="local-posix",
            resource_id="project",
            config_revision="other",
            root="/other",
            root_device=1,
            root_inode=4,
        )
    registry.close()
