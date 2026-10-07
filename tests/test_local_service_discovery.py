"""Security and ownership contracts for local daemon discovery files."""

import asyncio
import stat
import subprocess
import sys
from pathlib import Path

import msgspec
import pytest

from msgflux.runtime.service.local.files import (
    RECORD_NAME,
    prepare_runtime_dir,
    process_lock,
    read_record,
    remove_record,
    write_record,
)
from msgflux.runtime.service.local.records import LocalServiceRecord
from msgflux.runtime.service.local.discovery import connect_local_service
from msgflux.runtime.service import ServiceConflictError, ServiceRecoveryRequiredError


def _record(**changes):
    values = {
        "instance_id": "instance-123",
        "factory": "myapp.agent:build",
        "cwd": str(Path.cwd()),
        "url": "http://127.0.0.1:8341",
        "pid": 123,
        "token": "top-secret-token",
    }
    values.update(changes)
    return LocalServiceRecord(**values)


def test_runtime_directory_record_permissions_and_redacted_repr(tmp_path):
    runtime = prepare_runtime_dir(tmp_path / "runtime")
    assert stat.S_IMODE(runtime.stat().st_mode) == 0o700

    record = _record()
    write_record(runtime, record)
    metadata = runtime / RECORD_NAME
    assert stat.S_IMODE(metadata.stat().st_mode) == 0o600
    assert read_record(runtime) == record
    assert "top-secret-token" not in repr(record)

    remove_record(runtime, "different-instance")
    assert read_record(runtime) == record
    remove_record(runtime, "instance-123")
    assert read_record(runtime) is None


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:8000",
        "http://127.0.0.2:8000",
        "https://127.0.0.1:8000",
        "http://127.0.0.1",
        "http://user:pass@127.0.0.1:8000",
        "http://127.0.0.1:99999",
        "http://127.0.0.1:8000/prefix",
    ],
)
def test_metadata_rejects_non_loopback_or_invalid_urls(url):
    with pytest.raises(ValueError):
        _record(url=url)


def test_record_decoder_rejects_unknown_fields_and_wrong_version():
    data = msgspec.to_builtins(_record())
    data["unexpected"] = True
    with pytest.raises(msgspec.DecodeError):
        msgspec.json.decode(msgspec.json.encode(data), type=LocalServiceRecord)
    with pytest.raises(msgspec.DecodeError):
        msgspec.json.decode(
            b'{"instance_id":"i","factory":"m:f","cwd":"/tmp",'
            b'"url":"http://127.0.0.1:8000","pid":1,"token":"x",'
            b'"version":2}',
            type=LocalServiceRecord,
        )


def test_metadata_symlink_and_insecure_permissions_are_rejected(tmp_path):
    runtime = prepare_runtime_dir(tmp_path / "runtime")
    external = tmp_path / "external.json"
    external.write_bytes(msgspec.json.encode(_record()))
    external.chmod(0o600)
    (runtime / RECORD_NAME).symlink_to(external)
    with pytest.raises((ValueError, OSError)):
        read_record(runtime)

    (runtime / RECORD_NAME).unlink()
    write_record(runtime, _record())
    (runtime / RECORD_NAME).chmod(0o644)
    with pytest.raises(PermissionError):
        read_record(runtime)


def test_runtime_directory_symlink_is_rejected(tmp_path):
    real = prepare_runtime_dir(tmp_path / "real")
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError, match="not a symlink"):
        prepare_runtime_dir(alias)


def test_process_lock_nonblocking_and_kernel_release(tmp_path):
    runtime = prepare_runtime_dir(tmp_path / "runtime")
    path = runtime / "startup.lock"
    with process_lock(path) as held:
        assert held is not None
        with process_lock(path) as contended:
            assert contended is None

    with process_lock(path) as released:
        assert released is not None
        child = subprocess.run(  # noqa: S603
            [
                sys.executable,
                "-c",
                "import fcntl,os,sys; fd=os.open(sys.argv[1],os.O_RDWR); "
                "fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)",
                str(path),
            ],
            check=False,
            capture_output=True,
        )
        assert child.returncode != 0
    with process_lock(path) as unlocked:
        assert unlocked is not None


@pytest.mark.asyncio
async def test_connect_reuses_only_matching_authenticated_instance(
    tmp_path, monkeypatch
):
    seen = []

    async def health_server(reader, writer):
        request = await reader.readline()
        headers = {}
        while line := await reader.readline():
            if line == b"\r\n":
                break
            name, value = line.decode().split(":", 1)
            headers[name.lower()] = value.strip()
        seen.append((request.decode().strip(), headers.get("authorization")))
        body = b'{"instance_id":"instance-123","version":1}'
        writer.write(
            b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
            + f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode()
            + body
        )
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_server(health_server, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    runtime = prepare_runtime_dir(tmp_path / "runtime")
    write_record(runtime, _record(url=f"http://127.0.0.1:{port}"))

    def no_spawn(*args, **kwargs):
        raise AssertionError("a healthy matching daemon must be reused")

    monkeypatch.setattr(
        "msgflux.runtime.service.local.discovery._start_child", no_spawn
    )
    client = await connect_local_service(
        "myapp.agent:build", runtime_dir=runtime, cwd=Path.cwd()
    )
    try:
        assert (await client.health()).instance_id == "instance-123"
    finally:
        await client.aclose()

    assert seen and all(auth == "Bearer top-secret-token" for _, auth in seen)

    # A healthy daemon is reused across frontend working directories. Its
    # recorded cwd remains the daemon bootstrap directory, not a reuse key.
    other_cwd = Path(__file__).resolve().parent
    assert other_cwd.is_dir()
    reused = await connect_local_service(
        "myapp.agent:build", runtime_dir=runtime, cwd=other_cwd
    )
    try:
        assert (await reused.health()).instance_id == "instance-123"
    finally:
        await reused.aclose()

    with pytest.raises(ServiceConflictError, match="different factory"):
        await connect_local_service(
            "myapp.other:build", runtime_dir=runtime, cwd=other_cwd
        )
    with pytest.raises(NotADirectoryError, match="existing directory"):
        await connect_local_service(
            "myapp.agent:build", runtime_dir=runtime, cwd=tmp_path / "missing"
        )
    server.close()
    await server.wait_closed()


@pytest.mark.asyncio
async def test_connect_does_not_spawn_when_unhealthy_daemon_owns_lock(
    tmp_path, monkeypatch
):
    runtime = prepare_runtime_dir(tmp_path / "runtime")
    write_record(runtime, _record(url="http://127.0.0.1:1"))

    def no_spawn(*args, **kwargs):
        raise AssertionError("an unhealthy daemon owner must not be replaced")

    monkeypatch.setattr(
        "msgflux.runtime.service.local.discovery._start_child", no_spawn
    )
    with process_lock(runtime / "daemon.lock") as lock:
        assert lock is not None
        with pytest.raises(
            ServiceRecoveryRequiredError, match="owns its lifetime lock"
        ):
            await connect_local_service(
                "myapp.agent:build", runtime_dir=runtime, cwd=Path.cwd()
            )


@pytest.mark.asyncio
async def test_connect_replaces_stale_metadata_for_different_configuration(
    tmp_path, monkeypatch
):
    runtime = prepare_runtime_dir(tmp_path / "runtime")
    write_record(
        runtime, _record(factory="oldapp.agent:build", url="http://127.0.0.1:1")
    )

    def observe_spawn(*args, **kwargs):
        raise RuntimeError("new daemon spawn reached")

    monkeypatch.setattr(
        "msgflux.runtime.service.local.discovery._start_child", observe_spawn
    )
    with pytest.raises(RuntimeError, match="new daemon spawn reached"):
        await connect_local_service(
            "myapp.agent:build", runtime_dir=runtime, cwd=Path.cwd()
        )
    assert read_record(runtime) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan"), True])
async def test_connect_rejects_invalid_startup_timeout(tmp_path, timeout):
    with pytest.raises(ValueError, match="finite positive"):
        await connect_local_service(
            "myapp.agent:build",
            runtime_dir=tmp_path / "runtime",
            cwd=Path.cwd(),
            startup_timeout=timeout,
        )


@pytest.mark.asyncio
async def test_connect_requires_existing_working_directory(tmp_path):
    with pytest.raises(NotADirectoryError, match="existing directory"):
        await connect_local_service(
            "myapp.agent:build",
            runtime_dir=tmp_path / "runtime",
            cwd=tmp_path / "missing",
        )


def test_failed_atomic_metadata_replace_preserves_old_record_and_discards_staging(
    tmp_path, monkeypatch
):
    from msgflux.runtime.service.local import files

    directory = files.prepare_runtime_dir(tmp_path / "runtime")
    old = LocalServiceRecord(
        instance_id="old",
        factory="module:factory",
        cwd=str(tmp_path),
        url="http://127.0.0.1:1234",
        pid=1,
        token="private-token",
    )
    files.write_record(directory, old)
    replacement = msgspec.structs.replace(old, instance_id="new")

    def fail_replace(*_args):
        raise OSError("replace failed")

    monkeypatch.setattr(files.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        files.write_record(directory, replacement)
    assert files.read_record(directory) == old
    assert list(directory.glob(".daemon-*")) == []
