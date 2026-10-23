"""Security and ownership contracts for local daemon discovery files."""

import asyncio
import stat
import subprocess
import sys
from contextlib import suppress
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


def test_record_removed_during_shutdown_before_open_is_absent(tmp_path, monkeypatch):
    from msgflux.runtime.service.local import files

    runtime = prepare_runtime_dir(tmp_path / "runtime")
    write_record(runtime, _record())
    original_open = files.os.open
    record_path = runtime / RECORD_NAME

    def remove_before_open(path, flags, *args, **kwargs):
        if Path(path) == record_path:
            record_path.unlink()
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(files.os, "open", remove_before_open)
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
async def test_invalid_timeout_does_not_initialize_default_config(
    tmp_path, monkeypatch
):
    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", lambda: home)
    with pytest.raises(ValueError, match="finite positive"):
        await connect_local_service(startup_timeout=0)
    assert not home.exists()


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


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["healthy", "wrong-instance", "error", "cancel"])
async def test_health_probe_closes_its_client_and_returns_only_metadata(
    monkeypatch, outcome
):
    from unittest.mock import AsyncMock
    from msgflux.runtime.service.local import discovery
    from msgflux.runtime.service.http.records import HealthRecord

    record = _record()
    client = AsyncMock()
    client.health.return_value = HealthRecord(
        instance_id=record.instance_id if outcome == "healthy" else "other-instance"
    )
    if outcome == "error":
        client.health.side_effect = RuntimeError("health failed")
    elif outcome == "cancel":
        client.health.side_effect = asyncio.CancelledError()
    created = []

    def create_client(url, **kwargs):
        assert url == record.url
        assert kwargs["token"] == record.token
        created.append(client)
        return client

    monkeypatch.setattr(discovery, "AgentServiceClient", create_client)
    if outcome == "cancel":
        with pytest.raises(asyncio.CancelledError):
            await discovery._probe(record)
    else:
        result = await discovery._probe(record)
        assert result == (record if outcome == "healthy" else None)
    client.aclose.assert_awaited_once()
    assert created == [client]


@pytest.mark.asyncio
@pytest.mark.parametrize("crash", [False, True])
async def test_concurrent_connectors_share_daemon_and_recover_its_metadata(
    tmp_path, monkeypatch, crash
):
    from msgflux.runtime.service.local import discovery
    from msgflux.runtime.service.http.client import AgentServiceClient

    runtime = tmp_path / "runtime"
    factory = "tests.test_local_service_runner:local_factory"
    original_start = discovery._start_child
    processes = []
    clients = []

    def start(*args, **kwargs):
        process = original_start(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(discovery, "_start_child", start)
    try:
        async with asyncio.timeout(30):
            results = await asyncio.gather(
                *(
                    connect_local_service(factory, runtime_dir=runtime, cwd=Path.cwd())
                    for _ in range(8)
                ),
                return_exceptions=True,
            )
            clients.extend(
                item for item in results if isinstance(item, AgentServiceClient)
            )
            assert len(clients) == 8, results
            record = read_record(runtime)
            assert record is not None
            assert len(processes) == 1
            assert len({id(client) for client in clients}) == 8
            assert {client.base_url for client in clients} == {record.url}
            health = await asyncio.gather(*(client.health() for client in clients))
            assert {item.instance_id for item in health} == {record.instance_id}
            await asyncio.gather(*(client.aclose() for client in clients))
            clients.clear()

            # Closing all frontends leaves a reusable, independently owned daemon.
            reused = await connect_local_service(factory, runtime_dir=runtime)
            clients.append(reused)
            assert (await reused.health()).instance_id == record.instance_id
            assert len(processes) == 1
            if crash:
                processes[0].kill()
                while processes[0].poll() is None:
                    await asyncio.sleep(0.01)
                assert read_record(runtime) == record
                replacement = await connect_local_service(factory, runtime_dir=runtime)
                clients.append(replacement)
                new_record = read_record(runtime)
                assert new_record is not None
                assert new_record.instance_id != record.instance_id
                assert replacement.base_url == new_record.url
                assert (
                    await replacement.health()
                ).instance_id == new_record.instance_id
                assert len(processes) == 2
            active = clients[-1]
            await active.shutdown(
                expected_instance_id=(await active.health()).instance_id
            )
            while read_record(runtime) is not None or not discovery._daemon_is_stopped(
                runtime
            ):
                await asyncio.sleep(0.01)
            assert not (runtime / RECORD_NAME).exists()
    finally:
        await asyncio.gather(*(client.aclose() for client in clients))
        record = read_record(runtime) if runtime.exists() else None
        if record is not None:
            cleanup_client = AgentServiceClient(record.url, token=record.token)
            try:
                with suppress(Exception):
                    await cleanup_client.shutdown(
                        expected_instance_id=record.instance_id
                    )
            finally:
                await cleanup_client.aclose()
        # Test-owned subprocesses are always reaped, including failed assertions.
        for process in processes:
            if process.poll() is None:
                await discovery._stop_owned_child(process)
