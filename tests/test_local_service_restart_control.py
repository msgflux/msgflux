"""Restart coordination and backward-compatible local service CLI controls."""

from pathlib import Path

import pytest

from msgflux.runtime.service import ServiceConflictError, ServiceRecoveryRequiredError
from msgflux.runtime.service.local import cli, discovery
from msgflux.runtime.service.local.files import (
    prepare_runtime_dir,
    read_record,
    write_record,
)
from msgflux.runtime.service.local.records import LocalServiceRecord
from msgflux.runtime.service.http.records import ShutdownResponse


def _record(**changes):
    values = {
        "instance_id": "old-instance",
        "factory": "sample.backend:create",
        "cwd": str(Path.cwd()),
        "url": "http://127.0.0.1:8234",
        "pid": 123,
        "token": "private-token",
    }
    values.update(changes)
    return LocalServiceRecord(**values)


@pytest.mark.asyncio
async def test_restart_uses_stored_configuration_and_returns_new_client(
    tmp_path, monkeypatch
):
    runtime = prepare_runtime_dir(tmp_path / "runtime")
    record = _record(cwd=str(tmp_path))
    write_record(runtime, record)
    events = []

    class Client:
        async def shutdown(self, *, expected_instance_id):
            events.append(("shutdown", expected_instance_id))
            return ShutdownResponse(instance_id=expected_instance_id)

        async def aclose(self):
            events.append(("close",))

    async def probe(observed):
        assert observed == record
        return Client()

    async def stopped(runtime_dir, deadline):
        events.append(("stopped", runtime_dir))

    async def start(runtime_dir, factory, cwd, deadline):
        events.append(("start", factory, cwd))
        return object()

    monkeypatch.setattr(discovery, "_probe", probe)
    monkeypatch.setattr(discovery, "_daemon_is_stopped", lambda _: False)
    monkeypatch.setattr(discovery, "_wait_for_daemon_stop", stopped)
    monkeypatch.setattr(discovery, "_start_and_wait", start)

    result = await discovery.restart_local_service(runtime_dir=runtime)
    assert events == [
        ("shutdown", "old-instance"),
        ("close",),
        ("stopped", runtime),
        ("start", "sample.backend:create", tmp_path.resolve()),
    ]
    assert result is not None
    assert read_record(runtime) is None


@pytest.mark.asyncio
async def test_restart_rejects_factory_mismatch_while_owner_is_alive(
    tmp_path, monkeypatch
):
    runtime = prepare_runtime_dir(tmp_path / "runtime")
    write_record(runtime, _record())

    class Client:
        async def aclose(self):
            pass

        async def shutdown(self, **kwargs):
            raise AssertionError("mismatched configuration must not be stopped")

    async def probe(_):
        return Client()

    monkeypatch.setattr(discovery, "_probe", probe)
    monkeypatch.setattr(discovery, "_daemon_is_stopped", lambda _: False)
    with pytest.raises(ServiceConflictError, match="different factory"):
        await discovery.restart_local_service(
            "other.backend:create", runtime_dir=runtime
        )


@pytest.mark.asyncio
async def test_restart_requires_explicit_factory_without_stored_record(tmp_path):
    with pytest.raises(ServiceRecoveryRequiredError, match="pass factory explicitly"):
        await discovery.restart_local_service(runtime_dir=tmp_path / "runtime")


@pytest.mark.asyncio
async def test_expired_restart_deadline_does_not_spawn_child(tmp_path, monkeypatch):
    def no_spawn(*args, **kwargs):
        raise AssertionError("expired restart must not spawn a daemon")

    monkeypatch.setattr(discovery, "_start_child", no_spawn)
    with pytest.raises(TimeoutError, match="before starting"):
        await discovery._start_and_wait(
            tmp_path, "sample.backend:create", tmp_path, deadline=0
        )


def test_cli_parser_preserves_legacy_serve_and_adds_restart():
    legacy = cli._parser().parse_args(
        ["--factory", "sample.backend:create", "--runtime-dir", "runtime-dir"]
    )
    assert legacy.action == "serve"
    assert legacy.factory == "sample.backend:create"
    restarted = cli._parser().parse_args(
        ["restart", "--runtime-dir", "runtime-dir", "--restart-timeout", "12"]
    )
    assert restarted.action == "restart"
    assert restarted.factory is None
    assert restarted.restart_timeout == 12


def test_cli_requires_factory_for_foreground_serve():
    with pytest.raises(SystemExit, match="serve requires --factory"):
        cli.main(["serve"])
