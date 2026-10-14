"""Read-only CLI discovery against absent, invalid, and real daemon state."""

import asyncio
import json
import os
import sys

import msgspec
import pytest

pytest.importorskip("litestar")
pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX local daemon")

from msgflux.runtime.service.local import connect_local_service, inspect_local_service
from msgflux.runtime.service.local.cli import main
from msgflux.runtime.service.local.files import read_record, write_record
from tests.test_local_service_integration import (
    FACTORY,
    daemon_workspace,
    _shutdown_record,
)


@pytest.mark.asyncio
async def test_missing_runtime_inspection_creates_no_files(tmp_path):
    runtime = tmp_path / "absent"
    result = await inspect_local_service(runtime_dir=runtime)
    assert result.status == "stopped"
    assert result.url is None
    assert not runtime.exists()


def test_cli_stopped_json_is_nonzero_and_does_not_start(tmp_path, capsys):
    runtime = tmp_path / "absent"
    with pytest.raises(SystemExit) as exited:
        main(["status", "--runtime-dir", str(runtime), "--json"])
    assert exited.value.code == 1
    output = capsys.readouterr()
    assert json.loads(output.out)["status"] == "stopped"
    assert output.err == ""
    assert not runtime.exists()


def test_administration_failure_reports_cause_without_traceback(tmp_path, capsys):
    runtime = tmp_path / "runtime"
    runtime.mkdir(mode=0o700)
    record = runtime / "daemon.json"
    record.write_text("invalid json")
    record.chmod(0o600)
    with pytest.raises(SystemExit) as exited:
        main(["status", "--runtime-dir", str(runtime)])
    assert exited.value.code == 1
    output = capsys.readouterr()
    assert "metadata is malformed" in output.err
    assert "Traceback" not in output.err
    assert record.read_text() == "invalid json"


@pytest.mark.asyncio
async def test_real_daemon_status_checks_identity_and_never_exposes_token(
    daemon_workspace,
):
    project, _b, _c, runtime = daemon_workspace
    client = await connect_local_service(FACTORY, runtime_dir=runtime, cwd=project)
    try:
        record = read_record(runtime)
        result = await inspect_local_service(runtime_dir=runtime)
        assert result.status == "running"
        assert result.url == client.base_url
        assert result.instance_id == record.instance_id
        assert record.token not in msgspec.json.encode(result).decode()

        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "msgflux.cli",
            "service",
            "status",
            "--runtime-dir",
            str(runtime),
            "--json",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), 10)
        assert process.returncode == 0 and not stderr
        assert json.loads(stdout)["url"] == client.base_url
        assert record.token.encode() not in stdout + stderr

        values = msgspec.to_builtins(record)
        values["instance_id"] = "different-instance"
        write_record(runtime, type(record)(**values))
        try:
            result = await inspect_local_service(runtime_dir=runtime)
            assert result.status == "unreachable"
            assert result.url is None
            assert read_record(runtime).instance_id == "different-instance"
        finally:
            write_record(runtime, record)
    finally:
        await client.aclose()
    await _shutdown_record(runtime)
    assert (await inspect_local_service(runtime_dir=runtime)).status == "stopped"
