"""CLI observation through a real daemon owned by a separate process."""

import asyncio
import json
import os
import signal
import sys
from contextlib import asynccontextmanager

import pytest

pytest.importorskip("litestar")
pytest.importorskip("uvicorn")
pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX local daemon")

from msgflux.runtime.service.local import connect_local_service
from msgflux.runtime.service.local.cli import main
from msgflux.runtime.service.local.files import read_record
from tests.test_local_service_integration import (
    FACTORY,
    _wait_completed,
    daemon_workspace,
)
from tests.test_local_service_operations import _command


@asynccontextmanager
async def _watch(runtime, thread_id, *, as_json):
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "msgflux.cli",
        "threads",
        "watch",
        thread_id,
        "--runtime-dir",
        str(runtime),
        *(["--json"] if as_json else []),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        yield process
    finally:
        if process.returncode is None:
            process.kill()
        await process.communicate()


async def _read_until(process, text):
    async with asyncio.timeout(15):
        while True:
            line = (await process.stdout.readline()).decode()
            assert line, f"Observer exited before {text}: {await process.stderr.read()}"
            if text in line:
                return line


@pytest.mark.asyncio
async def test_watch_absent_service_never_starts_daemon(tmp_path):
    runtime = tmp_path / "absent"
    code, out, err = await _command(runtime, "threads", "watch", "thread")
    assert code == 1 and not out and "stopped" in err
    assert "Traceback" not in err
    assert not runtime.exists()


@pytest.mark.parametrize(
    "args, message",
    [
        (["watch"], "requires exactly one THREAD_ID"),
        (["watch", "t", "r"], "requires exactly one THREAD_ID"),
        (["watch", "t", "--active"], "only accepted with runs ls"),
        (["watch", "t", "--config", "config.toml"], "does not accept"),
    ],
)
def test_watch_arguments_are_validated(args, message):
    with pytest.raises(SystemExit, match=message):
        main(args, resource="threads")


@pytest.mark.asyncio
async def test_watch_unknown_thread_reports_error_without_admission(daemon_workspace):
    project, _b, _c, runtime = daemon_workspace
    client = await connect_local_service(FACTORY, runtime_dir=runtime, cwd=project)
    try:
        code, out, err = await _command(runtime, "threads", "watch", "unknown")
        assert code == 1 and not out and err and "Traceback" not in err
        assert not (runtime / "model_calls").exists()
        assert await client.threads() == ()
    finally:
        await client.aclose()


@pytest.mark.parametrize("as_json", [False, True])
@pytest.mark.asyncio
async def test_watch_follows_runs_and_ctrl_c_only_disconnects(
    daemon_workspace, as_json
):
    project, _b, _c, runtime = daemon_workspace
    client = await connect_local_service(FACTORY, runtime_dir=runtime, cwd=project)
    try:
        thread = await client.open_thread("main")
        record = read_record(runtime)
        async with _watch(runtime, thread.thread_id, as_json=as_json) as observer:
            line = await _read_until(observer, "snapshot")
            assert thread.thread_id in line
            if as_json:
                snapshot = json.loads(line)
                assert snapshot["type"] == "snapshot" and snapshot["version"] == 1
                assert snapshot["snapshot"]["active_runs"] == []
            first = await client.prompt(thread.thread_id, "hello", request_id="one")
            line = await _read_until(observer, "run.end")
            if as_json:
                event = json.loads(line)
                assert event["type"] == "event"
                assert event["event"]["run_id"] == first.run_id
            assert observer.returncode is None
            assert (await _wait_completed(client, thread.thread_id, "one")).status == (
                "completed"
            )
            second = await client.prompt(thread.thread_id, "block", request_id="two")
            await _read_until(observer, "run.start")
            observer.send_signal(signal.SIGINT)
            output, errors = await asyncio.wait_for(observer.communicate(), 10)
            assert observer.returncode == 130, errors.decode()
            assert record.token not in output.decode()
        assert (await client.receipt(thread.thread_id, "two")).status == "running"
        assert (await client.health()).instance_id == record.instance_id
        (runtime / "release").touch()
        assert (await _wait_completed(client, thread.thread_id, "two")).status == (
            "completed"
        )
        async with _watch(runtime, thread.thread_id, as_json=True) as reattached:
            line = await _read_until(reattached, "snapshot")
            snapshot = json.loads(line)["snapshot"]
            assert snapshot["active_runs"] == []
            assert "daemon response: block" in str(snapshot["messages"])
            reattached.send_signal(signal.SIGINT)
            await asyncio.wait_for(reattached.communicate(), 10)
            assert reattached.returncode == 130
        assert (runtime / "model_calls").read_text().splitlines() == ["call", "call"]
        assert len(await client.runs(thread.thread_id)) == 2
        assert (await client.run_result(thread.thread_id, second.run_id)).output == (
            "daemon response: block"
        )
    finally:
        await client.aclose()
