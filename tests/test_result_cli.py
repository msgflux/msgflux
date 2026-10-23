"""Retrieve final answers through real CLI consumers without changing execution."""

import asyncio
import json
import os
import signal
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("litestar")
pytest.importorskip("uvicorn")
pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX local daemon")

from msgflux.runtime.service import AdmissionReceipt, RunResult, SQLiteServiceStore
from msgflux.runtime.service.local import connect_local_service
from msgflux.runtime.service.local.results import show_result
from tests.test_local_service_integration import (
    FACTORY,
    _until,
    _wait_completed,
    daemon_workspace,
)
from tests.test_local_service_operations import _command


@pytest.mark.parametrize(
    "resource, ids", [("threads", ("thread",)), ("runs", ("t", "r"))]
)
@pytest.mark.asyncio
async def test_result_absent_service_does_not_start_daemon(tmp_path, resource, ids):
    runtime = tmp_path / "absent"
    code, out, err = await _command(runtime, resource, "result", *ids)
    assert code == 1 and not out and "stopped" in err
    assert "Traceback" not in err and not runtime.exists()


@pytest.mark.asyncio
async def test_result_plain_json_history_empty_and_interrupted(daemon_workspace):
    project, _b, _c, runtime = daemon_workspace
    client = await connect_local_service(FACTORY, runtime_dir=runtime, cwd=project)
    try:
        thread = await client.open_thread("main")
        for thread_id in (thread.thread_id, "unknown"):
            code, out, err = await _command(runtime, "threads", "result", thread_id)
            assert code == 1 and not out and err and "Traceback" not in err
        first = await client.prompt(thread.thread_id, "hello", request_id="first")
        await _wait_completed(client, thread.thread_id, "first")
        for resource, ids in (
            ("threads", (thread.thread_id,)),
            ("runs", (thread.thread_id, first.run_id)),
        ):
            code, out, err = await _command(
                runtime, resource, "result", *ids, as_json=False
            )
            assert code == 0 and out == "daemon response: hello\n" and not err
            code, out, err = await _command(runtime, resource, "result", *ids)
            assert code == 0 and not err
            assert json.loads(out) == {
                "thread_id": thread.thread_id,
                "run_id": first.run_id,
                "status": "completed",
                "output": "daemon response: hello",
                "error": None,
            }
        second = await client.prompt(thread.thread_id, "block", request_id="second")
        await _until(
            lambda: len((runtime / "model_calls").read_text().splitlines()) == 2
        )
        code, out, err = await _command(runtime, "threads", "result", thread.thread_id)
        assert code == 1 and not err
        assert json.loads(out)["status"] == "running"
        assert json.loads(out)["output"] is None
        await client.interrupt(thread.thread_id, second.run_id)
        await _wait_completed(client, thread.thread_id, "second")
        code, out, err = await _command(runtime, "threads", "result", thread.thread_id)
        assert code == 1 and not err
        assert json.loads(out)["status"] == "interrupted"
        code, out, err = await _command(
            runtime, "threads", "result", thread.thread_id, as_json=False
        )
        assert code == 1 and not out and "interrupted" in err
        code, out, err = await _command(
            runtime, "runs", "result", thread.thread_id, first.run_id
        )
        assert (
            code == 0
            and not err
            and json.loads(out)["output"] == ("daemon response: hello")
        )
        assert (runtime / "model_calls").read_text().splitlines() == ["call", "call"]
    finally:
        (runtime / "release").touch()
        await client.aclose()


@pytest.mark.asyncio
async def test_result_dot_segment_id_stays_on_its_thread_route(daemon_workspace):
    project, _b, _c, runtime = daemon_workspace
    client = await connect_local_service(FACTORY, runtime_dir=runtime, cwd=project)
    try:
        thread = await client.open_thread("main", thread_id="..")
        receipt = await client.prompt(thread.thread_id, "hello", request_id="dot")
        assert (
            await _wait_completed(client, thread.thread_id, "dot")
        ).status == "completed"
        for resource, ids in (
            ("threads", (thread.thread_id,)),
            ("runs", (thread.thread_id, receipt.run_id)),
        ):
            code, out, err = await _command(runtime, resource, "result", *ids)
            assert code == 0 and not err
            value = json.loads(out)
            assert value["thread_id"] == thread.thread_id
            assert value["run_id"] == receipt.run_id
            assert value["output"] == "daemon response: hello"
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_result_wait_ctrl_c_detaches_and_second_wait_returns_answer(
    daemon_workspace,
):
    project, _b, _c, runtime = daemon_workspace
    factory = project / "daemon_fixture.py"
    factory.write_text(
        factory.read_text().replace(
            "    def create_agent(thread):",
            "    original_wait_receipt = service._wait_receipt\n"
            "    def observed_wait(thread_id, request_id):\n"
            "        (runtime_dir / 'wait-attached').touch()\n"
            "        return original_wait_receipt(thread_id, request_id)\n"
            "    service._wait_receipt = observed_wait\n\n"
            "    def create_agent(thread):",
        )
    )
    client = await connect_local_service(FACTORY, runtime_dir=runtime, cwd=project)
    process = None
    try:
        thread = await client.open_thread("main")
        receipt = await client.prompt(thread.thread_id, "block", request_id="first")
        await _until((runtime / "model_calls").exists)
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "msgflux.cli",
            "threads",
            "result",
            thread.thread_id,
            "--wait",
            "--json",
            "--runtime-dir",
            str(runtime),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await _until((runtime / "wait-attached").exists)
        # The attached wait must remain blocked while the admitted model is gated.
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.shield(process.wait()), 0.5)
        process.send_signal(signal.SIGINT)
        out, err = await asyncio.wait_for(process.communicate(), 10)
        assert process.returncode == 130 and not out and b"Traceback" not in err
        assert (await client.receipt(thread.thread_id, "first")).status == "running"
        waiting = asyncio.create_task(
            _command(
                runtime, "runs", "result", thread.thread_id, receipt.run_id, "--wait"
            )
        )
        (runtime / "release").touch()
        code, out, err = await waiting
        assert (
            code == 0
            and not err
            and json.loads(out)["output"] == ("daemon response: block")
        )
        assert (runtime / "model_calls").read_text().splitlines() == ["call"]
    finally:
        (runtime / "release").touch()
        if process is not None and process.returncode is None:
            process.kill()
            await process.communicate()
        await client.aclose()


@pytest.mark.asyncio
async def test_wait_orphan_reports_recovery_required_without_replaying(
    daemon_workspace,
):
    project, _b, _c, runtime = daemon_workspace
    client = await connect_local_service(FACTORY, runtime_dir=runtime, cwd=project)
    try:
        thread = await client.open_thread("main")
        journal = SQLiteServiceStore(runtime / "service.sqlite3")
        try:
            admission = journal.admit(thread.thread_id, "old", "block", "main")
            assert journal.claim(admission, "former-worker")
        finally:
            journal.close()
        code, out, err = await _command(
            runtime, "threads", "result", thread.thread_id, "--wait"
        )
        assert code == 1 and not err
        value = json.loads(out)
        assert value["status"] == "running" and value["output"] is None
        assert value["run_id"] == admission.receipt.run_id
        assert "does not own" in value["error"]
        assert not (runtime / "model_calls").exists()
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_wait_pins_selected_run_even_if_new_admission_arrives(capsys):
    selected = RunResult("thread", "first", "running")
    client = SimpleNamespace(
        thread_result=AsyncMock(return_value=selected),
        inspect_run=AsyncMock(
            return_value=SimpleNamespace(
                receipt=AdmissionReceipt("thread", "request", "first", "running")
            )
        ),
        receipt=AsyncMock(
            return_value=AdmissionReceipt("thread", "request", "first", "completed")
        ),
        run_result=AsyncMock(
            return_value=RunResult(
                "thread", "first", "completed", output="first answer"
            )
        ),
    )

    async def newer_admission(*_args, **_kwargs):
        client.thread_result.return_value = RunResult("thread", "newer", "running")
        return AdmissionReceipt("thread", "request", "first", "completed")

    client.receipt.side_effect = newer_admission
    args = SimpleNamespace(resource="threads", thread_id="thread", json=True, wait=True)
    await show_result(client, args)
    value = json.loads(capsys.readouterr().out)
    assert value["run_id"] == "first" and value["output"] == "first answer"
    client.thread_result.assert_awaited_once_with("thread")
    client.run_result.assert_awaited_once_with("thread", "first")
    client.receipt.assert_awaited_once_with("thread", "request", require_waitable=True)


@pytest.mark.parametrize(
    "status", ["accepted", "running", "paused", "interrupted", "failed"]
)
@pytest.mark.parametrize("as_json", [False, True])
@pytest.mark.asyncio
async def test_unfinished_result_never_waits_or_returns_previous_answer(
    status, as_json, capsys
):
    client = SimpleNamespace(
        thread_result=AsyncMock(
            return_value=RunResult(
                "thread",
                "latest",
                status,
                error="provider unavailable" if status == "failed" else None,
            )
        ),
        inspect_run=AsyncMock(),
        receipt=AsyncMock(),
    )
    args = SimpleNamespace(
        resource="threads", thread_id="thread", json=as_json, wait=False
    )
    with pytest.raises(SystemExit) as caught:
        await show_result(client, args)
    assert caught.value.code == 1
    captured = capsys.readouterr()
    if as_json:
        value = json.loads(captured.out)
        assert value["run_id"] == "latest" and value["status"] == status
        assert value["output"] is None and not captured.err
    else:
        assert not captured.out and status in captured.err
        if status == "failed":
            assert "provider unavailable" in captured.err
    client.inspect_run.assert_not_awaited()
    client.receipt.assert_not_awaited()


@pytest.mark.parametrize(
    "resource, args, message",
    [
        ("threads", ["result"], "requires exactly one THREAD_ID"),
        ("threads", ["result", "t", "r"], "requires exactly one THREAD_ID"),
        ("runs", ["result"], "requires THREAD_ID and RUN_ID"),
        ("runs", ["result", "t"], "requires THREAD_ID and RUN_ID"),
        ("threads", ["watch", "t", "--wait"], "--wait is only accepted"),
        ("runs", ["inspect", "t", "r", "--wait"], "--wait is only accepted"),
        ("service", ["status", "--wait"], "--wait is only accepted"),
        ("threads", ["result", "t", "-m", "hello"], "only accepted with threads steer"),
    ],
)
def test_result_arguments_are_validated(resource, args, message):
    from msgflux.runtime.service.local.cli import main

    with pytest.raises(SystemExit, match=message):
        main(args, resource=resource)
