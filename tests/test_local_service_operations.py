"""Operation commands against a real daemon, without implicit startup."""

import asyncio
import json
import os
import sys

import msgspec
import pytest

pytest.importorskip("litestar")
pytest.importorskip("uvicorn")
pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX local daemon")

from msgflux.runtime.service import ServiceRecoveryRequiredError, SQLiteServiceStore
from msgflux.runtime.service.local import (
    connect_local_service,
    inspect_local_service,
    stop_local_service,
)
from msgflux.runtime.service.local.cli import main
from msgflux.runtime.service.local.files import read_record, write_record
from tests.test_local_service_integration import (
    FACTORY,
    daemon_workspace,
    _until,
    _wait_completed,
)


async def _command(runtime, *args, as_json=True):
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "msgflux.cli",
        *args,
        "--runtime-dir",
        str(runtime),
        *(["--json"] if as_json else []),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), 15)
        return process.returncode, stdout.decode(), stderr.decode()
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()


@pytest.mark.asyncio
async def test_absent_service_commands_do_not_create_files(tmp_path):
    runtime = tmp_path / "absent"
    for action in ("inspect", "stop", "down"):
        code, out, err = await _command(runtime, "service", action)
        assert code == 0 and not err
        value = json.loads(out)
        assert (value["service"] if action == "inspect" else value)[
            "status"
        ] == "stopped"
        assert not runtime.exists()
    for action in ("inspect", "resume"):
        code, out, err = await _command(runtime, "runs", action, "thread", "run")
        assert code == 1 and not out
        assert "stopped" in err and "Traceback" not in err
        assert not runtime.exists()
    for action in ("interrupt", "stop"):
        code, out, err = await _command(runtime, "threads", action, "thread")
        assert code == 1 and not out and "stopped" in err
        assert not runtime.exists()
    code, out, err = await _command(runtime, "runs", "ls", "--active")
    assert code == 1 and not out and "stopped" in err
    assert not runtime.exists()


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan"), True])
@pytest.mark.asyncio
async def test_stop_rejects_invalid_timeout_without_creating_files(tmp_path, timeout):
    runtime = tmp_path / "absent"
    with pytest.raises(ValueError, match="finite positive"):
        await stop_local_service(runtime_dir=runtime, stop_timeout=timeout)
    assert not runtime.exists()


@pytest.mark.parametrize(
    "args, resource, message",
    [
        (["inspect", "thread"], "runs", "requires THREAD_ID and RUN_ID"),
        (["resume"], "runs", "requires THREAD_ID and RUN_ID"),
        (["interrupt"], "threads", "requires exactly one THREAD_ID"),
        (["stop", "t", "r"], "threads", "requires exactly one THREAD_ID"),
        (
            ["stop", "t", "--stop-timeout", "1"],
            "threads",
            "only accepted with service stop",
        ),
        (["ls"], "runs", "requires --active"),
        (["resume", "t", "r", "--active"], "runs", "only accepted with runs ls"),
        (["inspect", "thread", "run"], "service", "require msgflux runs"),
        (["inspect", "--factory", "m:f"], "service", "does not accept"),
        (
            ["inspect", "--stop-timeout", "1"],
            "service",
            "only accepted with service stop",
        ),
    ],
)
def test_operation_arguments_are_validated(args, resource, message):
    with pytest.raises(SystemExit, match=message):
        main(args, resource=resource)


@pytest.mark.asyncio
async def test_real_daemon_inspection_and_idempotent_stop(daemon_workspace):
    project, _b, _c, runtime = daemon_workspace
    client = await connect_local_service(FACTORY, runtime_dir=runtime, cwd=project)
    try:
        record = read_record(runtime)
        thread = await client.open_thread("main", cwd=project)
        receipt = await client.prompt(thread.thread_id, "hello", request_id="hello")
        assert (
            await _wait_completed(client, thread.thread_id, "hello")
        ).status == "completed"
        code, out, err = await _command(runtime, "service", "inspect")
        assert code == 0 and not err
        details = json.loads(out)
        assert details["service"]["instance_id"] == record.instance_id
        assert details["service"]["url"] == client.base_url
        assert details["factory"] == FACTORY and details["cwd"] == str(project)
        assert record.token not in out
        code, out, err = await _command(runtime, "service", "inspect", as_json=False)
        assert code == 0 and not err and f"Address: {client.base_url}" in out
        assert record.token not in out
        code, out, err = await _command(
            runtime, "runs", "inspect", thread.thread_id, "unknown-run"
        )
        assert code == 1 and not out and err and "Traceback" not in err

        code, out, err = await _command(
            runtime, "runs", "inspect", thread.thread_id, receipt.run_id
        )
        assert code == 0 and not err
        inspection = json.loads(out)
        assert inspection["receipt"]["status"] == "completed"
        assert inspection["run_id"] == receipt.run_id
        assert inspection["checkpoint_status"] == "completed"
        assert "output" not in inspection

        code, out, err = await _command(
            runtime, "runs", "resume", thread.thread_id, receipt.run_id
        )
        assert code == 0 and not err
        assert json.loads(out)["status"] == "completed"
        assert (runtime / "model_calls").read_text().splitlines() == ["call"]
        assert (runtime / "boots").read_text().splitlines() == ["boot"]

        code, out, err = await _command(runtime, "service", "stop")
        assert code == 0 and not err and json.loads(out)["status"] == "stopped"
        assert read_record(runtime) is None
        assert (await inspect_local_service(runtime_dir=runtime)).status == "stopped"
        code, out, err = await _command(runtime, "service", "down")
        assert code == 0 and not err and json.loads(out)["status"] == "stopped"
        # A stale record with no lifetime owner can safely be removed.
        write_record(runtime, record)
        assert (await stop_local_service(runtime_dir=runtime)).status == "stopped"
        assert read_record(runtime) is None
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_stop_refuses_unverified_live_instance(daemon_workspace):
    project, _b, _c, runtime = daemon_workspace
    client = await connect_local_service(FACTORY, runtime_dir=runtime, cwd=project)
    record = read_record(runtime)
    values = msgspec.to_builtins(record)
    values["instance_id"] = "wrong-instance"
    try:
        write_record(runtime, type(record)(**values))
        with pytest.raises(ServiceRecoveryRequiredError, match="Cannot verify"):
            await stop_local_service(runtime_dir=runtime)
        assert read_record(runtime).instance_id == "wrong-instance"
        assert (await client.health()).instance_id == record.instance_id
    finally:
        write_record(runtime, record)
        await client.aclose()


@pytest.mark.asyncio
async def test_stop_interrupts_and_drains_run_before_returning(daemon_workspace):
    project, _b, _c, runtime = daemon_workspace
    client = await connect_local_service(FACTORY, runtime_dir=runtime, cwd=project)
    try:
        thread = await client.open_thread("main")
        receipt = await client.prompt(thread.thread_id, "block", request_id="blocked")
        await _until((runtime / "model_calls").exists)
        await stop_local_service(runtime_dir=runtime)
        assert read_record(runtime) is None
        # Reconnect explicitly to verify the persisted interruption, not replay it.
        replacement = await connect_local_service(
            FACTORY, runtime_dir=runtime, cwd=project
        )
        try:
            assert (
                await replacement.receipt(thread.thread_id, "blocked")
            ).status == "interrupted"
            code, out, err = await _command(
                runtime, "runs", "resume", thread.thread_id, receipt.run_id
            )
            assert code == 0 and not err
            assert json.loads(out)["status"] == "interrupted"
            assert (runtime / "model_calls").read_text().splitlines() == ["call"]
        finally:
            await replacement.aclose()
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_cli_resume_preserves_paused_run_and_requires_approval(daemon_workspace):
    project, _b, _c, runtime = daemon_workspace
    factory = project / "daemon_fixture.py"
    source = factory.read_text()
    source = source.replace(
        "from msgflux.runtime import AgentWorkspace",
        "from msgflux.runtime import AgentWorkspace, AgentApprovals, InMemoryApprovalStore",
    )
    source = source.replace(
        "        agent = Agent(",
        "        def change(value: str):\n"
        "            (runtime_dir / 'changed').write_text(value)\n"
        "            return value\n"
        "        agent = Agent(",
    )
    source = source.replace(
        "tools=[BashTool()],",
        "tools=[BashTool(), change],\n"
        "            approvals=AgentApprovals(InMemoryApprovalStore(), "
        "{'change': 'v1'}, 'approval-test'),",
    )
    source = source.replace(
        '            if prompt.startswith("workspace:"):',
        '            if prompt == "needs-approval":\n'
        "                calls = ToolCallAggregator()\n"
        "                calls.process(0, 'change-1', 'change', "
        '\'{"value":"approved"}\')\n'
        "                response = ModelResponse()\n"
        "                response.set_response_type('tool_call')\n"
        "                response.add(calls)\n"
        "                return response\n"
        '            if prompt.startswith("workspace:"):',
    )
    source = source.replace(
        "return AgentSession(agent, on_close=checkpoints.close)",
        "return AgentSession(agent, on_close=checkpoints.close, "
        "scope_factory=lambda scope: scope.with_overrides(principal='operator'), approval_reviewer='operator')",
    )
    factory.write_text(source)
    client = await connect_local_service(FACTORY, runtime_dir=runtime, cwd=project)
    try:
        thread = await client.open_thread("main")
        receipt = await client.prompt(
            thread.thread_id, "needs-approval", request_id="approval"
        )
        assert (
            await _wait_completed(client, thread.thread_id, "approval")
        ).status == "paused"
        code, out, err = await _command(
            runtime, "runs", "resume", thread.thread_id, receipt.run_id
        )
        assert code == 0 and not err and json.loads(out)["run_id"] == receipt.run_id
        assert (
            await _wait_completed(client, thread.thread_id, "approval")
        ).status == "paused"
        assert not (runtime / "changed").exists()
        reviews = await client.approval_reviews(thread.thread_id, receipt.run_id)
        assert len(reviews) == 1
        await client.decide_approval(
            thread.thread_id,
            receipt.run_id,
            reviews[0].request_id,
            approved=True,
            expected_revision=reviews[0].revision,
        )
        code, out, err = await _command(
            runtime, "runs", "resume", thread.thread_id, receipt.run_id
        )
        assert code == 0 and not err and json.loads(out)["run_id"] == receipt.run_id
        assert (
            await _wait_completed(client, thread.thread_id, "approval")
        ).status == "completed"
        assert (runtime / "changed").read_text() == "approved"
        assert (runtime / "boots").read_text().splitlines() == ["boot"]
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_stop_timeout_leaves_worker_cleanup_running(daemon_workspace):
    from tests.test_local_service_restart import _delay_factory_cleanup

    project, _b, _c, runtime = daemon_workspace
    _delay_factory_cleanup(project)
    client = await connect_local_service(FACTORY, runtime_dir=runtime, cwd=project)
    try:
        thread = await client.open_thread("main")
        await client.prompt(thread.thread_id, "block", request_id="blocked")
        await _until((runtime / "model_calls").exists)
        code, out, err = await _command(
            runtime, "service", "stop", "--stop-timeout", "0.4"
        )
        assert code == 1 and not out
        assert "deadline exceeded" in err and "Traceback" not in err
        assert read_record(runtime) is not None
        assert (runtime / "boots").read_text().splitlines() == ["boot"]
        (runtime / "release").touch()
        await _until(lambda: read_record(runtime) is None)
        assert (await stop_local_service(runtime_dir=runtime)).status == "stopped"
    finally:
        (runtime / "release").touch()
        await client.aclose()


@pytest.mark.asyncio
async def test_thread_interrupt_targets_only_live_run_and_keeps_service_available(
    daemon_workspace,
):
    project, _b, _c, runtime = daemon_workspace
    client = await connect_local_service(FACTORY, runtime_dir=runtime, cwd=project)
    try:
        record = read_record(runtime)
        assert await client.active_runs() == ()
        assert not (runtime / "model_calls").exists()
        first = await client.open_thread("main", thread_id="first")
        second = await client.open_thread("main", thread_id="second")
        idle = await client.open_thread("main", thread_id="idle")
        orphan = await client.open_thread("main", thread_id="orphan")
        journal = SQLiteServiceStore(runtime / "service.sqlite3")
        try:
            stale = journal.admit(orphan.thread_id, "old", "block", "main")
            assert journal.claim(stale, "former-owner")
        finally:
            journal.close()
        assert (await client.receipt(orphan.thread_id, "old")).status == "running"
        a = await client.prompt(first.thread_id, "block", request_id="a")
        b = await client.prompt(second.thread_id, "block", request_id="b")
        await _until(
            lambda: (
                (runtime / "model_calls").exists()
                and len((runtime / "model_calls").read_text().splitlines()) == 2
            )
        )
        code, out, err = await _command(runtime, "runs", "ls", "--active")
        assert code == 0 and not err
        runs = json.loads(out)["runs"]
        assert {run["run_id"] for run in runs} == {a.run_id, b.run_id}
        assert all(run["status"] == "running" for run in runs)
        assert record.token not in out
        code, out, err = await _command(
            runtime, "runs", "ls", first.thread_id, "--active"
        )
        assert code == 0 and not err
        assert [run["run_id"] for run in json.loads(out)["runs"]] == [a.run_id]
        code, out, err = await _command(
            runtime, "threads", "interrupt", first.thread_id
        )
        assert code == 0 and not err
        assert json.loads(out) == {
            "thread_id": first.thread_id,
            "run_id": a.run_id,
            "status": "interrupt_requested",
        }
        assert (
            await _wait_completed(client, first.thread_id, "a")
        ).status == "interrupted"
        assert (await client.receipt(second.thread_id, "b")).status == "running"
        assert [run.run_id for run in await client.active_runs()] == [b.run_id]
        code, out, err = await _command(
            runtime, "threads", "stop", second.thread_id, as_json=False
        )
        assert code == 0 and not err and "Interruption requested" in out
        assert (
            await _wait_completed(client, second.thread_id, "b")
        ).status == "interrupted"
        for thread_id in (first.thread_id, idle.thread_id, orphan.thread_id):
            code, out, err = await _command(runtime, "threads", "stop", thread_id)
            assert code == 0 and not err
            assert json.loads(out)["status"] == "no_active_run"
        code, out, err = await _command(runtime, "threads", "interrupt", "unknown")
        assert code == 1 and not out and "Unknown thread" in err
        code, out, err = await _command(
            runtime, "runs", "ls", "--active", as_json=False
        )
        assert code == 0 and not err and "No active runs" in out
        # Persisted interrupted checkpoints remain inspectable, but are not live runs.
        assert (
            await client.inspect_run(first.thread_id, a.run_id)
        ).checkpoint_status == "interrupted"
        assert (await client.health()).instance_id == record.instance_id
        await client.prompt(first.thread_id, "hello again", request_id="again")
        assert (
            await _wait_completed(client, first.thread_id, "again")
        ).status == "completed"
        assert await client.active_runs() == ()
        assert (runtime / "boots").read_text().splitlines() == ["boot"]
    finally:
        await client.aclose()
