from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest

from msgflux import cli
from msgflux.coding.prompt import PromptExecution
from msgflux.runtime.service.local import connect_local_service
from msgflux.runtime.service.local.files import read_record
from msgflux.runtime.service.http.client import AgentServiceClient


def test_cli_routes_administration_without_loading_a_prompt(monkeypatch):
    from msgflux.runtime.service.local import cli as service_cli

    calls = []
    monkeypatch.setattr(service_cli, "main", calls.append)
    cli.main(["init", "--model", "provider/model"])
    cli.main(["service", "restart"])
    assert calls == [["init", "--model", "provider/model"], ["restart"]]


def test_json_setup_error_has_one_result_and_no_traceback(tmp_path, capsys):
    with pytest.raises(SystemExit) as exited:
        cli.main(["-p", "hello", "--json", "--config", str(tmp_path / "missing.toml")])
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert exited.value.code == 1
    assert result["status"] == "failed" and result["error"]
    assert captured.err == ""
    assert result["run_id"] is None


@pytest.fixture
def coding_cli_project(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    (project / "fixture_agent.py").write_text("""
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, Mock
from msgflux.exceptions import TaskPauseRequestedError
from msgflux.nn import Agent
from msgflux.models.response import ModelResponse
from msgflux.runtime.service import AgentSession
from msgflux.runtime.events import emit_event

def create(thread):
    model = Mock()
    model.model_type = "chat_completion"
    agent = Agent(name=thread.agent_id, model=model, agent_dir=Path(__file__).parent / "state")
    async def respond(**kwargs):
        prompt = kwargs["messages"].to_chatml()[-1]["content"]
        if prompt == "error":
            raise RuntimeError("provider fixture failure: concrete cause")
        if prompt == "pause":
            raise TaskPauseRequestedError("Approval required by fixture")
        emit_event("commentary.delta", {"delta": "Inspecting\\nworkspace"})
        emit_event("tool.start", {"name": "fixture_read", "arguments": {"path": "hello.txt"}})
        emit_event("tool.end", {"name": "fixture_read", "output": "fixture contents"})
        if prompt == "stream-gated":
            while not (Path(__file__).parent / "release").exists():
                await asyncio.sleep(0.01)
        await asyncio.sleep(0.5 if prompt == "reconnect" else 0.02)
        result = ModelResponse()
        result.set_response_type("text_generation")
        result.add("FINAL: " + prompt)
        return result
    agent.generator.aforward = AsyncMock(side_effect=respond)
    return AgentSession(agent)
""")
    config = project / "config.toml"
    config.write_text(
        '[defaults]\nagent="worker"\n[agents.worker]\nfactory="fixture_agent:create"\n'
    )
    return project, config, tmp_path / "runtime"


async def _shutdown(runtime):
    record = read_record(runtime)
    if record is None:
        return
    client = AgentServiceClient(record.url, token=record.token)
    try:
        await client.shutdown(expected_instance_id=record.instance_id)
        async with asyncio.timeout(10):
            while read_record(runtime) is not None:
                await asyncio.sleep(0.02)
    finally:
        await client.aclose()


async def _command(project, config, runtime, *arguments):
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "msgflux.cli",
        *arguments,
        "--config",
        str(config),
        "--runtime-dir",
        str(runtime),
        "--cwd",
        str(project),
        cwd=Path(__file__).resolve().parents[1],
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), 20)
        return process.returncode, stdout.decode(), stderr.decode()
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()


@pytest.mark.asyncio
async def test_cli_final_json_resume_historical_result_and_failures(coding_cli_project):
    project, config, runtime = coding_cli_project
    client = None
    try:
        code, out, err = await _command(
            project, config, runtime, "-p", "alpha", "--json"
        )
        assert code == 0, err + out
        assert err == ""
        first = json.loads(out)
        assert first["output"] == "FINAL: alpha" and first["status"] == "completed"
        assert len(out.splitlines()) == 1
        client = await connect_local_service(
            config=config, runtime_dir=runtime, cwd=project
        )
        thread = next(
            t for t in await client.threads() if t.thread_id == first["thread_id"]
        )
        assert thread.interactive is False
        code, out, err = await _command(
            project, config, runtime, "-p", "beta", "--thread", thread.thread_id
        )
        assert code == 0 and out == "FINAL: beta\n", err
        assert f"thread_id={thread.thread_id}" in err
        # The endpoint identifies a run, even after a newer conversation turn.
        historical = await client.run_result(thread.thread_id, first["run_id"])
        assert historical.output == "FINAL: alpha"
        code, out, err = await _command(
            project, config, runtime, "-p", "error", "--json"
        )
        failure = json.loads(out)
        assert code == 1 and "concrete cause" in failure["error"]
        assert failure["thread_id"] and failure["run_id"] and failure["request_id"]
        assert failure["output"] is None
        code, out, err = await _command(
            project, config, runtime, "-p", "pause", "--json"
        )
        paused = json.loads(out)
        assert code == 1 and paused["status"] == "paused", err + out
        assert paused["output"] is None and "Approval required" in paused["error"]
    finally:
        if client is not None:
            await client.aclose()
        await _shutdown(runtime)


@pytest.mark.asyncio
async def test_print_watch_reconnect_does_not_admit_twice(
    coding_cli_project, monkeypatch
):
    project, config, runtime = coding_cli_project
    from contextlib import asynccontextmanager
    from msgflux.coding import prompt as prompt_module

    client = await connect_local_service(
        config=config, runtime_dir=runtime, cwd=project
    )
    original_watch = client.watch
    original_prompt = client.prompt
    admitted = []
    watches = []
    observed = []

    async def admit(*args, **kwargs):
        admitted.append(kwargs["request_id"])
        return await original_prompt(*args, **kwargs)

    @asynccontextmanager
    async def dropped(thread_id):
        async with original_watch(thread_id) as watcher:
            watches.append(thread_id)
            if len(watches) == 1:

                class BrokenWatch:
                    def __aiter__(self):
                        return self

                    async def __anext__(self):
                        import httpx2

                        raise httpx2.ReadError("lost SSE connection")

                broken = BrokenWatch()
                broken.snapshot = watcher.snapshot
                yield broken
            else:
                yield watcher

    async def connect(**kwargs):
        return client

    monkeypatch.setattr(client, "prompt", admit)
    monkeypatch.setattr(client, "watch", dropped)
    monkeypatch.setattr(prompt_module, "connect_local_service", connect)
    try:
        result = await asyncio.wait_for(
            PromptExecution().run(
                "reconnect",
                config=config,
                runtime_dir=runtime,
                cwd=project,
                observer=lambda kind, record: observed.append((kind, record)),
            ),
            10,
        )
        assert result.output == "FINAL: reconnect"
        assert len(admitted) == 1
        assert len(watches) == 2
        assert observed[0][0] == "snapshot"
        assert len([record for kind, record in observed if kind == "snapshot"]) == len(
            watches
        )
    finally:
        await client.aclose()
        await _shutdown(runtime)


@pytest.mark.asyncio
async def test_lost_prompt_response_recovers_one_admission(
    coding_cli_project, monkeypatch
):
    import httpx2
    from msgflux.coding import prompt as prompt_module

    project, config, runtime = coding_cli_project
    client = await connect_local_service(
        config=config, runtime_dir=runtime, cwd=project
    )
    original_prompt = client.prompt
    admissions = []

    async def admit_then_drop_response(*args, **kwargs):
        admissions.append(kwargs["request_id"])
        await original_prompt(*args, **kwargs)
        raise httpx2.ReadError("admission response was lost")

    async def connect(**kwargs):
        return client

    monkeypatch.setattr(client, "prompt", admit_then_drop_response)
    monkeypatch.setattr(prompt_module, "connect_local_service", connect)
    try:
        result = await asyncio.wait_for(
            PromptExecution().run(
                "reconnect", config=config, runtime_dir=runtime, cwd=project
            ),
            10,
        )
        assert result.status == "completed"
        assert result.output == "FINAL: reconnect"
        assert result.thread_id and result.run_id and result.request_id
        assert admissions == [result.request_id]
    finally:
        await client.aclose()
        await _shutdown(runtime)


@pytest.mark.asyncio
async def test_lost_prompt_before_admission_does_not_invent_or_retry_run(
    coding_cli_project, monkeypatch
):
    import httpx2
    from msgflux.coding import prompt as prompt_module
    from msgflux.runtime.service.http.client import AgentServiceHTTPError

    project, config, runtime = coding_cli_project
    client = await connect_local_service(
        config=config, runtime_dir=runtime, cwd=project
    )
    admissions = []
    execution = PromptExecution()

    async def fail_before_admission(*args, **kwargs):
        admissions.append(kwargs["request_id"])
        raise httpx2.ReadError("connection failed before request arrived")

    async def connect(**kwargs):
        return client

    monkeypatch.setattr(client, "prompt", fail_before_admission)
    monkeypatch.setattr(prompt_module, "connect_local_service", connect)
    try:
        with pytest.raises(AgentServiceHTTPError) as caught:
            await asyncio.wait_for(
                execution.run(
                    "reconnect", config=config, runtime_dir=runtime, cwd=project
                ),
                10,
            )
        assert caught.value.status_code == 404
        assert len(admissions) == 1
        assert execution.result.request_id == admissions[0]
        assert execution.result.run_id is None
    finally:
        await client.aclose()
        await _shutdown(runtime)


@pytest.mark.asyncio
async def test_watch_entry_failure_before_admission_does_not_lookup_receipt(
    monkeypatch,
):
    import httpx2
    from contextlib import asynccontextmanager
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from msgflux.coding import prompt as prompt_module

    @asynccontextmanager
    async def failed_watch(_thread_id):
        raise httpx2.ReadError("watch could not start")
        yield

    client = SimpleNamespace(
        open_thread=AsyncMock(
            return_value=SimpleNamespace(thread_id="thread", agent_id="worker")
        ),
        prompt=AsyncMock(),
        watch=failed_watch,
        receipt=AsyncMock(),
        aclose=AsyncMock(),
    )
    monkeypatch.setattr(
        prompt_module, "connect_local_service", AsyncMock(return_value=client)
    )
    execution = PromptExecution()
    with pytest.raises(httpx2.ReadError, match="watch could not start"):
        await execution.run("hello")
    client.prompt.assert_not_awaited()
    client.receipt.assert_not_awaited()
    client.aclose.assert_awaited_once()
    assert execution.result.run_id is None


@pytest.mark.asyncio
async def test_cancel_during_receipt_recovery_uses_failed_admission_request_id(
    coding_cli_project, monkeypatch
):
    import httpx2
    from msgflux.coding import prompt as prompt_module

    project, config, runtime = coding_cli_project
    client = await connect_local_service(
        config=config, runtime_dir=runtime, cwd=project
    )
    original_prompt = client.prompt
    original_receipt = client.receipt
    original_interrupt = client.interrupt
    admissions = []
    interrupts = []
    lookup_started = asyncio.Event()
    first_lookup = True

    async def admit_then_drop_response(*args, **kwargs):
        admissions.append(kwargs["request_id"])
        await original_prompt(*args, **kwargs)
        raise httpx2.ReadError("admission response was lost")

    async def block_first_receipt(thread_id, request_id, *, require_waitable=False):
        nonlocal first_lookup
        if first_lookup:
            first_lookup = False
            lookup_started.set()
            await asyncio.Event().wait()
        return await original_receipt(
            thread_id, request_id, require_waitable=require_waitable
        )

    async def interrupt(thread_id, run_id):
        interrupts.append((thread_id, run_id))
        return await original_interrupt(thread_id, run_id)

    async def connect(**kwargs):
        return client

    monkeypatch.setattr(client, "prompt", admit_then_drop_response)
    monkeypatch.setattr(client, "receipt", block_first_receipt)
    monkeypatch.setattr(client, "interrupt", interrupt)
    monkeypatch.setattr(prompt_module, "connect_local_service", connect)
    execution = PromptExecution()
    task = asyncio.create_task(
        execution.run(
            "stream-gated",
            config=config,
            runtime_dir=runtime,
            cwd=project,
            interrupt_on_cancel=True,
        )
    )
    try:
        await asyncio.wait_for(lookup_started.wait(), 10)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 10)
        assert admissions == [execution.result.request_id]
        assert execution.result.run_id is not None
        assert interrupts == [(execution.result.thread_id, execution.result.run_id)]
        assert execution.result.status == "interrupted"
    finally:
        (project / "release").touch(exist_ok=True)
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await client.aclose()
        await _shutdown(runtime)


@pytest.mark.asyncio
async def test_cli_stream_ndjson_events_and_final_result(coding_cli_project):
    project, config, runtime = coding_cli_project
    try:
        code, out, err = await _command(
            project, config, runtime, "-p", "stream", "--stream", "--json"
        )
        assert code == 0, err + out
        assert err == ""
        records = [json.loads(line) for line in out.splitlines()]
        assert records[0]["type"] == "snapshot"
        assert records[0]["snapshot"]["thread_id"]
        events = [record["event"] for record in records if record["type"] == "event"]
        assert any(event["type"] == "tool.start" for event in events)
        assert any(event["type"] == "tool.end" for event in events)
        assert any(
            event["type"] == "commentary.delta"
            and event["data"]["delta"] == "Inspecting\nworkspace"
            for event in events
        )
        assert records[-1]["type"] == "result"
        result = records[-1]["result"]
        assert result["output"] == "FINAL: stream"
        assert result["status"] == "completed"
        assert all(record["version"] == 1 for record in records)
        assert len([record for record in records if record["type"] == "result"]) == 1
        assert all(event["run_id"] == result["run_id"] for event in events)
    finally:
        await _shutdown(runtime)


@pytest.mark.asyncio
async def test_cli_plain_stream_labels_events_and_keeps_final_answer(
    coding_cli_project,
):
    project, config, runtime = coding_cli_project
    try:
        code, out, err = await _command(
            project, config, runtime, "-p", "plain", "--stream"
        )
        assert code == 0, err + out
        assert "[snapshot] thread_id=" in out
        assert "[tool.start" in out and "fixture_read" in out
        assert "[commentary.delta" in out
        assert out.endswith("FINAL: plain\n")
    finally:
        await _shutdown(runtime)


@pytest.mark.asyncio
@pytest.mark.parametrize("prompt,status", [("error", "failed"), ("pause", "paused")])
async def test_cli_stream_reports_failed_and_paused_results(
    coding_cli_project, prompt, status
):
    project, config, runtime = coding_cli_project
    try:
        code, out, err = await _command(
            project, config, runtime, "-p", prompt, "--stream", "--json"
        )
        records = [json.loads(line) for line in out.splitlines()]
        assert code == 1 and err == ""
        assert records[0]["type"] == "snapshot"
        assert records[-1]["type"] == "result"
        result = records[-1]["result"]
        assert result["status"] == status and result["error"]
        assert result["run_id"] and result["request_id"]
        assert result["output"] is None
    finally:
        await _shutdown(runtime)


def test_cli_stream_setup_error_is_a_result_envelope(tmp_path, capsys):
    with pytest.raises(SystemExit) as exited:
        cli.main(
            [
                "-p",
                "hello",
                "--stream",
                "--json",
                "--config",
                str(tmp_path / "missing.toml"),
            ]
        )
    captured = capsys.readouterr()
    record = json.loads(captured.out)
    assert exited.value.code == 1
    assert record["type"] == "result" and record["result"]["status"] == "failed"
    assert record["result"]["run_id"] is None and captured.err == ""


@pytest.mark.asyncio
async def test_cli_stream_flushes_before_generation_finishes(coding_cli_project):
    project, config, runtime = coding_cli_project
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "msgflux.cli",
        "-p",
        "stream-gated",
        "--stream",
        "--json",
        "--config",
        str(config),
        "--runtime-dir",
        str(runtime),
        "--cwd",
        str(project),
        cwd=Path(__file__).resolve().parents[1],
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        async with asyncio.timeout(10):
            snapshot = json.loads(await process.stdout.readline())
            assert snapshot["type"] == "snapshot"
            while True:
                record = json.loads(await process.stdout.readline())
                if (
                    record["type"] == "event"
                    and record["event"]["type"] == "tool.start"
                ):
                    break
            assert not (project / "release").exists()
            (project / "release").touch()
            remaining, error = await process.communicate()
        assert process.returncode == 0, error.decode()
        final = json.loads(remaining.decode().splitlines()[-1])
        assert final["result"]["output"] == "FINAL: stream-gated"
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()
        await _shutdown(runtime)


@pytest.mark.asyncio
async def test_stream_preserves_child_events_without_stopping_on_child_terminal():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from msgflux.runtime.service.records import EventRecord

    events = [
        EventRecord(
            "tool.start", "now", {"name": "read"}, "child", ("coding", "explorer")
        ),
        EventRecord("run.end", "now", {}, "child", ("coding", "explorer")),
        EventRecord("message.delta", "now", {"delta": "final"}, "root", ("coding",)),
        EventRecord("run.end", "now", {}, "root", ("coding",)),
    ]

    async def watcher():
        for event in events:
            yield event

    client = SimpleNamespace(
        receipt=AsyncMock(return_value=SimpleNamespace(status="running"))
    )
    execution = PromptExecution()
    execution.result.run_id = "root"
    observed = []
    await asyncio.wait_for(
        execution._observe(
            client, watcher(), lambda kind, record: observed.append((kind, record))
        ),
        1,
    )
    assert observed == [("event", event) for event in events]


@pytest.mark.parametrize("detach", [False, True])
def test_cli_stream_keyboard_interrupt_emits_one_result(monkeypatch, capsys, detach):
    def interrupt(coroutine):
        coroutine.close()
        raise KeyboardInterrupt

    monkeypatch.setattr(cli.asyncio, "run", interrupt)
    with pytest.raises(SystemExit) as exited:
        cli.main(
            ["-p", "hello", "--stream", "--json"]
            + (["--detach-on-interrupt"] if detach else [])
        )
    captured = capsys.readouterr()
    assert exited.value.code == 130 and captured.err == ""
    records = [json.loads(line) for line in captured.out.splitlines()]
    assert len(records) == 1
    assert records[0]["type"] == "result"
    assert records[0]["result"]["status"] == ("detached" if detach else "interrupted")


@pytest.mark.asyncio
@pytest.mark.parametrize("detach", [False, True])
async def test_cli_ctrl_c_interrupts_or_detaches_real_service(
    coding_cli_project, detach
):
    import signal

    project, config, runtime = coding_cli_project
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "msgflux.cli",
        "-p",
        "stream-gated",
        "--stream",
        "--json",
        *(["--detach-on-interrupt"] if detach else []),
        "--config",
        str(config),
        "--runtime-dir",
        str(runtime),
        "--cwd",
        str(project),
        cwd=Path(__file__).resolve().parents[1],
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    client = None
    try:
        async with asyncio.timeout(15):
            snapshot = json.loads(await process.stdout.readline())
            thread_id = snapshot["snapshot"]["thread_id"]
            while True:
                record = json.loads(await process.stdout.readline())
                if (
                    record["type"] == "event"
                    and record["event"]["type"] == "tool.start"
                ):
                    break
            process.send_signal(signal.SIGINT)
            remaining, error = await process.communicate()
        assert process.returncode == 130, error.decode() + remaining.decode()
        records = [json.loads(line) for line in remaining.decode().splitlines()]
        assert records[-1]["type"] == "result"
        result = records[-1]["result"]
        assert result["status"] == ("detached" if detach else "interrupted"), result
        assert result["run_id"] and result["request_id"]
        assert len([r for r in records if r["type"] == "result"]) == 1
        client = await connect_local_service(
            config=config, runtime_dir=runtime, cwd=project
        )
        receipt = await client.receipt(thread_id, result["request_id"])
        if detach:
            assert receipt.status == "running"
            (project / "release").touch()
            async with asyncio.timeout(10):
                while receipt.status == "running":
                    await asyncio.sleep(0.02)
                    receipt = await client.receipt(thread_id, result["request_id"])
            assert receipt.status == "completed"
            assert (
                await client.run_result(thread_id, result["run_id"])
            ).output == "FINAL: stream-gated"
        else:
            assert receipt.status == "interrupted"
            assert not (project / "release").exists()
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()
        if client is not None:
            await client.aclose()
        await _shutdown(runtime)


@pytest.mark.asyncio
async def test_cancellation_during_admission_recovers_receipt_and_interrupts_once(
    monkeypatch,
):
    from contextlib import asynccontextmanager
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from msgflux.coding import prompt as prompt_module

    started = asyncio.Event()
    release = asyncio.Event()

    async def admit(*args, **kwargs):
        started.set()
        await release.wait()
        return SimpleNamespace(run_id="run_admitted")

    @asynccontextmanager
    async def watch(thread_id):
        yield SimpleNamespace(snapshot=None)

    client = SimpleNamespace(
        open_thread=AsyncMock(
            return_value=SimpleNamespace(thread_id="thread", agent_id="coding")
        ),
        prompt=AsyncMock(side_effect=admit),
        watch=watch,
        interrupt=AsyncMock(return_value=True),
        receipt=AsyncMock(return_value=SimpleNamespace(status="interrupted")),
        run_result=AsyncMock(
            return_value=SimpleNamespace(
                status="interrupted", output=None, error="Service run interrupted"
            )
        ),
        aclose=AsyncMock(),
    )
    monkeypatch.setattr(
        prompt_module, "connect_local_service", AsyncMock(return_value=client)
    )
    execution = PromptExecution()
    task = asyncio.create_task(execution.run("hello", interrupt_on_cancel=True))
    await asyncio.wait_for(started.wait(), 1)
    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 1)
    client.prompt.assert_awaited_once()
    client.interrupt.assert_awaited_once_with("thread", "run_admitted")
    client.aclose.assert_awaited_once()
    assert execution.result.status == "interrupted"
    assert execution.result.run_id == "run_admitted"


@pytest.mark.asyncio
async def test_interrupt_failure_does_not_claim_the_run_stopped():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    import httpx2

    execution = PromptExecution()
    execution.result.thread_id = "thread"
    execution.result.run_id = "run"
    execution.result.request_id = "request"
    client = SimpleNamespace(
        interrupt=AsyncMock(side_effect=httpx2.ReadError("connection lost"))
    )
    await execution._on_cancel(client, None, interrupt=True)
    assert execution.result.status == "detached"
    assert "Could not confirm interruption" in execution.result.error
    assert "connection lost" in execution.result.error


@pytest.mark.asyncio
async def test_interrupt_racing_with_completion_preserves_completed_result():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    execution = PromptExecution()
    execution.result.thread_id = "thread"
    execution.result.run_id = "run"
    execution.result.request_id = "request"
    client = SimpleNamespace(
        interrupt=AsyncMock(return_value=False),
        receipt=AsyncMock(return_value=SimpleNamespace(status="completed")),
        run_result=AsyncMock(
            return_value=SimpleNamespace(
                status="completed", output="already done", error=None
            )
        ),
    )
    await execution._on_cancel(client, None, interrupt=True)
    assert execution.result.status == "completed"
    assert execution.result.output == "already done"
    assert execution.result.error is None


@pytest.mark.asyncio
async def test_interrupt_timeout_reports_requested_without_claiming_settlement(
    monkeypatch,
):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from msgflux.coding import prompt as prompt_module

    timeout = asyncio.timeout
    monkeypatch.setattr(prompt_module.asyncio, "timeout", lambda seconds: timeout(0.02))
    execution = PromptExecution()
    execution.result.thread_id = "thread"
    execution.result.run_id = "run"
    execution.result.request_id = "request"
    client = SimpleNamespace(
        interrupt=AsyncMock(return_value=True),
        receipt=AsyncMock(return_value=SimpleNamespace(status="running")),
    )
    await execution._on_cancel(client, None, interrupt=True)
    assert execution.result.status == "interrupt_requested"
    assert "has not settled" in execution.result.error


def test_ctrl_c_completion_race_preserves_final_json_result(monkeypatch, capsys):
    async def completed_before_interrupt(self, *args, **kwargs):
        self.result.status = "completed"
        self.result.output = "already done"
        raise KeyboardInterrupt

    monkeypatch.setattr(PromptExecution, "run", completed_before_interrupt)
    with pytest.raises(SystemExit) as exited:
        cli.main(["-p", "hello", "--json"])
    assert exited.value.code == 130
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "completed" and result["output"] == "already done"
    assert result["error"] is None


@pytest.mark.parametrize("as_json", [False, True])
@pytest.mark.asyncio
async def test_detached_cli_returns_ids_while_service_keeps_running(
    coding_cli_project, as_json
):
    project, config, runtime = coding_cli_project
    client = None
    try:
        code, out, err = await _command(
            project,
            config,
            runtime,
            "-p",
            "stream-gated",
            "--detach",
            *(["--json"] if as_json else []),
        )
        assert code == 0, out + err
        if as_json:
            result = json.loads(out)
            assert not err and result["output"] is None and result["error"] is None
        else:
            result = dict(field.split("=", 1) for field in err.split())
            assert out.startswith("status=")
            assert err.count("thread_id=") == 1
        for key in ("thread_id", "agent_id", "request_id", "run_id"):
            assert result[key]
        assert result["agent_id"] == "worker"
        client = await connect_local_service(
            config=config, runtime_dir=runtime, cwd=project
        )
        receipt = await client.receipt(result["thread_id"], result["request_id"])
        assert receipt.status in {"accepted", "running"}
        assert receipt.run_id == result["run_id"]
        (project / "release").touch()
        async with asyncio.timeout(10):
            while (
                await client.receipt(result["thread_id"], result["request_id"])
            ).status in {"accepted", "running"}:
                await asyncio.sleep(0.02)
        final = await client.run_result(result["thread_id"], result["run_id"])
        assert final.status == "completed" and final.output == "FINAL: stream-gated"
        code, out, err = await _command(
            project, config, runtime, "-p", "continued", "--thread", result["thread_id"]
        )
        assert code == 0 and out == "FINAL: continued\n"
        assert f"thread_id={result['thread_id']}" in err
        assert len(await client.runs(result["thread_id"])) == 2
    finally:
        if client is not None:
            await client.aclose()
        await _shutdown(runtime)


@pytest.mark.asyncio
async def test_plain_prompt_flushes_generated_ids_before_completion(coding_cli_project):
    project, config, runtime = coding_cli_project
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "msgflux.cli",
        "-p",
        "stream-gated",
        "--config",
        str(config),
        "--runtime-dir",
        str(runtime),
        "--cwd",
        str(project),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    client = None
    try:
        async with asyncio.timeout(15):
            line = (await process.stderr.readline()).decode()
        result = dict(field.split("=", 1) for field in line.split())
        assert all(
            result[key] for key in ("thread_id", "agent_id", "request_id", "run_id")
        )
        assert process.returncode is None
        client = await connect_local_service(
            config=config, runtime_dir=runtime, cwd=project
        )
        assert (
            await client.receipt(result["thread_id"], result["request_id"])
        ).status in {"accepted", "running"}
        (project / "release").touch()
        out, err = await asyncio.wait_for(process.communicate(), 15)
        assert process.returncode == 0 and out.decode() == "FINAL: stream-gated\n"
        assert not err
    finally:
        if process.returncode is None:
            process.kill()
            await process.communicate()
        if client is not None:
            await client.aclose()
        await _shutdown(runtime)


@pytest.mark.asyncio
async def test_detached_admission_loss_recovers_without_watch_or_resubmission(
    coding_cli_project, monkeypatch
):
    import httpx2
    from msgflux.coding import prompt as prompt_module

    project, config, runtime = coding_cli_project
    client = await connect_local_service(
        config=config, runtime_dir=runtime, cwd=project
    )
    original_prompt = client.prompt
    original_receipt = client.receipt
    submitted = []
    lookups = []
    admitted = []

    async def submit(*args, **kwargs):
        submitted.append(kwargs["request_id"])
        await original_prompt(*args, **kwargs)
        raise httpx2.ReadError("admission response lost")

    async def lookup(*args, **kwargs):
        lookups.append(args)
        if len(lookups) == 1:
            raise httpx2.ReadError("receipt lookup connection lost")
        return await original_receipt(*args, **kwargs)

    def watch(*args, **kwargs):
        raise AssertionError("Detached admission must not open a watcher")

    async def connect(**kwargs):
        return client

    monkeypatch.setattr(client, "prompt", submit)
    monkeypatch.setattr(client, "receipt", lookup)
    monkeypatch.setattr(client, "watch", watch)
    monkeypatch.setattr(prompt_module, "connect_local_service", connect)
    other = None
    try:
        result = await asyncio.wait_for(
            PromptExecution().run(
                "stream-gated", detach=True, on_admitted=admitted.append
            ),
            10,
        )
        assert result.status in {"accepted", "running"}
        assert submitted == [result.request_id]
        assert len(lookups) == 2 and lookups[0] == lookups[1]
        assert admitted == [result]
        other = await connect_local_service(
            config=config, runtime_dir=runtime, cwd=project
        )
        (project / "release").touch()
        async with asyncio.timeout(10):
            while (await other.receipt(result.thread_id, result.request_id)).status in {
                "accepted",
                "running",
            }:
                await asyncio.sleep(0.02)
        assert (await other.run_result(result.thread_id, result.run_id)).output == (
            "FINAL: stream-gated"
        )
        assert len(await other.runs(result.thread_id)) == 1
        snapshot = await other.snapshot(result.thread_id)
        assert sum(message.get("role") == "user" for message in snapshot.messages) == 1
    finally:
        await client.aclose()
        if other is not None:
            await other.aclose()
        await _shutdown(runtime)


def test_detach_and_stream_are_mutually_exclusive(capsys):
    with pytest.raises(SystemExit) as exited:
        cli.main(["-p", "hello", "--detach", "--stream"])
    assert exited.value.code == 2
    captured = capsys.readouterr()
    assert "not allowed with argument" in captured.err
    assert not captured.out


@pytest.mark.asyncio
async def test_detach_rejects_observer_before_connecting(monkeypatch):
    from unittest.mock import AsyncMock
    from msgflux.coding import prompt as prompt_module

    connect = AsyncMock()
    monkeypatch.setattr(prompt_module, "connect_local_service", connect)
    with pytest.raises(ValueError, match="cannot observe"):
        await PromptExecution().run("hello", detach=True, observer=lambda *_: None)
    connect.assert_not_called()


@pytest.mark.parametrize(
    "status, exit_code",
    [
        ("accepted", 0),
        ("running", 0),
        ("completed", 0),
        ("failed", 1),
        ("paused", 1),
        ("interrupted", 1),
    ],
)
def test_detached_json_reports_receipt_status_and_ids(
    monkeypatch, capsys, status, exit_code
):
    from msgflux.coding.prompt import PromptResult

    async def admit(self, prompt, **kwargs):
        self.result = PromptResult(
            status=status,
            thread_id="thread",
            agent_id="worker",
            request_id="request",
            run_id="run",
        )
        kwargs["on_admitted"](self.result)
        return self.result

    monkeypatch.setattr(PromptExecution, "run", admit)
    if exit_code:
        with pytest.raises(SystemExit) as exited:
            cli.main(["-p", "hello", "--detach", "--json"])
        assert exited.value.code == exit_code
    else:
        cli.main(["-p", "hello", "--detach", "--json"])
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert not captured.err and len(captured.out.splitlines()) == 1
    assert result["status"] == status
    assert result["thread_id"] == "thread" and result["request_id"] == "request"
    assert result["run_id"] == "run" and result["output"] is None


def test_plain_admission_failure_reports_known_ids(monkeypatch, capsys):
    async def failed(self, prompt, **kwargs):
        self.result.thread_id = "thread"
        self.result.agent_id = "worker"
        self.result.request_id = "request"
        raise OSError("Cannot confirm admission")

    monkeypatch.setattr(PromptExecution, "run", failed)
    with pytest.raises(SystemExit) as exited:
        cli.main(["-p", "hello", "--detach"])
    assert exited.value.code == 1
    captured = capsys.readouterr()
    assert not captured.out
    assert "thread_id=thread" in captured.err
    assert "request_id=request" in captured.err
    assert "Cannot confirm admission" in captured.err
