"""Deliver CLI steering through the existing service AgentInbox boundary."""

import json
from html import unescape
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("litestar")
pytest.importorskip("uvicorn")
pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX local daemon")

from msgflux.runtime.service.local import connect_local_service
from msgflux.runtime.service.local.cli import _steer_thread, main
from tests.test_local_service_integration import (
    FACTORY,
    _until,
    _wait_completed,
    daemon_workspace,
)
from tests.test_local_service_operations import _command


@pytest.mark.parametrize(
    "args, message",
    [
        (["steer", "t"], "non-empty --message"),
        (["steer", "t", "-m", "  "], "non-empty --message"),
        (["steer", "-m", "hello"], "exactly one THREAD_ID"),
        (["steer", "t", "r", "-m", "hello"], "exactly one THREAD_ID"),
        (["watch", "t", "-m", "hello"], "only accepted with threads steer"),
    ],
)
def test_steer_arguments_are_validated(args, message):
    with pytest.raises(SystemExit, match=message):
        main(args, resource="threads")


@pytest.mark.asyncio
async def test_steer_absent_daemon_does_not_create_files(tmp_path):
    runtime = tmp_path / "absent"
    code, out, err = await _command(runtime, "threads", "steer", "t", "-m", "hello")
    assert code == 1 and not out and "stopped" in err
    assert not runtime.exists()


@pytest.mark.asyncio
async def test_steer_rejects_idle_and_unknown_threads(daemon_workspace):
    project, _b, _c, runtime = daemon_workspace
    client = await connect_local_service(FACTORY, runtime_dir=runtime, cwd=project)
    try:
        thread = await client.open_thread("main")
        for thread_id in (thread.thread_id, "unknown"):
            code, out, err = await _command(
                runtime, "threads", "steer", thread_id, "-m", "hello"
            )
            assert code == 1 and not out and "No active run" in err
            assert "Traceback" not in err
        assert not (runtime / "model_calls").exists()
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_steer_refuses_ambiguous_target_without_sending():
    client = SimpleNamespace(
        active_runs=AsyncMock(
            return_value=(
                SimpleNamespace(thread_id="thread", run_id="one"),
                SimpleNamespace(thread_id="thread", run_id="two"),
            )
        ),
        steer=AsyncMock(),
    )
    args = SimpleNamespace(thread_id="thread", message="hello", json=True)
    with pytest.raises(RuntimeError, match="Multiple active runs"):
        await _steer_thread(client, args)
    client.steer.assert_not_called()


@pytest.mark.parametrize("as_json", [False, True])
@pytest.mark.asyncio
async def test_cli_steer_reaches_same_run_after_tool_without_new_prompt(
    daemon_workspace, as_json
):
    project, _b, _c, runtime = daemon_workspace
    factory = project / "daemon_fixture.py"
    source = (
        factory.read_text()
        .replace(
            "        agent = Agent(",
            "        async def gated():\n"
            "            (runtime_dir / 'tool-entered').touch()\n"
            "            while not (runtime_dir / 'release').exists():\n"
            "                await asyncio.sleep(0.02)\n"
            "            return 'tool complete'\n"
            "        agent = Agent(",
        )
        .replace("tools=[BashTool()],", "tools=[BashTool(), gated],")
    )
    source = source.replace(
        '            if prompt.startswith("workspace:"):',
        '            if prompt == "needs-steer":\n'
        "                calls = ToolCallAggregator()\n"
        "                calls.process(0, 'gated-call', 'gated', '{}')\n"
        "                response = ModelResponse()\n"
        "                response.set_response_type('tool_call')\n"
        "                response.add(calls)\n"
        "                return response\n"
        '            if prompt.startswith("workspace:"):',
    )
    factory.write_text(source)
    client = await connect_local_service(FACTORY, runtime_dir=runtime, cwd=project)
    try:
        thread = await client.open_thread("main")
        receipt = await client.prompt(thread.thread_id, "needs-steer", request_id="one")
        await _until((runtime / "tool-entered").exists)
        message = 'Priorize os testes de integração: "leitura e escrita".'
        code, out, err = await _command(
            runtime,
            "threads",
            "steer",
            thread.thread_id,
            "-m",
            message,
            as_json=as_json,
        )
        assert code == 0 and not err, out + err
        if as_json:
            queued = json.loads(out)
            assert queued["thread_id"] == thread.thread_id
            assert queued["run_id"] == receipt.run_id
            assert queued["notification_id"] and queued["status"] == "queued"
        else:
            assert "Message queued" in out and "notification_id=" in out
            assert thread.thread_id in out and receipt.run_id in out
        assert (await client.receipt(thread.thread_id, "one")).status == "running"
        (runtime / "release").touch()
        assert (
            await _wait_completed(client, thread.thread_id, "one")
        ).status == "completed"
        result = await client.run_result(thread.thread_id, receipt.run_id)
        assert message in unescape(result.output)
        assert len(await client.runs(thread.thread_id)) == 1
        assert (runtime / "model_calls").read_text().splitlines() == ["call", "call"]
        code, out, err = await _command(
            runtime, "threads", "steer", thread.thread_id, "-m", "too late"
        )
        assert code == 1 and not out and "No active run" in err
    finally:
        (runtime / "release").touch()
        await client.aclose()
