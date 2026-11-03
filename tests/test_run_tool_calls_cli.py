"""Bounded call observation never needs a snapshot, stream or tool output."""

import asyncio
import json
from contextlib import aclosing
from types import SimpleNamespace
from unittest.mock import patch

import httpx2
import msgspec
import pytest

from msgflux.runtime.events import EventType, emit_event
from msgflux.runtime.service import AgentSession, RunToolCall, RunToolCalls
from msgflux.runtime.service.http.client import (
    AgentServiceClient,
    AgentServiceHTTPError,
)
from msgflux.runtime.service.local import cli, connect_local_service, stop_local_service
from tests.test_agent_service_http_integration import TOKEN, _agent, _response, _server
from tests.test_local_service_operations import _command


def test_plain_and_json_output_do_not_expand_preview(capsys):
    result = RunToolCalls(
        "thread",
        "run",
        "running",
        local_worker=True,
        tool_calls=(
            RunToolCall(
                "grep",
                "stamp",
                arguments_preview='{"path":"src/"}',
                arguments_truncated=True,
            ),
        ),
    )
    cli._show_run_result(
        SimpleNamespace(action="activity", tools=True, json=False), result
    )
    out = capsys.readouterr().out
    assert "grep" in out and '"path":"src/"' in out and "[truncated]" in out
    cli._show_run_result(
        SimpleNamespace(action="activity", tools=True, json=True), result
    )
    assert json.loads(capsys.readouterr().out)["tool_calls"][0]["tool_name"] == "grep"


@pytest.mark.parametrize(
    "args",
    [
        ["activity", "thread", "run", "--limit", "3"],
        ["activity", "thread", "run", "--tools", "--limit", "0"],
        ["activity", "thread", "run", "--tools", "--limit", "-1"],
        ["inspect", "thread", "run", "--tools"],
    ],
)
def test_bad_tool_observation_options_fail_before_discovery(args):
    with patch.object(cli, "_run_operation", side_effect=AssertionError("discovery")):
        with pytest.raises(SystemExit):
            cli.main(args, resource="runs")


@pytest.mark.asyncio
async def test_http_calls_default_limit_authentication_and_no_hydration():
    entered, release = asyncio.Event(), asyncio.Event()

    async def answer(**kwargs):
        for index in range(16):
            emit_event(
                EventType.TOOL_START,
                {
                    "tool_name": "grep",
                    "tool_call_id": f"call-{index}",
                    "arguments": {"path": "src", "pattern": f"symbol_{index}"},
                },
            )
            emit_event(
                EventType.TOOL_END,
                {"tool_name": "grep", "result": "PRIVATE-TOOL-OUTPUT"},
            )
        entered.set()
        await release.wait()
        return _response("done")

    agent = _agent(answer)
    try:
        async with _server(lambda _: AgentSession(agent)) as (service, url):
            async with aclosing(AgentServiceClient(url, token=TOKEN)) as client:
                thread = await client.open_thread("main")
                receipt = await client.prompt(
                    thread.thread_id, "PRIVATE-PROMPT", request_id="one"
                )
                await asyncio.wait_for(entered.wait(), 3)

                # The model emitter can run ahead of the service event consumer.
                async def observed():
                    while True:
                        calls = await client.run_tool_calls(
                            thread.thread_id, receipt.run_id
                        )
                        if (
                            len(calls.tool_calls) == 10
                            and calls.tool_calls[-1].tool_call_id == "call-15"
                        ):
                            return calls
                        await asyncio.sleep(0)

                calls = await asyncio.wait_for(observed(), 3)
                assert calls.tool_calls[0].tool_call_id == "call-6"
                with patch.object(
                    service, "acquire_session", side_effect=AssertionError("hydration")
                ):
                    calls = await client.run_tool_calls(
                        thread.thread_id, receipt.run_id, limit=3
                    )
                assert [call.tool_call_id for call in calls.tool_calls] == [
                    "call-13",
                    "call-14",
                    "call-15",
                ]
                encoded = msgspec.json.encode(calls)
                assert (
                    b"PRIVATE-TOOL-OUTPUT" not in encoded
                    and b"PRIVATE-PROMPT" not in encoded
                )
                async with httpx2.AsyncClient() as raw:
                    path = f"{url}/v1/threads/{thread.thread_id}/runs/{receipt.run_id}/tool-calls"
                    assert (await raw.get(path)).status_code == 401
                    for limit in ("0", "51", "bad"):
                        response = await raw.get(
                            path,
                            params={"limit": limit},
                            headers={"Authorization": f"Bearer {TOKEN}"},
                        )
                        assert response.status_code == 422
                release.set()
                await asyncio.wait_for(service.wait(thread.thread_id, "one"), 3)
                settled = await client.run_tool_calls(thread.thread_id, receipt.run_id)
                assert not settled.local_worker and not settled.tool_calls
    finally:
        release.set()
        await agent.aclose()


@pytest.mark.asyncio
async def test_older_service_404_has_upgrade_message_without_snapshot():
    paths = []

    def respond(request):
        paths.append(request.url.path)
        if request.url.path.endswith("/activity"):
            return httpx2.Response(
                200, json={"thread_id": "t", "run_id": "r", "status": "running"}
            )
        return httpx2.Response(404, json={"code": "not_found", "message": "missing"})

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as raw:
        client = AgentServiceClient("http://test", token="secret", client=raw)
        with pytest.raises(AgentServiceHTTPError, match="Restart or upgrade"):
            await client.run_tool_calls("t", "r")
    assert paths == ["/v1/threads/t/runs/r/tool-calls", "/v1/threads/t/runs/r/activity"]


@pytest.mark.asyncio
async def test_real_cli_tool_call_selection_and_final_discard(tmp_path):
    project, runtime = tmp_path / "project", tmp_path / "runtime"
    project.mkdir()
    (project / "fixture.py").write_text("""
import asyncio
from unittest.mock import AsyncMock, Mock
from msgflux.models.response import ModelResponse
from msgflux.nn import Agent
from msgflux.runtime.events import emit_event
from msgflux.runtime.service import AgentService, AgentSession, SQLiteServiceStore

def create(directory):
    model = Mock(model_type="chat_completion")
    async def answer(**kwargs):
        for i in range(12):
            emit_event("tool.start", {"tool_name":"grep", "tool_call_id":f"call-{i}", "arguments":{"path":"src", "pattern":f"symbol_{i}", "api_key":"PRIVATE-KEY"}})
            emit_event("tool.end", {"tool_name":"grep", "result":"PRIVATE-OUTPUT"})
        (directory / "ready").touch()
        while not (directory / "release").exists():
            await asyncio.sleep(0.01)
        result = ModelResponse()
        result.set_response_type("text_generation")
        result.add("done")
        result.reasoning = None
        return result
    agent = Agent(name="activity", model=model)
    agent.generator.aforward = AsyncMock(side_effect=answer)
    service = AgentService(store=SQLiteServiceStore(directory / "service.sqlite3"))
    service.register("main", lambda _: AgentSession(agent))
    return service
""")
    client = await connect_local_service(
        "fixture:create", runtime_dir=runtime, cwd=project
    )
    try:
        thread = await client.open_thread("main")
        receipt = await client.prompt(
            thread.thread_id, "private prompt", request_id="one"
        )

        async def ready():
            while not (runtime / "ready").exists():
                await asyncio.sleep(0.01)

        await asyncio.wait_for(ready(), 5)
        code, out, err = await _command(
            runtime, "runs", "activity", thread.thread_id, receipt.run_id, "--tools"
        )
        assert code == 0 and not err
        calls = json.loads(out)["tool_calls"]
        assert len(calls) == 10 and calls[0]["tool_call_id"] == "call-2"
        assert "PRIVATE-KEY" not in out and "PRIVATE-OUTPUT" not in out
        code, out, err = await _command(
            runtime,
            "runs",
            "activity",
            thread.thread_id,
            receipt.run_id,
            "--tools",
            "--limit",
            "2",
            as_json=False,
        )
        assert code == 0 and not err and "Recent tool calls: 2" in out
        assert "symbol_10" in out and "symbol_0" not in out
    finally:
        (runtime / "release").touch()
        await client.aclose()
        await stop_local_service(runtime_dir=runtime)
