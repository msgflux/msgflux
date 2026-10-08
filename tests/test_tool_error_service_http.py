"""Tool errors remain useful and bounded across AgentService HTTP/SSE."""

import asyncio
import socket
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, Mock

import msgflux as mf
import pytest

pytest.importorskip("litestar")
uvicorn = pytest.importorskip("uvicorn")

from msgflux.models.response import ModelResponse
from msgflux.models.tool_call_agg import ToolCallAggregator
from msgflux.nn import Agent
from msgflux.nn.extensions import ToolOutputOffloadExtension
from msgflux.runtime import AgentWorkspace, ToolResultRef, ToolResultStore
from msgflux.runtime.service import AgentService, AgentSession, SQLiteServiceStore
from msgflux.runtime.service.http import (
    AgentServiceClient,
    AgentSessionClient,
    create_service_app,
)
from msgflux.tools.builtin import ApplyPatchTool, BashTool, ReadFileTool, WriteTool
from msgflux.utils.msgspec import msgspec_dumps

TOKEN = "tool-error-integration-token"


def _tool_call(name, arguments):
    calls = ToolCallAggregator()
    calls.process(0, "call-1", name, msgspec_dumps(arguments))
    response = ModelResponse()
    response.set_response_type("tool_call")
    response.add(calls)
    return response


def _text_response(text):
    response = ModelResponse()
    response.set_response_type("text_generation")
    response.add(text)
    return response


@asynccontextmanager
async def _remote_session(agent, *, thread_id):
    service = AgentService(store=SQLiteServiceStore())
    service.register("main", lambda _thread: AgentSession(agent))
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(128)
    listener.setblocking(False)
    port = listener.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(
            create_service_app(service, token=TOKEN),
            host="127.0.0.1",
            port=port,
            log_level="error",
            access_log=False,
            ws="none",
            lifespan="on",
            timeout_graceful_shutdown=2,
        )
    )
    server_task = asyncio.create_task(server.serve(sockets=[listener]))
    try:
        while not server.started:
            if server_task.done():
                await server_task
                raise RuntimeError("HTTP server exited before startup")
            await asyncio.sleep(0.01)
        client = AgentServiceClient(f"http://127.0.0.1:{port}", token=TOKEN)
        session = await AgentSessionClient.open(client, thread_id=thread_id)
        yield client, session
    finally:
        server.should_exit = True
        try:
            await asyncio.wait_for(server_task, timeout=5)
        finally:
            if not server_task.done():
                server_task.cancel()
                await asyncio.gather(server_task, return_exceptions=True)
            listener.close()
            await service.aclose()
            service.store.close()
        if "client" in locals():
            await client.aclose()


async def _collect_until_run_end(watcher, run_id):
    events = []
    while True:
        event = await asyncio.wait_for(anext(watcher), timeout=5)
        events.append(event)
        if event.run_id == run_id and event.type in {
            "run.end",
            "run.error",
            "run.paused",
            "run.interrupted",
        }:
            return events


@pytest.mark.asyncio
async def test_permission_error_reaches_model_and_http_events_without_arguments(
    tmp_path,
):
    root = tmp_path / "readonly"
    root.mkdir()
    workspace = AgentWorkspace.local(root)

    model = Mock()
    model.model_type = "chat_completion"
    agent = Agent(
        name="main",
        model=model,
        tools=[BashTool()],
        agent_dir=tmp_path / "permission-agent-state",
        workspace=workspace,
    )
    model_history = []
    request_count = 0

    async def respond(**kwargs):
        nonlocal request_count
        request_count += 1
        assert request_count <= 3
        if request_count == 1:
            return _tool_call("bash", {"command": "touch private.txt"})
        model_history.extend(kwargs["messages"].to_chatml())
        return _text_response("I could not write the file.")

    agent.generator.aforward = AsyncMock(side_effect=respond)
    async with _remote_session(agent, thread_id="permission-error-thread") as (
        client,
        session,
    ):
        policy = await session.workspace_policy()
        await session.update_workspace_policy(
            permissions="read-only", expected_revision=policy.revision
        )
        watch_context = session.watch()
        watcher = await asyncio.wait_for(watch_context.__aenter__(), timeout=5)
        try:
            admission = await session.prompt("run touch", request_id="write")
            events = await _collect_until_run_end(watcher, admission.run_id)
        finally:
            await watch_context.__aexit__(None, None, None)
        settled = await asyncio.wait_for(session.wait("write"), timeout=5)
        assert settled.status == "completed"
        assert not (root / "private.txt").exists()

        feedback = "\n".join(str(item.get("content", "")) for item in model_history)
        assert request_count == 2
        assert "missing tool permissions: process.execute" in feedback.lower()
        assert "private.txt" not in feedback

        assert any(event.type == "tool.permission_denied" for event in events), (
            ",".join(event.type for event in events)
        )
        denied = next(
            event for event in events if event.type == "tool.permission_denied"
        )
        blocked = next(event for event in events if event.type == "tool.blocked")
        assert denied.data["tool_call_id"] == "call-1"
        assert denied.data["tool_name"] == "bash"
        assert denied.data["code"] == "tool_permission_denied"
        assert "arguments" not in denied.data
        assert denied.data["error"]["code"] == "tool_permission_denied"
        assert denied.data["error"]["message"] in feedback
        assert blocked.data["error"] == denied.data["error"]
        assert await client.receipt(session.thread_id, "write") == settled
    await workspace.aclose()


@pytest.mark.asyncio
async def test_read_only_write_error_reaches_model_and_tool_end_event(tmp_path):
    root = tmp_path / "read-only-write"
    root.mkdir()
    workspace = AgentWorkspace.local(root, read_only=True)
    model = Mock()
    model.model_type = "chat_completion"
    agent = Agent(
        name="main",
        model=model,
        tools=[WriteTool()],
        agent_dir=tmp_path / "write-agent-state",
        workspace=workspace,
    )
    model_history = []
    request_count = 0

    async def respond(**kwargs):
        nonlocal request_count
        request_count += 1
        assert request_count <= 3
        if request_count == 1:
            return _tool_call("write", {"path": "note.txt", "content": "new"})
        model_history.extend(kwargs["messages"].to_chatml())
        return _text_response("The write was blocked.")

    agent.generator.aforward = AsyncMock(side_effect=respond)
    async with _remote_session(agent, thread_id="write-error-thread") as (
        _client,
        session,
    ):
        watch_context = session.watch()
        watcher = await asyncio.wait_for(watch_context.__aenter__(), timeout=5)
        try:
            admission = await session.prompt("write note", request_id="write")
            events = await _collect_until_run_end(watcher, admission.run_id)
        finally:
            await watch_context.__aexit__(None, None, None)
        settled = await asyncio.wait_for(session.wait("write"), timeout=5)
        assert settled.status == "completed"
        assert request_count == 2
        assert not (root / "note.txt").exists()
        feedback = "\n".join(str(item.get("content", "")) for item in model_history)
        assert "workspace is read-only" in feedback.lower()
        tool_end = next(event for event in events if event.type == "tool.end")
        assert "workspace is read-only" in tool_end.data["error"].lower()
        assert tool_end.data["result"] is None
    await workspace.aclose()


@pytest.mark.asyncio
async def test_missing_read_error_and_bash_exit_are_preserved_over_http(tmp_path):
    root = tmp_path / "error-workspace"
    root.mkdir()
    workspace = AgentWorkspace.local(root)
    model = Mock()
    model.model_type = "chat_completion"
    agent = Agent(
        name="main",
        model=model,
        tools=[ReadFileTool(), BashTool()],
        agent_dir=tmp_path / "error-agent-state",
        workspace=workspace,
    )
    histories = []
    request_count = 0

    async def respond(**kwargs):
        nonlocal request_count
        request_count += 1
        if request_count == 1:
            return _tool_call("read", {"path": "missing.txt"})
        histories.append(kwargs["messages"].to_chatml())
        if request_count == 2:
            return _tool_call(
                "bash", {"command": "printf stdout; printf stderr >&2; exit 7"}
            )
        return _text_response("Recovered from the read error and command exit.")

    agent.generator.aforward = AsyncMock(side_effect=respond)
    try:
        async with _remote_session(agent, thread_id="missing-read-thread") as (
            _client,
            session,
        ):
            watch_context = session.watch()
            watcher = await asyncio.wait_for(watch_context.__aenter__(), timeout=5)
            try:
                admission = await session.prompt("read missing file", request_id="read")
                events = await _collect_until_run_end(watcher, admission.run_id)
            finally:
                await watch_context.__aexit__(None, None, None)
            settled = await asyncio.wait_for(session.wait("read"), timeout=5)
            assert settled.status == "completed"
            assert request_count == 3

            read_end = next(
                event
                for event in events
                if event.type == "tool.end" and event.data["tool_name"] == "read"
            )
            read_info = read_end.data["error_info"]
            assert read_end.data["result"] is None
            assert read_end.data["error"] == read_info["message"]
            assert read_info["code"] == "tool_execution_failed"
            assert read_info["details"]["exception_type"] == "FileNotFoundError"
            assert "missing.txt" in read_info["message"]

            bash_end = next(
                event
                for event in events
                if event.type == "tool.end" and event.data["tool_name"] == "bash"
            )
            command = bash_end.data["result"]["results"][0]
            assert command == {
                "status": "exited",
                "stdout": "stdout",
                "stderr": "stderr",
                "returncode": 7,
            }
            assert bash_end.data["error"] is None
            assert bash_end.data["error_info"] is None

            feedback = "\n".join(str(history) for history in histories)
            assert read_info["message"] in feedback
            assert "returncode" in feedback and "7" in feedback
            snapshot = await session.snapshot()
            history = str(snapshot.messages)
            assert read_info["message"] in history
            assert "returncode" in history and "7" in history
    finally:
        await workspace.aclose()


@pytest.mark.asyncio
async def test_apply_patch_create_existing_error_reaches_model_and_http_events(
    tmp_path,
):
    root = tmp_path / "patch-workspace"
    root.mkdir()
    target = root / "note.txt"
    target.write_text("original")
    workspace = AgentWorkspace.local(root, approval_policy="never")
    model = Mock()
    model.model_type = "chat_completion"
    agent = Agent(
        name="main",
        model=model,
        tools=[ApplyPatchTool()],
        agent_dir=tmp_path / "patch-agent-state",
        workspace=workspace,
    )
    model_history = []
    request_count = 0

    async def respond(**kwargs):
        nonlocal request_count
        request_count += 1
        assert request_count <= 3
        if request_count == 1:
            return _tool_call(
                "apply_patch",
                {"operation": "create", "path": "note.txt", "diff": "+new"},
            )
        model_history.extend(kwargs["messages"].to_chatml())
        return _text_response("The file already exists.")

    agent.generator.aforward = AsyncMock(side_effect=respond)
    async with _remote_session(agent, thread_id="patch-error-thread") as (
        _client,
        session,
    ):
        watch_context = session.watch()
        watcher = await asyncio.wait_for(watch_context.__aenter__(), timeout=5)
        try:
            admission = await session.prompt("create the file", request_id="create")
            events = await _collect_until_run_end(watcher, admission.run_id)
        finally:
            await watch_context.__aexit__(None, None, None)
        settled = await asyncio.wait_for(session.wait("create"), timeout=5)
        assert settled.status == "completed"
        assert request_count == 2
        assert target.read_text() == "original"
        feedback = "\n".join(str(item.get("content", "")) for item in model_history)
        assert "file already exists: /note.txt" in feedback.lower()
        tool_events = [
            event for event in events if event.type in {"tool.end", "tool.blocked"}
        ]
        assert tool_events
        assert any(
            "file already exists: /note.txt" in str(event.data).lower()
            for event in tool_events
        )
    await workspace.aclose()


class _FailingResultStore(ToolResultStore):
    def put(self, chunks, *, media_type="application/octet-stream"):
        del chunks, media_type
        raise OSError("/private/server/path/result-store is unavailable")

    def get(self, result_id: str) -> ToolResultRef:
        raise KeyError(result_id)

    def iter_bytes(
        self, reference: ToolResultRef, *, offset=0, limit=None, chunk_size=65536
    ):
        del reference, offset, limit, chunk_size
        raise KeyError("unavailable")
        yield b""


@pytest.mark.asyncio
async def test_offload_failure_is_sanitized_in_feedback_and_remote_events(tmp_path):
    @mf.tool_config(name_override="large_report")
    def large_report() -> dict:
        return {"report": "payload-marker-" * 1000}

    model = Mock()
    model.model_type = "chat_completion"
    agent = Agent(
        name="main",
        model=model,
        tools=[large_report],
        agent_dir=tmp_path / "offload-agent-state",
    )
    handle = agent.tool_library.register_extension(
        "tool_output_offload",
        ToolOutputOffloadExtension(
            _FailingResultStore(), max_inline_bytes=64, preview_bytes=8
        ),
    )
    model_history = []
    request_count = 0

    async def respond(**kwargs):
        nonlocal request_count
        request_count += 1
        if request_count == 1:
            return _tool_call("large_report", {})
        model_history.extend(kwargs["messages"].to_chatml())
        return _text_response("The report was not available.")

    agent.generator.aforward = AsyncMock(side_effect=respond)
    async with _remote_session(agent, thread_id="offload-error-thread") as (
        client,
        session,
    ):
        watch_context = session.watch()
        watcher = await asyncio.wait_for(watch_context.__aenter__(), timeout=5)
        try:
            admission = await session.prompt("prepare report", request_id="report")
            events = await _collect_until_run_end(watcher, admission.run_id)
        finally:
            await watch_context.__aexit__(None, None, None)
        settled = await session.wait("report")
        assert settled.status == "completed"

        feedback = "\n".join(str(item.get("content", "")) for item in model_history)
        assert request_count == 2
        assert "Tool output processing failed (OSError)" in feedback
        assert "do not retry automatically" in feedback
        assert "private/server/path" not in feedback
        assert "payload-marker-" not in feedback

        handler_error = next(event for event in events if event.type == "handler.error")
        tool_end = next(event for event in events if event.type == "tool.end")
        for data in (handler_error.data, tool_end.data):
            encoded = str(data)
            assert "Tool output processing failed (OSError)" in encoded
            assert "private/server/path" not in encoded
            assert "payload-marker-" not in encoded
        assert "error" in tool_end.data and tool_end.data["result"] is None
        error_info = tool_end.data["error_info"]
        assert error_info["code"] == "tool_execution_failed"
        assert error_info["message"] == tool_end.data["error"]
        assert error_info["details"] == {}
        assert "private/server/path" not in str(error_info)
        assert "payload-marker-" not in str(error_info)
        assert await client.receipt(session.thread_id, "report") == settled
    handle.remove()
