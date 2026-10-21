"""MCPTool's sync adapter preserves errors returned by functional.wait_for."""

import pytest

from msgflux._private.executor import Executor
from msgflux.exceptions import TaskPauseRequestedError
from msgflux.nn.modules.tool import MCPTool, ToolLibrary
from msgflux.protocols.mcp import MCPError
from msgflux.protocols.mcp.types import (
    MCPContent,
    MCPTool as MCPToolInfo,
    MCPToolResult,
)
from msgflux.runtime.events import EventType, _EventSink, _capture_events


def _tool(client, *, retry=False):
    return MCPTool(
        name="lookup",
        mcp_client=client,
        mcp_tool_info=MCPToolInfo("lookup", "Lookup", {"type": "object"}),
        namespace="catalog",
        config={"retry": retry},
    )


class _FailingClient:
    def __init__(self, error):
        self.error = error
        self.calls = []

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        raise self.error


@pytest.mark.parametrize("control", [False, True], ids=["mcp-error", "pause"])
def test_sync_mcp_tool_call_preserves_client_exception(control):
    error = (
        TaskPauseRequestedError("task", "pause from MCP client")
        if control
        else MCPError("remote MCP request failed")
    )
    client = _FailingClient(error)
    tool = _tool(client)

    with pytest.raises(type(error)) as raised:
        tool()

    assert raised.value is error
    assert client.calls == [("lookup", {})]


@pytest.mark.asyncio
async def test_async_mcp_tool_call_preserves_client_exception():
    error = MCPError("remote MCP request failed")
    client = _FailingClient(error)
    tool = _tool(client)

    with pytest.raises(MCPError) as raised:
        await tool.acall()

    assert raised.value is error
    assert client.calls == [("lookup", {})]


def test_tool_library_records_sync_mcp_error_and_failure_event():
    error = MCPError("remote MCP request failed")
    client = _FailingClient(error)
    tool = _tool(client)
    library = ToolLibrary("catalog", [tool])
    events = []

    with _capture_events(_EventSink(events.append)):
        response = library([("call-mcp", "catalog__lookup", {})])

    call = response.tool_calls[0]
    assert call.result is None
    assert call.error == "remote MCP request failed"
    assert client.calls == [("lookup", {})]

    terminal = next(event for event in events if event.type == EventType.TOOL_END)
    assert terminal.data["tool_call_id"] == "call-mcp"
    assert terminal.data["error"] == "remote MCP request failed"
    assert terminal.data["error_info"] == {
        "code": "tool_execution_failed",
        "message": "remote MCP request failed",
        "details": {"exception_type": "MCPError"},
    }


def test_sync_mcp_dispatch_rejection_preserves_original_error(monkeypatch):
    error = RuntimeError("executor dispatch rejected")
    client = _FailingClient(MCPError("must not call MCP"))

    class RejectedExecutor:
        def submit(self, *_args, **_kwargs):
            raise error

    monkeypatch.setattr(
        Executor,
        "get_instance",
        classmethod(lambda _cls: RejectedExecutor()),
    )
    tool = _tool(client)

    with pytest.raises(RuntimeError) as raised:
        tool.forward()

    assert raised.value is error
    assert client.calls == []


def test_sync_mcp_tool_success_still_extracts_result():
    class SuccessfulClient:
        def __init__(self):
            self.calls = []

        async def call_tool(self, name, arguments):
            self.calls.append((name, arguments))
            return MCPToolResult(content=[MCPContent(type="text", text="found")])

    client = SuccessfulClient()
    assert _tool(client)() == "found"
    assert client.calls == [("lookup", {})]
