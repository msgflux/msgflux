from __future__ import annotations

import pytest

from msgflux.nn.extensions import tool_library as tool_library_extension
from msgflux.nn.modules.tool import ToolLibrary


class _AsyncMCPClient:
    def __init__(self, *, connect_error=None, list_error=None):
        self.connect_error = connect_error
        self.list_error = list_error
        self.connect_calls = 0
        self.list_calls = 0
        self.disconnect_calls = 0

    async def connect(self):
        self.connect_calls += 1
        if self.connect_error is not None:
            raise self.connect_error

    async def list_tools(self, *, use_cache=True):
        self.list_calls += 1
        if self.list_error is not None:
            raise self.list_error
        return []

    async def disconnect(self):
        self.disconnect_calls += 1


@pytest.mark.parametrize("failure", ["connect", "list_tools"])
def test_mcp_setup_task_errors_are_not_registered(monkeypatch, caplog, failure):
    original_error = RuntimeError(f"{failure} failed")
    client = _AsyncMCPClient(
        connect_error=original_error if failure == "connect" else None,
        list_error=original_error if failure == "list_tools" else None,
    )
    monkeypatch.setattr(
        tool_library_extension.MCPClient,
        "from_stdio",
        lambda **_kwargs: client,
    )

    library = ToolLibrary(
        name="test",
        tools=[],
        mcp_servers=[{"name": "server", "transport": "stdio", "command": "unused"}],
    )

    assert client.connect_calls == 1
    assert client.list_calls == (failure == "list_tools")
    assert client.disconnect_calls == 1
    assert "server" not in library.mcp_clients
    assert not any(name.startswith("server__") for name in library.library)
    assert str(original_error) in caplog.text
    assert any(
        record.exc_info and record.exc_info[1] is original_error
        for record in caplog.records
    )
