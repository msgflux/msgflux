"""Stdio MCP servers cannot deadlock the transport by writing to stderr."""

from __future__ import annotations

import asyncio
import gc
import json
import sys
import warnings

import pytest

from msgflux.protocols.mcp.transports import StdioTransport

_SERVER = r"""
import json
import sys

amount = int(sys.argv[1])
# A real server commonly logs to stderr. Once the OS pipe buffer fills, this
# write blocks before the server can read stdin.
sys.stderr.write("E" * amount)
sys.stderr.flush()
for line in sys.stdin:
    message = json.loads(line)
    request_id = message.get("id")
    if request_id is not None:
        print(
            json.dumps({"jsonrpc": "2.0", "id": request_id, "result": {"ok": True}}),
            flush=True,
        )
"""


def _roundtrip(server, stderr_bytes):
    transport = StdioTransport(
        sys.executable,
        ["-u", str(server), str(stderr_bytes)],
        timeout=2.0,
    )
    processes = []

    async def scenario():
        await transport.connect()
        processes.append(transport._process)
        try:
            return await transport.send_request("ping")
        finally:
            await transport.disconnect()

    response = asyncio.run(asyncio.wait_for(scenario(), timeout=10))
    return response, transport, processes[0]


@pytest.mark.parametrize(
    "stderr_bytes",
    [4 * 1024, 4 * 1024 * 1024],
    ids=["small-stderr", "over-pipe-capacity"],
)
def test_stdio_stderr_does_not_block_requests(tmp_path, stderr_bytes):
    server = tmp_path / "stderr_server.py"
    server.write_text(_SERVER, encoding="utf-8")

    response, transport, process = _roundtrip(server, stderr_bytes)

    assert response["result"] == {"ok": True}
    assert process is not None
    assert process.returncode is not None

    # Force the closed event loop's transport objects to be collected while
    # capturing any unraisable teardown errors and resource warnings.
    unraisable = []
    previous_hook = sys.unraisablehook
    sys.unraisablehook = unraisable.append
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            del transport, process
            gc.collect()
    finally:
        sys.unraisablehook = previous_hook

    assert unraisable == []
    assert not [
        warning for warning in caught if issubclass(warning.category, ResourceWarning)
    ]
