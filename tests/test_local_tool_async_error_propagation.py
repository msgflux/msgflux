from __future__ import annotations

import pytest

from msgflux.exceptions import (
    AbortRequestedError,
    TaskInterruptRequestedError,
    TaskPauseRequestedError,
)
from msgflux.nn.modules.tool.implementations import LocalTool


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [
        ValueError("local async failure"),
        AbortRequestedError("abort local tool"),
        TaskPauseRequestedError(message="pause local tool"),
        TaskInterruptRequestedError("task-1", "interrupt local tool"),
    ],
)
async def test_local_tool_sync_bridge_preserves_async_implementation_error(failure):
    calls = 0

    async def implementation():
        nonlocal calls
        calls += 1
        raise failure

    tool = LocalTool(
        name="sample",
        description="A sample async tool",
        annotations={},
        tool_config={"retry": False},
        impl=implementation,
    )

    with pytest.raises(type(failure)) as sync_error:
        tool()
    assert sync_error.value is failure

    with pytest.raises(type(failure)) as async_error:
        await tool.acall()
    assert async_error.value is failure
    assert calls == 2


@pytest.mark.asyncio
async def test_local_tool_sync_bridge_keeps_async_implementation_success():
    async def implementation(value):
        return value * 2

    tool = LocalTool(
        name="double",
        description="Double a value",
        annotations={"value": int},
        tool_config={"retry": False},
        impl=implementation,
    )

    assert tool(value=3) == 6
    assert await tool.acall(value=4) == 8
