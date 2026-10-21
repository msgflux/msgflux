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


class _AsyncCallable:
    """Return a value asynchronously through `__call__`."""

    async def __call__(self, value: int) -> str:
        return f"value:{value}"


@pytest.mark.asyncio
async def test_async_callable_objects_are_awaited_by_tool_and_library_apis():
    import inspect

    from msgflux.nn.modules.tool.library import ToolLibrary

    direct = LocalTool(
        name="async_callable",
        description="An async callable object",
        annotations={"value": int},
        tool_config={"retry": False},
        impl=_AsyncCallable(),
    )
    assert direct(value=1) == "value:1"
    async_result = await direct.acall(value=2)
    assert async_result == "value:2"
    assert not inspect.isawaitable(async_result)

    library = ToolLibrary("async_callable", [_AsyncCallable])
    handle = library.get_handle()
    sync_result = handle("_AsyncCallable", value=3)
    async_result = await handle.acall("_AsyncCallable", value=4)
    assert sync_result == "value:3"
    assert async_result == "value:4"
    assert not inspect.isawaitable(sync_result)
    assert not inspect.isawaitable(async_result)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [
        ValueError("callable object failed"),
        AbortRequestedError("abort callable object"),
        TaskPauseRequestedError(message="pause callable object"),
        TaskInterruptRequestedError("task-2", "interrupt callable object"),
    ],
)
async def test_async_callable_object_errors_propagate_without_raw_coroutines(failure):
    import inspect

    class FailingAsyncCallable:
        """Raise one captured error asynchronously."""

        async def __call__(self, value: int) -> str:
            raise failure

    tool = LocalTool(
        name="failing_async_callable",
        description="An async callable object that fails",
        annotations={"value": int},
        tool_config={"retry": False},
        impl=FailingAsyncCallable(),
    )

    with pytest.raises(type(failure)) as sync_error:
        tool(value=1)
    assert sync_error.value is failure

    with pytest.raises(type(failure)) as async_error:
        await tool.acall(value=2)
    assert async_error.value is failure
    assert not inspect.isawaitable(async_error.value)


@pytest.mark.asyncio
async def test_sync_callable_objects_keep_sync_behavior():
    class SyncCallable:
        """Double a value synchronously."""

        def __call__(self, value: int) -> int:
            return value * 2

    tool = LocalTool(
        name="sync_callable",
        description="A sync callable object",
        annotations={"value": int},
        tool_config={"retry": False},
        impl=SyncCallable(),
    )
    assert tool(value=3) == 6
    assert await tool.acall(value=4) == 8


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [
        ValueError("sync callable failed"),
        TaskPauseRequestedError(message="pause sync callable"),
    ],
)
async def test_sync_callable_object_errors_keep_existing_propagation(failure):
    class FailingSyncCallable:
        """Raise one captured error synchronously."""

        def __call__(self, value: int) -> str:
            raise failure

    tool = LocalTool(
        name="failing_sync_callable",
        description="A sync callable object that fails",
        annotations={"value": int},
        tool_config={"retry": False},
        impl=FailingSyncCallable(),
    )

    with pytest.raises(type(failure)) as sync_error:
        tool(value=1)
    assert sync_error.value is failure

    with pytest.raises(type(failure)) as async_error:
        await tool.acall(value=2)
    assert async_error.value is failure
