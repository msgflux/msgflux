from __future__ import annotations

import asyncio
import inspect
from contextvars import ContextVar
from functools import partial
from threading import local

import pytest

from msgflux import TaskError
from msgflux._private.executor import AsyncWorker, Executor
from msgflux.envs import envs
from msgflux.exceptions import AbortRequestedError
from msgflux.nn import functional as F
from msgflux.nn.modules.tool.implementations import LocalTool


@pytest.fixture
def fresh_executor(monkeypatch):
    monkeypatch.setattr(envs, "executor_num_threads", 1)
    monkeypatch.setattr(envs, "executor_num_async_workers", 1)
    owned = Executor()
    monkeypatch.setattr(Executor, "_instance", owned)
    try:
        yield
    finally:
        owned.shutdown()


class _AsyncCallable:
    async def __call__(self, value: int, *, scale: int = 1, offset: int = 0) -> int:
        await asyncio.sleep(0)
        return value * scale + offset


def _close_coroutines(values):
    for value in values:
        if inspect.iscoroutine(value):
            value.close()


def test_wait_for_awaits_async_callable_objects_and_context(fresh_executor):
    marker = ContextVar("async_callable_fanout_marker", default=None)
    token = marker.set("caller-context")
    try:

        class ReadContext:
            async def __call__(self):
                return marker.get()

        context_result = F.wait_for(ReadContext())
        value_result = F.wait_for(_AsyncCallable(), 4, scale=2)
        try:
            assert context_result == "caller-context"
            assert value_result == 8
        finally:
            _close_coroutines((context_result, value_result))
    finally:
        marker.reset(token)


def test_map_gather_awaits_partial_async_callable_and_overrides_kwargs(
    fresh_executor,
):
    result = F.map_gather(
        partial(_AsyncCallable(), scale=3),
        args_list=[(2,), (4,)],
        kwargs_list=[{"scale": 5, "offset": 1}, {"offset": 2}],
    )
    try:
        assert result == (11, 14)
    finally:
        _close_coroutines(result)


def test_scatter_and_bcast_gather_await_async_callable_objects(fresh_executor):
    target = _AsyncCallable()
    scattered = F.scatter_gather(
        [target, partial(target, scale=4)], args_list=[(3,), (2,)]
    )
    broadcast = F.bcast_gather([target, partial(target, scale=5)], 2)
    try:
        assert scattered == (3, 8)
        assert broadcast == (2, 10)
    finally:
        _close_coroutines((*scattered, *broadcast))


def test_async_callable_failure_remains_indexed_with_original_exception(
    fresh_executor,
):
    error = ValueError("async callable failed")

    class Failing:
        async def __call__(self):
            raise error

    results = F.scatter_gather([Failing()])
    try:
        assert isinstance(results[0], TaskError)
        assert results[0].exception is error
        assert results[0].index == 0
    finally:
        _close_coroutines(results)


def test_async_callable_control_failure_keeps_existing_task_error_policy(
    fresh_executor,
):
    control = AbortRequestedError("cancel async callable")

    class Interrupted:
        async def __call__(self):
            raise control

    results = F.scatter_gather([Interrupted()])
    assert isinstance(results[0], TaskError)
    assert results[0].exception is control


@pytest.mark.asyncio
async def test_adetached_awaits_async_callable_objects_and_partials():
    marker = ContextVar("async_callable_detached_marker", default=None)
    token = marker.set("detached-context")
    observed = []
    completed = asyncio.Event()

    class Recording:
        async def __call__(self, value: int, *, offset: int = 0):
            observed.append((value, offset, marker.get()))
            if len(observed) == 2:
                completed.set()

    try:
        target = Recording()
        await F.adetached(target, 2)
        await F.adetached(partial(target, offset=5), 3)
        await asyncio.wait_for(completed.wait(), timeout=2)
        assert observed == [
            (2, 0, "detached-context"),
            (3, 5, "detached-context"),
        ]
    finally:
        marker.reset(token)


@pytest.mark.asyncio
async def test_adetached_keeps_acall_precedence():
    called = asyncio.Event()

    class AsyncCapable:
        async def __call__(self):
            raise AssertionError("__call__ must not replace .acall")

        async def acall(self):
            called.set()

    await F.adetached(AsyncCapable())
    await asyncio.wait_for(called.wait(), timeout=2)


def test_local_tool_uses_shared_partial_async_call_target():
    tool = LocalTool(
        name="partial_async_callable",
        description="A partially bound async callable",
        annotations={"value": int},
        tool_config={"retry": False},
        impl=partial(_AsyncCallable(), scale=4),
    )
    assert tool(value=2) == 8


@pytest.mark.asyncio
async def test_local_tool_acall_uses_shared_partial_async_call_target():
    tool = LocalTool(
        name="partial_async_callable",
        description="A partially bound async callable",
        annotations={"value": int},
        tool_config={"retry": False},
        impl=partial(_AsyncCallable(), scale=4),
    )
    assert await tool.acall(value=3) == 12


def test_rejected_async_callable_submission_closes_coroutine(monkeypatch):
    worker = AsyncWorker()
    worker.shutdown()
    executor = Executor.__new__(Executor)
    executor.async_workers = [worker]
    executor.async_worker_index = 0
    executor.num_async_workers = 1
    executor._thread_local = local()
    submitted = []
    original_submit = asyncio.run_coroutine_threadsafe

    def observe_submission(coro, loop):
        submitted.append(coro)
        return original_submit(coro, loop)

    monkeypatch.setattr(
        "msgflux._private.executor.asyncio.run_coroutine_threadsafe",
        observe_submission,
    )
    with pytest.raises(RuntimeError, match="Event loop is closed"):
        executor.submit(_AsyncCallable(), 1)

    assert len(submitted) == 1
    assert submitted[0].cr_frame is None
