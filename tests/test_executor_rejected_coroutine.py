from __future__ import annotations

import asyncio
import gc
import warnings
from threading import Event

import pytest

from msgflux._private import executor as executor_module
from msgflux._private.executor import AsyncWorker, Executor


def _executor_for(worker):
    executor = Executor.__new__(Executor)
    executor.async_workers = [worker]
    executor.async_worker_index = 0
    executor.num_async_workers = 1
    return executor


def test_rejected_async_submission_closes_coroutine_without_warning(monkeypatch):
    async def work():
        return "done"

    worker = AsyncWorker()
    worker.shutdown()
    executor = _executor_for(worker)
    original_submit = asyncio.run_coroutine_threadsafe
    submitted_coroutines = []

    def observe_submission(coro, loop):
        submitted_coroutines.append(coro)
        return original_submit(coro, loop)

    monkeypatch.setattr(
        executor_module.asyncio,
        "run_coroutine_threadsafe",
        observe_submission,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        with pytest.raises(RuntimeError, match="Event loop is closed"):
            executor.submit(work)

        assert len(submitted_coroutines) == 1
        assert submitted_coroutines[0].cr_frame is None
        submitted_coroutines.clear()
        gc.collect()

    assert not [
        warning for warning in caught if "never awaited" in str(warning.message)
    ]


def test_rejected_async_submission_preserves_original_error(monkeypatch):
    async def work():
        return "done"

    worker = AsyncWorker()
    executor = _executor_for(worker)
    submission_error = RuntimeError("submission rejected")

    def reject_submission(_coro, _loop):
        raise submission_error

    monkeypatch.setattr(
        executor_module.asyncio,
        "run_coroutine_threadsafe",
        reject_submission,
    )
    try:
        with pytest.raises(RuntimeError) as raised:
            executor.submit(work)
        assert raised.value is submission_error
    finally:
        worker.shutdown()


def test_successful_async_submission_keeps_coroutine_owned_by_worker(monkeypatch):
    worker = AsyncWorker()
    executor = _executor_for(worker)
    started = Event()
    gate = {}
    submitted_coroutines = []
    original_submit = asyncio.run_coroutine_threadsafe

    async def work():
        event = asyncio.Event()
        gate["event"] = event
        started.set()
        await event.wait()
        return "done"

    def observe_submission(coro, loop):
        submitted_coroutines.append(coro)
        return original_submit(coro, loop)

    monkeypatch.setattr(
        executor_module.asyncio,
        "run_coroutine_threadsafe",
        observe_submission,
    )
    future = executor.submit(work)
    try:
        assert started.wait(timeout=2)
        assert len(submitted_coroutines) == 1
        assert submitted_coroutines[0].cr_frame is not None
        worker.loop.call_soon_threadsafe(gate["event"].set)
        assert future.result(timeout=2) == "done"
    finally:
        if "event" in gate:
            worker.loop.call_soon_threadsafe(gate["event"].set)
        worker.shutdown()
