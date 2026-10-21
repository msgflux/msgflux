from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from threading import Event

import pytest

from msgflux._private.client import BaseClient
from msgflux._private.executor import Executor
from msgflux.data.retrievers.base import BaseRetriever


class _SyncClient(BaseClient):
    def __init__(self, function):
        self.function = function

    def __call__(self):
        return self.function()


def _use_executor(monkeypatch, executor):
    monkeypatch.setattr(
        Executor,
        "get_instance",
        classmethod(lambda _cls: executor),
    )


@pytest.mark.asyncio
async def test_async_client_call_returns_sync_worker_result(monkeypatch):
    with ThreadPoolExecutor(max_workers=1) as executor:
        _use_executor(monkeypatch, executor)
        client = _SyncClient(lambda: "result")

        assert await client.acall() == "result"


@pytest.mark.asyncio
async def test_async_client_call_preserves_sync_worker_exception(monkeypatch):
    error = ValueError("worker failed")

    def fail():
        raise error

    with ThreadPoolExecutor(max_workers=1) as executor:
        _use_executor(monkeypatch, executor)
        client = _SyncClient(fail)

        with pytest.raises(ValueError) as raised:
            await client.acall()

        assert raised.value is error


@pytest.mark.asyncio
async def test_cancelling_queued_async_client_call_prevents_execution(monkeypatch):
    release_blocker = Event()
    client_called = Event()
    client_submitted = Event()

    class _TrackingExecutor(ThreadPoolExecutor):
        submissions = 0
        client_future = None

        def submit(self, function, *args, **kwargs):
            self.submissions += 1
            future = super().submit(function, *args, **kwargs)
            if self.submissions == 2:
                self.client_future = future
                client_submitted.set()
            return future

    def block():
        assert release_blocker.wait(timeout=3)

    def call_client():
        client_called.set()
        return "unexpected"

    with _TrackingExecutor(max_workers=1) as executor:
        executor.submit(block)
        _use_executor(monkeypatch, executor)
        client = _SyncClient(call_client)
        call = asyncio.create_task(client.acall())
        try:
            assert await asyncio.to_thread(client_submitted.wait, 2)
            call.cancel()
            with pytest.raises(asyncio.CancelledError):
                await call
            assert executor.client_future.cancelled()
        finally:
            release_blocker.set()

    assert not client_called.is_set()


@pytest.mark.asyncio
async def test_cancelling_running_async_client_call_does_not_claim_to_stop_thread(
    monkeypatch,
):
    release_worker = Event()
    worker_started = Event()
    worker_finished = Event()

    def slow_client_call():
        worker_started.set()
        assert release_worker.wait(timeout=3)
        worker_finished.set()
        return "finished"

    with ThreadPoolExecutor(max_workers=1) as executor:
        _use_executor(monkeypatch, executor)
        client = _SyncClient(slow_client_call)
        call = asyncio.create_task(client.acall())
        try:
            assert await asyncio.to_thread(worker_started.wait, 2)
            call.cancel()
            with pytest.raises(asyncio.CancelledError):
                await call
            assert not worker_finished.is_set()
        finally:
            release_worker.set()

        assert await asyncio.to_thread(worker_finished.wait, 2)


@pytest.mark.asyncio
async def test_retriever_fallback_preserves_arguments_and_execution_context():
    marker = ContextVar("retriever_bridge_marker", default=None)

    class SyncRetriever(BaseRetriever):
        def __call__(self, query, *, top_k):
            return query, top_k, marker.get()

    token = marker.set("caller-context")
    try:
        result = await SyncRetriever().acall("question", top_k=3)
    finally:
        marker.reset(token)

    assert result == ("question", 3, "caller-context")
