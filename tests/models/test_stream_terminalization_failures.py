"""Stream primary failures and finalizer failures must remain diagnosable."""

import asyncio

import pytest

from msgflux.exceptions import AbortRequestedError
from msgflux.models.response import ModelStreamResponse


def test_single_finalizer_failure_is_not_self_chained():
    response = ModelStreamResponse()
    finalizer_error = RuntimeError("stream settlement failed")

    def fail_finalizer(_state):
        raise finalizer_error

    response.add_finalizer(fail_finalizer)

    with pytest.raises(RuntimeError, match="stream settlement failed") as raised:
        response.finish()

    assert raised.value is finalizer_error
    assert finalizer_error.__cause__ is None
    assert response.error is finalizer_error


def test_sync_finalizer_failure_groups_it_with_primary_stream_error():
    response = ModelStreamResponse()
    primary_error = OSError("provider stream disconnected")
    finalizer_error = RuntimeError("stream settlement failed")

    def fail_finalizer(_state):
        raise finalizer_error

    response.add_finalizer(fail_finalizer)
    response.set_error(primary_error)

    with pytest.raises(
        ExceptionGroup, match="Stream and finalizer both failed"
    ) as raised:
        response.finish(status="failed")

    assert raised.value.exceptions == (primary_error, finalizer_error)
    assert response.error is raised.value
    assert response._final_status == "failed"
    assert response._content_closed
    assert response._events_closed


@pytest.mark.asyncio
@pytest.mark.parametrize("consumer", ["content", "events"])
async def test_async_consumer_groups_primary_stream_and_finalizer_errors(consumer):
    response = ModelStreamResponse(mode="async")
    primary_error = OSError("provider stream disconnected")
    finalizer_error = RuntimeError("async settlement failed")

    async def fail_finalizer(_state):
        await asyncio.sleep(0)
        raise finalizer_error

    response.add_finalizer(fail_finalizer)
    if consumer == "content":
        response.add("partial")
    response.set_error(primary_error)
    response.finish(status="failed")

    async def drain():
        if consumer == "content":
            async for _chunk in response.consume():
                pass
        else:
            async for _event in response.consume_events():
                pass

    with pytest.raises(
        ExceptionGroup, match="Stream and finalizer both failed"
    ) as raised:
        await asyncio.wait_for(drain(), timeout=5)

    assert raised.value.exceptions == (primary_error, finalizer_error)
    assert response.error is primary_error


@pytest.mark.asyncio
async def test_async_consumer_preserves_primary_cancellation_with_finalizer_failure():
    response = ModelStreamResponse(mode="async")
    primary_error = asyncio.CancelledError("stream cancelled")
    finalizer_error = RuntimeError("async settlement failed")

    async def fail_finalizer(_state):
        await asyncio.sleep(0)
        raise finalizer_error

    response.add_finalizer(fail_finalizer)
    response.set_error(primary_error)
    response.finish(status="interrupted")

    with pytest.raises(asyncio.CancelledError, match="stream cancelled") as raised:
        await asyncio.wait_for(response.consume_events().__anext__(), timeout=5)

    assert raised.value is primary_error
    assert raised.value.__cause__ is finalizer_error
    assert response.error is primary_error
    assert response._final_status == "interrupted"


@pytest.mark.asyncio
async def test_closing_async_consumer_surfaces_pending_finalizer_failure():
    response = ModelStreamResponse(mode="async")
    finalizer_error = RuntimeError("async settlement failed")

    async def fail_finalizer(_state):
        await asyncio.sleep(0)
        raise finalizer_error

    response.add_finalizer(fail_finalizer)
    response.add("partial")
    response.finish(status="completed")
    consumer = response.consume()

    assert await asyncio.wait_for(anext(consumer), timeout=5) == "partial"
    with pytest.raises(RuntimeError, match="async settlement failed") as raised:
        await asyncio.wait_for(consumer.aclose(), timeout=5)

    assert raised.value is finalizer_error


def test_finalizer_failure_preserves_primary_cancellation_identity_and_status():
    response = ModelStreamResponse()
    primary_error = asyncio.CancelledError("stream cancelled")
    finalizer_error = RuntimeError("stream settlement failed")

    def fail_finalizer(_state):
        raise finalizer_error

    response.add_finalizer(fail_finalizer)
    response.set_error(primary_error)

    with pytest.raises(asyncio.CancelledError, match="stream cancelled") as raised:
        response.finish(status="interrupted")

    assert raised.value is primary_error
    assert raised.value.__cause__ is finalizer_error
    assert response.error is primary_error
    assert response._final_status == "interrupted"
    assert response._content_closed
    assert response._events_closed


def test_finalizer_failure_preserves_cooperative_abort_identity_and_status():
    response = ModelStreamResponse()
    primary_error = AbortRequestedError("user pressed esc")
    finalizer_error = RuntimeError("stream settlement failed")

    def fail_finalizer(_state):
        raise finalizer_error

    response.add_finalizer(fail_finalizer)

    with pytest.raises(AbortRequestedError, match="user pressed esc") as raised:
        response.finish(error=primary_error, status="interrupted")

    assert raised.value is primary_error
    assert raised.value.__cause__ is finalizer_error
    assert response.error is primary_error
    assert response._final_status == "interrupted"
    assert response._content_closed
    assert response._events_closed
