"""Consumer cancellation must not hide failures from owned finalizers."""

import asyncio
import logging

import pytest

from msgflux.models.response import ModelStreamResponse


@pytest.mark.asyncio
@pytest.mark.parametrize("finalizer_fails", [False, True])
@pytest.mark.parametrize("provider_fails", [False, True])
async def test_cancelled_consumer_exits_promptly_and_observes_finalizer_failure(
    finalizer_fails, provider_fails, caplog
):
    caplog.set_level(logging.ERROR, logger="msgflux._private.response")
    response_logger = logging.getLogger("msgflux._private.response")
    response_logger.addHandler(caplog.handler)
    log_observed = asyncio.Event()

    class FinalizerLogHandler(logging.Handler):
        def emit(self, record):
            if record.exc_info and record.exc_info[1] is finalizer_error:
                log_observed.set()

    observer = FinalizerLogHandler()
    response_logger.addHandler(observer)
    response = ModelStreamResponse(mode="async")
    finalizer_started = asyncio.Event()
    release_finalizer = asyncio.Event()
    finalizer_done = asyncio.Event()
    finalizer_error = RuntimeError("stream settlement failed")
    loop_errors = []
    loop = asyncio.get_running_loop()
    old_exception_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: loop_errors.append(context))

    async def wait_then_fail(_state):
        finalizer_started.set()
        try:
            await release_finalizer.wait()
            if finalizer_fails:
                raise finalizer_error
        finally:
            finalizer_done.set()

    response.add_finalizer(wait_then_fail)
    primary_error = OSError("provider disconnected") if provider_fails else None
    response.finish(error=primary_error)

    async def consume():
        async for _event in response.consume_events():
            pass

    task = asyncio.create_task(consume())
    try:
        await asyncio.wait_for(finalizer_started.wait(), timeout=5)
        task.cancel("consumer cancelled")
        with pytest.raises(
            asyncio.CancelledError, match="consumer cancelled"
        ) as raised:
            await asyncio.wait_for(task, timeout=1)

        assert raised.value.__cause__ is primary_error
        assert not finalizer_done.is_set()
        release_finalizer.set()
        await asyncio.wait_for(finalizer_done.wait(), timeout=5)
        # Allow the gather's done callback to retrieve its terminal exception.
        if finalizer_fails:
            await asyncio.wait_for(log_observed.wait(), timeout=5)
        else:
            await asyncio.sleep(0)
        assert not loop_errors
        records = [
            record
            for record in caplog.records
            if record.exc_info is not None and record.exc_info[1] is finalizer_error
        ]
        assert bool(records) is finalizer_fails
        if finalizer_fails:
            assert records[0].getMessage() == (
                "Stream finalizer failed after consumer cancellation"
            )
            assert all(record.exc_info[1] is finalizer_error for record in records)
    finally:
        release_finalizer.set()
        try:
            await asyncio.wait_for(
                asyncio.gather(task, return_exceptions=True), timeout=5
            )
        finally:
            try:
                await asyncio.wait_for(finalizer_done.wait(), timeout=5)
                if finalizer_fails:
                    await asyncio.wait_for(log_observed.wait(), timeout=5)
            finally:
                loop.set_exception_handler(old_exception_handler)
                response_logger.removeHandler(caplog.handler)
                response_logger.removeHandler(observer)


@pytest.mark.asyncio
@pytest.mark.parametrize("primary_kind", [None, "error", "cancelled"])
async def test_finalizer_self_cancellation_preserves_primary_error(primary_kind):
    response = ModelStreamResponse(mode="async")
    finalizer_cancellation = asyncio.CancelledError("finalizer cancelled")
    if primary_kind == "error":
        primary_error = OSError("provider disconnected")
    elif primary_kind == "cancelled":
        primary_error = asyncio.CancelledError("provider cancelled")
    else:
        primary_error = None

    async def cancel_finalizer(_state):
        raise finalizer_cancellation

    response.add_finalizer(cancel_finalizer)
    response.finish(error=primary_error)

    async def consume():
        async for _event in response.consume_events():
            pass

    if primary_error is None:
        with pytest.raises(asyncio.CancelledError) as raised:
            await consume()
        assert isinstance(raised.value, asyncio.CancelledError)
    else:
        with pytest.raises(type(primary_error)) as raised:
            await consume()
        assert raised.value is primary_error
        assert isinstance(raised.value.__cause__, asyncio.CancelledError)
