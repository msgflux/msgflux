"""Consumer completion callbacks must run after producer finalizers fail."""

import asyncio

import pytest

from msgflux.models.response import ModelStreamResponse
from msgflux.nn.events import emit_model_response_events
from msgflux.nn.modules.module import Module
from msgflux.runtime.events import _EventSink, _capture_events


@pytest.mark.asyncio
@pytest.mark.parametrize("primary", ["provider", "cancel", "none"])
async def test_pending_finalizer_failure_preserves_stream_outcome_and_terminal_event(
    primary,
):
    response = ModelStreamResponse(mode="async")
    finalizer_error = RuntimeError("stream settlement failed")
    primary_error = None

    async def fail_finalizer(_state):
        await asyncio.sleep(0)
        raise finalizer_error

    response.set_response_type("text_generation")
    response.add_finalizer(fail_finalizer)
    if primary == "provider":
        primary_error = OSError("provider disconnected")
        response.set_error(primary_error)
        response.finish(status="failed")
    elif primary == "cancel":
        primary_error = asyncio.CancelledError("stream cancelled")
        response.set_error(primary_error)
        response.finish(status="interrupted")
    else:
        response.finish()

    events = []
    with _capture_events(_EventSink(events.append)):
        emit_model_response_events(response)
        if primary == "provider":
            error_type = ExceptionGroup
        elif primary == "cancel":
            error_type = asyncio.CancelledError
        else:
            error_type = RuntimeError
        with pytest.raises(error_type) as raised:
            await asyncio.wait_for(Module._aconsume_event_response(response), timeout=5)

    if primary == "provider":
        assert raised.value.exceptions == (primary_error, finalizer_error)
    elif primary == "cancel":
        assert raised.value is primary_error
        assert raised.value.__cause__ is finalizer_error
    else:
        assert raised.value is finalizer_error
    assert any(event.type == "model.response" for event in events)


@pytest.mark.asyncio
@pytest.mark.parametrize("finalizer_fails", [False, True])
async def test_early_consumer_error_waits_for_pending_finalizer_and_preserves_causes(
    finalizer_fails,
):
    response = ModelStreamResponse(mode="async")
    finalizer_started = asyncio.Event()
    release_finalizer = asyncio.Event()
    transformer_entered = asyncio.Event()
    transform_error = ValueError("output transform failed")
    finalizer_error = RuntimeError("stream settlement failed")

    async def wait_to_settle(_state):
        finalizer_started.set()
        await asyncio.wait_for(release_finalizer.wait(), timeout=5)
        if finalizer_fails:
            raise finalizer_error

    class FailingTransformer:
        def feed(self, _text):
            transformer_entered.set()
            raise transform_error

    response.set_response_type("text_generation")
    response.add("partial")
    response.add_finalizer(wait_to_settle)
    response.finish()

    events = []
    with _capture_events(_EventSink(events.append)):
        emit_model_response_events(response)
        task = asyncio.create_task(
            Module._aconsume_event_response(
                response, output_transformer=FailingTransformer()
            )
        )
        try:
            await asyncio.wait_for(finalizer_started.wait(), timeout=5)
            await asyncio.wait_for(transformer_entered.wait(), timeout=5)
            assert not task.done()

            release_finalizer.set()
            if finalizer_fails:
                with pytest.raises(ExceptionGroup) as raised:
                    await asyncio.wait_for(task, timeout=5)
                assert raised.value.exceptions == (transform_error, finalizer_error)
            else:
                with pytest.raises(
                    ValueError, match="output transform failed"
                ) as raised:
                    await asyncio.wait_for(task, timeout=5)
                assert raised.value is transform_error
        finally:
            release_finalizer.set()
            await asyncio.wait_for(
                asyncio.gather(task, return_exceptions=True), timeout=5
            )

    assert [event.type for event in events].count("model.response") == 1
