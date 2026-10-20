import asyncio

import pytest

from msgflux.models.response import ModelStreamResponse


@pytest.mark.asyncio
async def test_consume_follows_stream_and_scalar_continuations():
    first = ModelStreamResponse(mode="async")
    second = ModelStreamResponse(mode="async")
    first.add("first")
    first.finish()
    second.add("second")
    second.finish()
    first._set_continuation(second)
    second._set_continuation("final value")

    assert [chunk async for chunk in first.consume()] == [
        "first",
        "second",
        "final value",
    ]
    assert first._settled_output == "final value"
    assert first._final_status == "completed"


@pytest.mark.asyncio
async def test_consume_events_preserves_boundaries_and_runs_finalizers_once():
    first = ModelStreamResponse(mode="async")
    second = ModelStreamResponse(mode="async")
    finalized = []
    first.add_finalizer(lambda _state: finalized.append("first"))
    second.add_finalizer(lambda _state: finalized.append("second"))
    first._add_consumer_finalizer(lambda _state: finalized.append("consumer-first"))
    second._add_consumer_finalizer(lambda _state: finalized.append("consumer-second"))
    first.add("first")
    first.finish()
    second.add("second")
    second.finish()
    first._set_continuation(second)

    events = [event async for event in first.consume_events()]
    assert [(event.type, event.data) for event in events] == [
        ("output.delta", "first"),
        ("message.end", {"content": "first"}),
        ("message.start", {"buffered": False}),
        ("output.delta", "second"),
    ]
    assert finalized.count("first") == finalized.count("second") == 1
    assert finalized.index("consumer-first") < finalized.index("consumer-second")
    assert first._final_status == second._final_status == "completed"


@pytest.mark.asyncio
async def test_continuation_attached_after_source_finish_waits_for_async_finalizer():
    source = ModelStreamResponse(mode="async")
    child = ModelStreamResponse(mode="async")
    finalizer_done = asyncio.Event()
    source.finish()

    async def finalize(_state):
        await asyncio.sleep(0)
        finalizer_done.set()

    source.add_finalizer(finalize)
    source._set_continuation(child)
    child.add("later")
    child.finish()

    chunks = [chunk async for chunk in source.consume()]
    assert chunks == ["later"]
    assert finalizer_done.is_set()


@pytest.mark.asyncio
async def test_continuation_failure_is_propagated_and_adopted():
    source = ModelStreamResponse(mode="async")
    child = ModelStreamResponse(mode="async")
    error = RuntimeError("continuation failed")
    source.add("before")
    source.finish()
    child.set_error(error)
    child.finish(error=error, status="failed")
    source._set_continuation(child)

    with pytest.raises(RuntimeError, match="continuation failed"):
        _ = [event async for event in source.consume_events()]
    assert source.error is error
    assert source._final_status == "failed"

    replayed = []
    with pytest.raises(RuntimeError) as replay_error:
        async for event in source.consume_events():
            replayed.append(event.type)
    assert replay_error.value is error
    assert "message.start" in replayed


@pytest.mark.asyncio
async def test_abandoned_event_consumer_interrupts_unconsumed_child():
    source = ModelStreamResponse(mode="async")
    child = ModelStreamResponse(mode="async")
    child_finalized = asyncio.Event()
    consumer_finalized = asyncio.Event()
    source.add("first")
    source.finish()
    child.add("later")

    async def finalize(_state):
        child_finalized.set()

    child.add_finalizer(finalize)
    child._add_consumer_finalizer(lambda _state: consumer_finalized.set())
    source._set_continuation(child)
    events = source.consume_events()

    assert (await anext(events)).data == "first"
    await events.aclose()

    assert child._final_status == "interrupted"
    assert child_finalized.is_set()
    assert consumer_finalized.is_set()


@pytest.mark.asyncio
async def test_abandoned_outer_event_consumer_removes_source_subscriber_immediately():
    source = ModelStreamResponse(mode="async")
    source.add("first")
    events = source.consume_events()

    assert (await anext(events)).data == "first"
    assert len(source._event_subscribers) == 1
    await events.aclose()

    assert source._event_subscribers == []


@pytest.mark.asyncio
async def test_abandoned_child_cleanup_does_not_mask_consumer_error():
    source = ModelStreamResponse(mode="async")
    child = ModelStreamResponse(mode="async")
    primary = RuntimeError("consumer failed")
    source.add("first")
    source.finish()

    def fail_cleanup(_state):
        raise OSError("child finalizer failed")

    child.add_finalizer(fail_cleanup)
    source._set_continuation(child)
    events = source.consume_events()
    assert (await anext(events)).data == "first"

    with pytest.raises(RuntimeError) as caught:
        await events.athrow(primary)
    assert caught.value is primary
    assert any("Continuation cleanup failed" in note for note in primary.__notes__)


@pytest.mark.asyncio
async def test_module_event_consumer_forwards_internal_message_boundaries(monkeypatch):
    from msgflux.nn.modules import module as module_lib
    from msgflux.nn.modules.module import Module
    from msgflux.runtime.events import EventType

    source = ModelStreamResponse(mode="async")
    child = ModelStreamResponse(mode="async")
    source.add("before")
    source.finish()
    child.add("after")
    child.finish()
    source._set_continuation(child)
    observed = []
    monkeypatch.setattr(
        module_lib,
        "emit_event",
        lambda event_type, data=None: observed.append((event_type, data)),
    )

    await Module._aconsume_event_response(source)

    assert [event_type for event_type, _data in observed] == [
        EventType.MESSAGE_DELTA,
        EventType.MESSAGE_END,
        EventType.MESSAGE_START,
        EventType.MESSAGE_DELTA,
    ]
    assert observed[1][1] == {"content": "before"}


@pytest.mark.asyncio
async def test_module_buffers_intermediate_chain_content_when_requested(monkeypatch):
    from msgflux.nn.modules import module as module_lib
    from msgflux.nn.modules.module import Module

    source = ModelStreamResponse(mode="async")
    child = ModelStreamResponse(mode="async")
    source.add("private intermediate")
    source.finish()
    child.add("final")
    child.finish()
    source._set_continuation(child)
    observed = []
    monkeypatch.setattr(
        module_lib,
        "emit_event",
        lambda event_type, data=None: observed.append((event_type, data)),
    )

    await Module._aconsume_event_response(source, emit_content=False)

    assert observed == []


@pytest.mark.asyncio
async def test_module_omits_untransformed_intermediate_boundaries(monkeypatch):
    from msgflux.nn.modules import module as module_lib
    from msgflux.nn.modules.module import Module
    from msgflux.runtime.events import EventType

    source = ModelStreamResponse(mode="async")
    child = ModelStreamResponse(mode="async")
    source.add("secret")
    source.finish()
    child.add("answer")
    child.finish()
    source._set_continuation(child)
    observed = []
    monkeypatch.setattr(
        module_lib,
        "emit_event",
        lambda event_type, data=None: observed.append((event_type, data)),
    )

    class Transformer:
        @staticmethod
        def feed(_text):
            return "masked"

    await Module._aconsume_event_response(source, output_transformer=Transformer())

    assert (EventType.MESSAGE_END, {"content": "secret"}) not in observed
    assert (EventType.MESSAGE_START, {"buffered": False}) not in observed
    assert observed == [
        (EventType.MESSAGE_DELTA, {"delta": "masked"}),
        (EventType.MESSAGE_DELTA, {"delta": "masked"}),
    ]
