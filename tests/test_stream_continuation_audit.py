import asyncio

import pytest

from msgflux.models.response import ModelStreamResponse


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("add_chunk", "consume"),
    [
        ("add_reasoning", "consume_reasoning"),
        ("add_reasoning_summary", "consume_reasoning_summary"),
    ],
)
async def test_reasoning_and_content_consumers_share_linked_streams(add_chunk, consume):
    source = ModelStreamResponse(mode="async")
    child = ModelStreamResponse(mode="async")
    finalized = []
    source.add("first-output")
    child.add("second-output")
    getattr(source, add_chunk)("first")
    getattr(child, add_chunk)("second")
    source.finish()
    child.finish()
    source._add_consumer_finalizer(lambda _state: finalized.append("source"))
    child._add_consumer_finalizer(lambda _state: finalized.append("child"))
    source._set_continuation(child)

    reasoning, content = await asyncio.gather(
        _collect(getattr(source, consume)()), _collect(source.consume())
    )
    assert reasoning == ["first", "second"]
    assert content == ["first-output", "second-output"]
    assert finalized == ["source", "child"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("add_chunk", "consume"),
    [
        ("add_reasoning", "consume_reasoning"),
        ("add_reasoning_summary", "consume_reasoning_summary"),
    ],
)
async def test_abandoned_reasoning_consumer_does_not_interrupt_linked_child(
    add_chunk, consume
):
    source = ModelStreamResponse(mode="async")
    child = ModelStreamResponse(mode="async")
    source.add("first-output")
    getattr(source, add_chunk)("first")
    source.finish()
    child.add("later-output")
    getattr(child, add_chunk)("later")
    source._set_continuation(child)
    reasoning = getattr(source, consume)()
    content = asyncio.create_task(_collect(source.consume()))

    assert await anext(reasoning) == "first"
    await reasoning.aclose()
    assert not child._is_finalized()
    child.add("last-output")
    child.finish()

    assert await content == ["first-output", "later-output", "last-output"]
    assert child._final_status == "completed"


@pytest.mark.asyncio
async def test_event_observer_close_preserves_multilevel_chain_for_root_observer():
    source, child, grandchild = _three_stream_chain()
    observer_a = source.consume_events()
    observer_b = source.consume_events()

    # B owns the root chain but has not advanced into the first continuation.
    assert (await anext(observer_a)).data == "root"
    assert (await anext(observer_b)).data == "root"
    assert (await anext(observer_a)).type == "message.end"
    assert (await anext(observer_a)).type == "message.start"
    assert (await anext(observer_a)).data == "child"
    assert (await anext(observer_a)).type == "message.end"
    assert (await anext(observer_a)).type == "message.start"
    assert (await anext(observer_a)).data == "grandchild"

    try:
        await observer_a.aclose()
        assert child._is_finalized()
        assert not grandchild._is_finalized()

        child.finish()
        grandchild.finish()
        remaining = [event async for event in observer_b]
        assert [event.data for event in remaining if event.type == "output.delta"] == [
            "child",
            "grandchild",
        ]
        assert child._final_status == grandchild._final_status == "completed"
    finally:
        await observer_b.aclose()


@pytest.mark.asyncio
async def test_event_observer_close_preserves_chain_for_content_observer():
    source, child, grandchild = _three_stream_chain()
    observer_a = source.consume_events()
    observer_b = source.consume()

    assert (await anext(observer_a)).data == "root"
    assert await anext(observer_b) == "root"
    assert (await anext(observer_a)).type == "message.end"
    assert (await anext(observer_a)).type == "message.start"
    assert (await anext(observer_a)).data == "child"
    assert (await anext(observer_a)).type == "message.end"
    assert (await anext(observer_a)).type == "message.start"
    assert (await anext(observer_a)).data == "grandchild"

    try:
        await observer_a.aclose()
        assert child._is_finalized()
        assert not grandchild._is_finalized()

        child.finish()
        grandchild.finish()
        assert [chunk async for chunk in observer_b] == ["child", "grandchild"]
        assert child._final_status == grandchild._final_status == "completed"
    finally:
        await observer_b.aclose()


@pytest.mark.asyncio
async def test_continuation_rejects_child_with_active_consumer_without_rebinding_it():
    source = ModelStreamResponse(mode="async")
    child = ModelStreamResponse(mode="async")
    child_consumer = child.consume()
    pending_chunk = asyncio.create_task(anext(child_consumer))
    await asyncio.sleep(0)

    with pytest.raises(RuntimeError, match="active or cleaned-up consumers"):
        source._set_continuation(child)

    child.add("still owned")
    child.finish()
    assert await pending_chunk == "still owned"
    assert [chunk async for chunk in child_consumer] == []


@pytest.mark.asyncio
async def test_late_event_observer_can_replay_while_child_cleanup_is_pending():
    source = ModelStreamResponse(mode="async")
    child = ModelStreamResponse(mode="async")
    finalizer_started = asyncio.Event()
    release_finalizer = asyncio.Event()
    finalizer_error = OSError("child finalizer failed")
    finalizer_calls = 0

    async def fail_finalizer(_state):
        nonlocal finalizer_calls
        finalizer_calls += 1
        finalizer_started.set()
        await release_finalizer.wait()
        raise finalizer_error

    source.add("root output")
    source.finish()
    child.add("child output")
    child.add_finalizer(fail_finalizer)
    source._set_continuation(child)

    first = source.consume_events()
    close_first = None
    late_observer = None
    try:
        assert (await anext(first)).data == "root output"
        close_first = asyncio.create_task(first.aclose())
        await asyncio.wait_for(finalizer_started.wait(), timeout=1)

        late_observer = source.consume_events()
        assert (await anext(late_observer)).data == "root output"
        assert (await anext(late_observer)).type == "message.end"
        assert (await anext(late_observer)).type == "message.start"
        assert (await anext(late_observer)).data == "child output"

        release_finalizer.set()
        with pytest.raises(OSError) as close_error:
            await asyncio.wait_for(close_first, timeout=3)
        assert close_error.value is finalizer_error

        with pytest.raises(OSError) as observer_error:
            async for _event in late_observer:
                pass
        assert observer_error.value is finalizer_error
        assert finalizer_calls == 1
        assert child._final_status == "interrupted"
        assert source._event_subscribers == child._event_subscribers == []
    finally:
        release_finalizer.set()
        if close_first is not None:
            cleanup = [close_first]
        else:
            cleanup = [first.aclose()]
        if late_observer is not None:
            cleanup.append(late_observer.aclose())
        await asyncio.wait_for(asyncio.gather(*cleanup, return_exceptions=True), 3)


async def _collect(stream):
    return [chunk async for chunk in stream]


def _three_stream_chain():
    source = ModelStreamResponse(mode="async")
    child = ModelStreamResponse(mode="async")
    grandchild = ModelStreamResponse(mode="async")
    source.add("root")
    source.finish()
    child.add("child")
    child.finish()
    grandchild.add("grandchild")
    source._set_continuation(child)
    child._set_continuation(grandchild)
    return source, child, grandchild
