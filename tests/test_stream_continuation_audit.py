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
