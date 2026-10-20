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


async def _collect(stream):
    return [chunk async for chunk in stream]
