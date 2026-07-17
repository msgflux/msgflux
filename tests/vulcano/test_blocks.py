import pytest

from msgflux.vulcano import (
    BlockKind,
    BlockStatus,
    CommandContext,
    ContentBlock,
    EventType,
)
from msgflux.vulcano.commands import CommandRegistry


class _CommandApi:
    services = {}


def test_content_block_round_trips_transport_payload():
    block = ContentBlock(
        block_id="blk_diff",
        kind=BlockKind.DIFF,
        content="+added",
        status=BlockStatus.COMPLETED,
        title="Patch",
        details={"path": "src/parser.py"},
    )

    assert ContentBlock.from_payload(block.to_payload()) == block


@pytest.mark.asyncio
async def test_command_context_emits_typed_block_lifecycle():
    events = []

    async def emit(event):
        events.append(event)

    context = CommandContext(
        commands=CommandRegistry(),
        api=_CommandApi(),
        _event_emitter=emit,
    )

    block_id = await context.start_block(
        BlockKind.REASONING,
        block_id="blk_reasoning",
        title="Thinking",
    )
    await context.update_block(block_id, "Inspecting the parser.")
    await context.complete_block(block_id)

    assert [event.type for event in events] == [
        EventType.BLOCK_STARTED,
        EventType.BLOCK_DELTA,
        EventType.BLOCK_COMPLETED,
    ]
    assert events[0].payload == {
        "block_id": "blk_reasoning",
        "kind": BlockKind.REASONING,
        "content": "",
        "status": BlockStatus.STREAMING,
        "title": "Thinking",
        "details": {},
    }
    assert events[1].payload["delta"] == "Inspecting the parser."
    assert events[2].payload["status"] == BlockStatus.COMPLETED


@pytest.mark.asyncio
async def test_command_context_emits_tool_lifecycle():
    events = []

    async def emit(event):
        events.append(event)

    context = CommandContext(
        commands=CommandRegistry(),
        api=_CommandApi(),
        _event_emitter=emit,
    )

    tool_call_id = await context.start_tool(
        "search",
        {"query": "streaming"},
        tool_call_id="tool_search",
    )
    await context.update_tool(tool_call_id, "search", {"matches": 1})
    await context.complete_tool(tool_call_id, "search", ["runtime.py"])

    assert [event.type for event in events] == [
        EventType.TOOL_STARTED,
        EventType.TOOL_UPDATED,
        EventType.TOOL_COMPLETED,
    ]
    assert events[0].payload["arguments"] == {"query": "streaming"}
    assert events[1].payload["update"] == {"matches": 1}
    assert events[2].payload["result"] == ["runtime.py"]
    assert events[2].payload["status"] == BlockStatus.COMPLETED
