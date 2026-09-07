import pytest
from unittest.mock import AsyncMock

from msgflux.data.stores import InMemoryCheckpointStore
from msgflux.models.response import ModelStreamResponse
from msgflux.nn import ArtifactExtension, ArtifactReferenceRenderer, ArtifactRegistry
from msgflux.nn.modules.agent import Agent
from msgflux.runtime.context import ExecutionScope
from msgflux.runtime.events import EventType


def test_artifact_renderer_handles_split_markers_and_missing_values():
    registry = ArtifactRegistry()
    registry.register("REPORT", artifact_id="report-1")
    renderer = ArtifactReferenceRenderer(registry)

    assert renderer.feed("before {{arti") == "before "
    assert renderer.feed("fact:report-1}} after") == "REPORT after"
    assert renderer.render(" {{artifact:unknown}}") == " {{artifact:unknown}}"


def test_artifact_renderer_handles_every_boundary_and_escape_boundary():
    registry = ArtifactRegistry()
    registry.register("REPORT", artifact_id="report-1")
    value = "left {{artifact:report-1}} right"
    for split in range(1, len(value)):
        renderer = ArtifactReferenceRenderer(registry)
        rendered = renderer.feed(value[:split]) + renderer.feed(value[split:]) + renderer.finish()
        assert rendered == "left REPORT right"

    renderer = ArtifactReferenceRenderer(registry)
    assert renderer.feed("\\") == ""
    assert renderer.feed("{{artifact:report-1}}") == "{{artifact:report-1}}"


def test_artifact_renderer_does_not_expand_nested_content_or_oversized_marker():
    registry = ArtifactRegistry()
    registry.register("{{artifact:inner}}", artifact_id="outer")
    renderer = ArtifactReferenceRenderer(registry, max_marker_length=20)
    assert renderer.render("{{artifact:outer}}") == "{{artifact:inner}}"
    assert renderer.render("{{artifact:report-1}}") == "{{artifact:report-1}}"


def test_artifact_renderer_escapes_markers_and_bounds_buffer():
    registry = ArtifactRegistry()
    registry.register("x", artifact_id="x")
    renderer = ArtifactReferenceRenderer(registry, max_marker_length=32)

    assert renderer.render(r"\{{artifact:x}}") == "{{artifact:x}}"
    renderer.feed("{{artifact:" + "a" * 100)
    assert len(renderer._buffer) <= 32


def test_registry_rejects_paths_and_mutation():
    registry = ArtifactRegistry()
    registry.register("one", artifact_id="stable")
    with pytest.raises(ValueError):
        registry.register("two", artifact_id="stable")
    with pytest.raises(ValueError):
        registry.register("x", artifact_id="../secret")


def test_extension_exposes_incremental_renderer():
    extension = ArtifactExtension()
    extension.registry.register("value", artifact_id="v")
    renderer = extension.create_output_transformer()
    assert renderer.feed("{{artifact:v}}") == "value"


@pytest.mark.asyncio
async def test_agent_stream_renders_events_but_checkpoints_canonical_reference():
    registry = ArtifactRegistry()
    registry.register("expanded report", artifact_id="report")
    response = ModelStreamResponse(mode="async")
    response.set_response_type("text_generation")
    response.add("prefix {{artifact:")
    response.add("report}} suffix")
    response.finish()
    model = type("Model", (), {"model_type": "chat_completion"})()
    model.acall = AsyncMock(return_value=response)
    store = InMemoryCheckpointStore()
    agent = Agent(
        name="agent",
        model=model,
        checkpoint_store=store,
        extensions=[ArtifactExtension(registry)],
        config={"stream": True},
    )
    scope = ExecutionScope(thread_id="artifact-thread", run_id="artifact-run")
    events = [event async for event in agent.stream_events("question", scope=scope)]

    deltas = [
        event.data["delta"]
        for event in events
        if event.type == EventType.MESSAGE_DELTA
    ]
    end = next(event for event in events if event.type == EventType.MESSAGE_END)
    assert "".join(deltas) == "prefix expanded report suffix"
    assert end.data["content"] == "prefix expanded report suffix"
    state = store.load_state("agent", "artifact-thread", "artifact-run")
    assistant = [
        item
        for item in state["messages"]["items"]
        if item.get("role") == "assistant"
    ][-1]
    assert assistant["content"] == "prefix {{artifact:report}} suffix"


@pytest.mark.asyncio
async def test_artifact_renderer_isolated_for_concurrent_runs_and_watch_snapshot():
    registry = ArtifactRegistry()
    registry.register("ONE", artifact_id="one")
    registry.register("TWO", artifact_id="two")

    async def acall(**kwargs):
        user = next(item for item in kwargs["messages"] if item.get("role") == "user")
        text = "one" if user["content"] == "first" else "two"
        response = ModelStreamResponse(mode="async")
        response.set_response_type("text_generation")
        response.add("{{artifact:")
        response.add(f"{text}" + "}}")
        response.finish()
        return response

    model = type("Model", (), {"model_type": "chat_completion"})()
    model.acall = acall
    store = InMemoryCheckpointStore()
    agent = Agent(
        name="agent",
        model=model,
        checkpoint_store=store,
        extensions=[ArtifactExtension(registry)],
        config={"stream": True},
    )

    async def collect(message, thread, run):
        return [
            event
            async for event in agent.stream_events(
                message,
                scope=ExecutionScope(thread_id=thread, run_id=run),
            )
        ]

    first, second = await __import__("asyncio").gather(
        collect("first", "thread-one", "run-one"),
        collect("second", "thread-two", "run-two"),
    )
    assert next(e for e in first if e.type == EventType.MESSAGE_END).data["content"] == "ONE"
    assert next(e for e in second if e.type == EventType.MESSAGE_END).data["content"] == "TWO"
    async with agent.watch("thread-one") as watcher:
        assert watcher.snapshot.messages.to_chatml()[-1]["content"] == "{{artifact:one}}"


@pytest.mark.asyncio
async def test_wrapped_stream_response_keeps_envelope_and_renders_content():
    registry = ArtifactRegistry()
    registry.register("expanded", artifact_id="item")
    response = ModelStreamResponse(mode="async")
    response.set_response_type("text_generation")
    response.add("{{artifact:item}}")
    response.finish()
    model = type("Model", (), {"model_type": "chat_completion"})()
    model.acall = AsyncMock(return_value=response)
    agent = Agent(
        name="agent",
        model=model,
        extensions=[ArtifactExtension(registry)],
        config={"stream": True, "return_messages": True},
    )
    events = [event async for event in agent.stream_events("question")]
    content = next(event for event in events if event.type == EventType.MESSAGE_END).data[
        "content"
    ]
    assert content["response"] == "expanded"
    assert "messages" in content
