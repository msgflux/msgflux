"""Tests for public Module event streams."""

import asyncio
import threading
from dataclasses import replace

import pytest
from unittest.mock import AsyncMock, Mock

from msgflux.chat_messages import ChatMessages
from msgflux.data.stores import InMemoryCheckpointStore
from msgflux.models.response import ModelResponse, ModelStreamResponse
from msgflux.models.compaction import ContextTokenEstimate, ModelCompaction
from msgflux.models.tool_call_agg import ToolCallAggregator
from msgflux.nn.hooks import Hook
from msgflux.nn.hooks.events import BeforeRun
from msgflux.nn.modules.agent import Agent
from msgflux.nn.extensions import CompactionExtension, CompactionPolicy
from msgflux.nn.modules.module import Module
from msgflux.nn.modules.tool import ToolLibrary
from msgflux.runtime.events import EventType, emit_event, event_source
from msgflux.runtime.context import ExecutionScope
from msgflux.runtime.event_hub import get_event_hub
from msgflux.tools.builtin import AgentTool
from msgflux.tools.config import tool_config


class EchoModule(Module):
    def forward(self, value):
        return value.upper()

    async def aforward(self, value):
        return value.upper()


class StreamingModule(Module):
    def forward(self):
        response = ModelStreamResponse()
        response.set_response_type("text_generation")
        response.add_reasoning("think")
        response.add("hello")
        response.add(" world")
        response.finish()
        return response


class FailingModule(Module):
    def forward(self):
        raise RuntimeError("boom")

    async def aforward(self):
        raise RuntimeError("boom")


class ParallelEventModule(Module):
    async def aforward(self):
        def emit_many(name):
            with event_source(name, "worker"):
                for index in range(50):
                    emit_event(EventType.TOOL_UPDATE, {"worker_index": index})

        await asyncio.gather(
            asyncio.to_thread(emit_many, "left"),
            asyncio.to_thread(emit_many, "right"),
        )
        return "done"


def make_agent(*, hooks=None):
    model = Mock()
    model.model_type = "chat_completion"
    response = ModelResponse()
    response.set_response_type("text_generation")
    response.add("hello")
    model.return_value = response
    model.acall = AsyncMock(return_value=response)
    return Agent(name="agent", model=model, hooks=hooks), model


@pytest.mark.asyncio
async def test_stream_events_yields_ordered_message_lifecycle():
    events = [event async for event in EchoModule().stream_events("hello")]

    assert [event.type for event in events] == [
        EventType.RUN_START,
        EventType.TURN_START,
        EventType.MESSAGE_START,
        EventType.MESSAGE_END,
        EventType.TURN_END,
        EventType.RUN_END,
    ]
    assert events[-3].data == {"content": "HELLO"}
    assert events[-2].data == {}
    assert events[-1].data == {"outcome": "completed"}
    assert set(vars(events[0])) == {
        "type",
        "timestamp",
        "data",
        "run_id",
        "source_path",
    }


@pytest.mark.asyncio
async def test_stream_events_owns_and_consumes_model_stream():
    events = [event async for event in StreamingModule().stream_events()]

    content = [
        event.data["delta"] for event in events if event.type == EventType.MESSAGE_DELTA
    ]
    reasoning = [
        event.data["delta"]
        for event in events
        if event.type == EventType.REASONING_DELTA
    ]

    assert content == ["hello", " world"]
    assert reasoning == ["think"]
    message_end = next(event for event in events if event.type == EventType.MESSAGE_END)
    assert message_end.data["content"] == "hello world"


@pytest.mark.asyncio
async def test_stream_events_preserve_reasoning_summary_and_output_order():
    class OrderedStreamingModule(Module):
        async def aforward(self):
            response = ModelStreamResponse(mode="async")
            response.set_response_type("text_generation")
            response.add_reasoning("private")
            response.add_reasoning_summary("safe summary")
            response.add("visible answer")
            response.finish()
            return response

    events = [event async for event in OrderedStreamingModule().stream_events()]
    streamed = [
        (event.type, event.data["delta"])
        for event in events
        if event.type
        in {
            EventType.REASONING_DELTA,
            EventType.REASONING_SUMMARY_DELTA,
            EventType.MESSAGE_DELTA,
        }
    ]

    assert streamed == [
        (EventType.REASONING_DELTA, "private"),
        (EventType.REASONING_SUMMARY_DELTA, "safe summary"),
        (EventType.MESSAGE_DELTA, "visible answer"),
    ]


@pytest.mark.asyncio
async def test_parallel_internal_events_preserve_delivery_and_source_path():
    events = [event async for event in ParallelEventModule().stream_events()]
    updates = [event for event in events if event.type == EventType.TOOL_UPDATE]

    assert len(updates) == 100
    assert {event.source_path[-1] for event in updates} == {
        "worker:left",
        "worker:right",
    }
    for worker in ("left", "right"):
        assert [
            event.data["worker_index"]
            for event in updates
            if event.source_path[-1] == f"worker:{worker}"
        ] == list(range(50))


@pytest.mark.asyncio
async def test_stream_events_reraises_execution_error_after_error_event():
    seen = []

    with pytest.raises(RuntimeError, match="boom"):
        async for event in FailingModule().stream_events():
            seen.append(event)

    assert seen[-1].type == EventType.RUN_ERROR
    assert seen[-1].data["error"] == "boom"
    assert seen[-1].source_path == ("module:FailingModule",)


@pytest.mark.asyncio
async def test_agent_run_start_carries_execution_context_and_model_boundaries():
    agent, _ = make_agent()

    events = [event async for event in agent.stream_events("hi")]

    assert all(event.run_id for event in events)
    run_start = events[0]
    assert run_start.type == EventType.RUN_START
    assert run_start.data["thread_id"]
    assert run_start.data["namespace"] == "agent"
    assert run_start.data["root_run_id"] == run_start.run_id
    assert "parent_run_id" not in run_start.data
    assert EventType.MODEL_REQUEST in [event.type for event in events]
    assert EventType.MODEL_RESPONSE in [event.type for event in events]


@pytest.mark.asyncio
async def test_agent_stream_events_reports_compaction_without_summary_content():
    class CompactingModel:
        model_type = "chat_completion"
        context_capacity = 100
        provider = "test"
        api_mode = "messages"
        model_id = "test-model"

        async def acount_context_tokens(self, messages, **_kwargs):
            return ContextTokenEstimate(input_tokens=90, source="heuristic")

        async def acompact_context(self, messages, **_kwargs):
            return ModelCompaction(
                format="messages",
                items=[{"role": "system", "content": "SECRET SUMMARY"}],
                provider=self.provider,
                api_mode=self.api_mode,
                model_id=self.model_id,
                usage={"input_tokens": 40, "output_tokens": 5},
            )

        async def acall(self, **_kwargs):
            response = ModelResponse()
            response.set_response_type("text_generation")
            response.add("done")
            response.metadata = {}
            return response

    messages = ChatMessages(thread_id="thread_compact", namespace="agent")
    messages.begin_turn(turn_id="old_turn")
    messages.add_user("old question")
    messages.add_assistant("old answer")
    messages.end_turn()
    agent = Agent(
        name="agent",
        model=CompactingModel(),
        extensions=[
            CompactionExtension(
                CompactionPolicy(
                    trigger_ratio=0.8,
                    reserved_output_tokens=0,
                    safety_margin_tokens=0,
                )
            )
        ],
    )

    events = [
        event
        async for event in agent.stream_events(
            "new question",
            messages=messages,
            scope=ExecutionScope(
                thread_id="thread_compact",
                namespace="agent",
                run_id="new_turn",
            ),
        )
    ]
    event_types = [event.type for event in events]
    start = next(event for event in events if event.type == EventType.COMPACTION_START)
    end = next(event for event in events if event.type == EventType.COMPACTION_END)

    assert (
        event_types.index(EventType.COMPACTION_START)
        < event_types.index(EventType.COMPACTION_END)
        < event_types.index(EventType.MODEL_REQUEST)
    )
    assert start.data["input_tokens_before"] == 90
    assert end.data["status"] == "completed"
    assert end.data["usage"] == {"input_tokens": 40, "output_tokens": 5}
    assert "SECRET SUMMARY" not in repr(start.data)
    assert "SECRET SUMMARY" not in repr(end.data)


@pytest.mark.asyncio
async def test_model_response_event_exposes_compact_usage_and_timing():
    model = Mock()
    model.model_type = "chat_completion"
    response = ModelResponse()
    response.set_response_type("text_generation")
    response.add("hello")
    response.reasoning = "private reasoning"
    response.reasoning_summary = "safe summary"
    response.metadata = {
        "usage": {
            "input_tokens": 100,
            "output_tokens": 20,
            "total_tokens": 120,
            "cache_hit_percentage": 25.0,
            "input_tokens_details": {"cached_tokens": 25},
            "raw": {"provider_field": "discarded"},
        },
        "timing": {
            "source": "provider",
            "latency_ms": 1250.5,
        },
    }
    model.acall = AsyncMock(return_value=response)
    agent = Agent(name="agent", model=model)

    events = [event async for event in agent.stream_events("hi")]

    model_response = next(
        event for event in events if event.type == EventType.MODEL_RESPONSE
    )
    reasoning = next(
        event for event in events if event.type == EventType.REASONING_DELTA
    )
    summary = next(
        event for event in events if event.type == EventType.REASONING_SUMMARY_DELTA
    )
    assert reasoning.data == {"delta": "private reasoning"}
    assert summary.data == {"delta": "safe summary"}
    assert (
        events.index(reasoning) < events.index(summary) < events.index(model_response)
    )
    assert model_response.data == {
        "response_type": "text_generation",
        "usage": {
            "input_tokens": 100,
            "output_tokens": 20,
            "cached_input_tokens": 25,
        },
        "timing": {
            "source": "provider",
            "latency_ms": 1250.5,
        },
    }


@pytest.mark.asyncio
async def test_streamed_model_response_event_waits_for_terminal_metrics():
    model = Mock()
    model.model_type = "chat_completion"

    async def stream_call(**_kwargs):
        response = ModelStreamResponse(mode="async")
        response.set_response_type("text_generation")

        async def produce():
            await asyncio.sleep(0.01)
            response.add("hello")
            response.add(" world")
            response.set_metadata(
                {
                    "usage": {
                        "input_tokens": 12,
                        "output_tokens": 2,
                        "input_tokens_details": {"cached_tokens": 8},
                    },
                    "timing": {
                        "source": "provider",
                        "latency_ms": 900.0,
                        "ttft_ms": 300.0,
                    },
                }
            )
            response.finish()

        asyncio.create_task(produce())
        return response

    model.acall = AsyncMock(side_effect=stream_call)
    agent = Agent(name="streaming_agent", model=model, config={"stream": True})

    events = [event async for event in agent.stream_events("hi")]

    model_responses = [
        event for event in events if event.type == EventType.MODEL_RESPONSE
    ]
    assert len(model_responses) == 1
    assert model_responses[0].data == {
        "response_type": "text_generation",
        "usage": {
            "input_tokens": 12,
            "output_tokens": 2,
            "cached_input_tokens": 8,
        },
        "timing": {
            "source": "provider",
            "latency_ms": 900.0,
            "ttft_ms": 300.0,
        },
    }
    assert model_responses[0].source_path == ("agent:streaming_agent",)
    assert events.index(model_responses[0]) > max(
        index
        for index, event in enumerate(events)
        if event.type == EventType.MESSAGE_DELTA
    )
    assert events.index(model_responses[0]) < next(
        index
        for index, event in enumerate(events)
        if event.type == EventType.MESSAGE_END
    )
    assert events.index(model_responses[0]) < next(
        index for index, event in enumerate(events) if event.type == EventType.RUN_END
    )


@pytest.mark.asyncio
async def test_watch_reconnects_to_normal_agent_stream_with_live_snapshot():
    thread_id = "thread_watch_reconnect"
    release = asyncio.Event()
    model = Mock()
    model.model_type = "chat_completion"

    async def stream_call(**_kwargs):
        response = ModelStreamResponse(mode="async")
        response.set_response_type("text_generation")

        async def produce():
            response.add_reasoning_summary("checking inventory")
            response.add("hel")
            await release.wait()
            response.add("lo")
            response.finish()

        asyncio.create_task(produce())
        return response

    model.acall = AsyncMock(side_effect=stream_call)
    agent = Agent(name="watch_agent", model=model, config={"stream": True})
    scope = ExecutionScope(thread_id=thread_id)

    first_events = []
    async with agent.watch(thread_id) as first_watcher:
        call = asyncio.create_task(agent.acall("hi", scope=scope))
        while True:
            event = await asyncio.wait_for(first_watcher.__anext__(), timeout=1)
            first_events.append(event)
            if event.type == EventType.MESSAGE_DELTA:
                break

    response = await call
    async with agent.watch(thread_id) as reconnected:
        assert reconnected.snapshot.streaming_message == "hel"
        assert reconnected.snapshot.active_run.reasoning_summary == "checking inventory"
        release.set()
        reconnected_events = []
        while True:
            event = await asyncio.wait_for(reconnected.__anext__(), timeout=1)
            reconnected_events.append(event)
            if event.type == EventType.RUN_END:
                break

    assert EventType.RUN_START in [event.type for event in first_events]
    assert [
        event.data["delta"]
        for event in reconnected_events
        if event.type == EventType.MESSAGE_DELTA
    ] == ["lo"]
    assert [(event.type, event.data) async for event in response.consume_events()] == [
        ("reasoning_summary.delta", "checking inventory"),
        ("output.delta", "hel"),
        ("output.delta", "lo"),
    ]
    assert thread_id not in get_event_hub()._threads


@pytest.mark.asyncio
async def test_stream_events_and_watch_receive_one_copy_of_each_event():
    agent, _ = make_agent()
    thread_id = "thread_watch_stream_events"
    scope = ExecutionScope(thread_id=thread_id)

    async with agent.watch(thread_id) as watcher:
        direct_task = asyncio.create_task(
            _collect_events(agent.stream_events("hi", scope=scope))
        )
        watched = []
        while True:
            event = await asyncio.wait_for(watcher.__anext__(), timeout=1)
            watched.append(event)
            if event.type == EventType.RUN_END:
                break
        direct = await direct_task

    assert [event.type for event in watched] == [event.type for event in direct]


@pytest.mark.asyncio
async def test_after_run_end_observer_failure_keeps_completed_stream_event():
    def fail_after(_ctx):
        raise RuntimeError("observer failed")

    agent, _ = make_agent(hooks=[Hook(event="after_run_end", handler=fail_after)])

    with pytest.warns(RuntimeWarning, match="after_run_end hook failed"):
        events = [event async for event in agent.stream_events("hi")]

    assert events[-1].type == EventType.RUN_END
    assert events[-1].data["outcome"] == "completed"
    assert EventType.RUN_ERROR not in [event.type for event in events]


@pytest.mark.asyncio
async def test_watch_snapshot_loads_durable_messages_after_run_completion():
    store = InMemoryCheckpointStore()
    agent, _ = make_agent()
    agent.checkpoint_store = store
    thread_id = "thread_watch_checkpoint"

    await agent.acall("persist me", scope=ExecutionScope(thread_id=thread_id))

    async with agent.watch(thread_id) as watcher:
        snapshot = watcher.snapshot

    assert snapshot.active_runs == ()
    assert isinstance(snapshot.messages, ChatMessages)
    assert [
        item["content"] for item in snapshot.messages if item.get("type") == "message"
    ] == ["persist me", "hello"]


@pytest.mark.asyncio
async def test_watch_keeps_background_task_visible_after_root_run_ends():
    started = threading.Event()
    release = threading.Event()

    @tool_config(background=True)
    def long_job(value: int) -> int:
        """Double a value after an external release."""
        started.set()
        release.wait(timeout=2)
        return value * 2

    calls = ToolCallAggregator()
    calls.process(0, "call_1", "long_job", '{"value":21}')
    tool_response = ModelResponse()
    tool_response.set_response_type("tool_call")
    tool_response.add(calls)
    final_response = ModelResponse()
    final_response.set_response_type("text_generation")
    final_response.add("dispatched")

    model = Mock()
    model.model_type = "chat_completion"
    model.acall = AsyncMock(side_effect=[tool_response, final_response])
    agent = Agent(name="background_agent", model=model, tools=[long_job])
    thread_id = "thread_watch_background"

    assert (
        await agent.acall("start the job", scope=ExecutionScope(thread_id=thread_id))
        == "dispatched"
    )
    assert await asyncio.to_thread(started.wait, 1)

    async with agent.watch(thread_id) as watcher:
        assert len(watcher.snapshot.background_tasks) == 1
        task = watcher.snapshot.background_tasks[0]
        assert task.tool_name == "long_job"
        assert task.status == "running"
        release.set()
        while True:
            event = await asyncio.wait_for(watcher.__anext__(), timeout=1)
            if event.type == EventType.TASK_END:
                break

    assert event.data["status"] == "completed"


async def _collect_events(stream):
    return [event async for event in stream]


def test_agent_lifecycle_hooks_transform_canonical_response():
    def add_suffix(ctx):
        ctx.response.data += "!"
        return ctx

    agent, _ = make_agent(hooks=[Hook(event="after_response", handler=add_suffix)])

    assert agent("hi") == "hello!"


@pytest.mark.asyncio
async def test_tool_events_wrap_validated_foreground_execution():
    def double(value: int) -> int:
        """Double a value."""
        return value * 2

    library = ToolLibrary(name="math", tools=[double])

    events = [
        event
        async for event in library.stream_events(
            tool_callings=[("call_1", "double", {"value": 3})]
        )
    ]
    tool_events = [
        event
        for event in events
        if event.type in {EventType.TOOL_START, EventType.TOOL_END}
    ]

    assert [event.type for event in tool_events] == [
        EventType.TOOL_START,
        EventType.TOOL_END,
    ]
    assert tool_events[0].data["tool_call_id"] == "call_1"
    assert tool_events[1].data["result"] == 6


@pytest.mark.asyncio
async def test_blocked_tool_emits_one_terminal_policy_event():
    def dangerous(command: str) -> str:
        """Run a dangerous command."""
        raise AssertionError("blocked tool must not execute")

    library = ToolLibrary(name="shell", tools=[dangerous])
    Hook(
        event="before_tool",
        handler=lambda event: replace(event, block="Command denied."),
    ).register(library)

    events = [
        event
        async for event in library.stream_events(
            tool_callings=[("call_1", "dangerous", {"command": "rm -rf ~"})]
        )
    ]
    blocked = [event for event in events if event.type == EventType.TOOL_BLOCKED]

    assert len(blocked) == 1
    assert blocked[0].data == {
        "tool_call_id": "call_1",
        "tool_name": "dangerous",
        "arguments": {"command": "rm -rf ~"},
        "reason": "Command denied.",
    }
    assert blocked[0].source_path[-1] == "tool:dangerous"
    assert not [
        event
        for event in events
        if event.type in {EventType.TOOL_START, EventType.TOOL_END}
    ]


def test_transform_output_changes_return_without_changing_canonical_history():
    agent, _ = make_agent(
        hooks=[
            Hook(
                event="transform_output",
                handler=lambda ctx: replace(
                    ctx,
                    output=ctx.output.replace("hello", "expanded report"),
                ),
            )
        ]
    )
    history = ChatMessages()

    output = agent("hi", messages=history)
    assistant_messages = [item for item in history if item.get("role") == "assistant"]

    assert output == "expanded report"
    assert assistant_messages[-1]["content"] == "hello"


@pytest.mark.asyncio
async def test_transform_output_buffers_stream_content_but_keeps_reasoning_live():
    module = StreamingModule()
    Hook(
        event="transform_output",
        handler=lambda output: output.upper(),
    ).register(module)

    events = [event async for event in module.stream_events()]

    assert not [event for event in events if event.type == EventType.MESSAGE_DELTA]
    assert [
        event.data["delta"]
        for event in events
        if event.type == EventType.REASONING_DELTA
    ] == ["think"]
    message_start = next(
        event for event in events if event.type == EventType.MESSAGE_START
    )
    message_end = next(event for event in events if event.type == EventType.MESSAGE_END)
    assert message_start.data["buffered"] is True
    assert message_end.data["content"] == "HELLO WORLD"


@pytest.mark.asyncio
async def test_async_transform_output_runs_after_stream_accumulation():
    async def transform(output):
        return f"[{output}]"

    module = StreamingModule()
    Hook(event="transform_output", handler=transform).register(module)

    events = [event async for event in module.stream_events()]

    message_end = next(event for event in events if event.type == EventType.MESSAGE_END)
    assert message_end.data["content"] == "[hello world]"


@pytest.mark.asyncio
async def test_foreground_agent_tool_events_include_nested_source_identity():
    root_model = Mock()
    root_model.model_type = "chat_completion"
    tool_calls = ToolCallAggregator()
    tool_calls.process(
        0,
        "call_reviewer",
        "agent",
        '{"name":"reviewer","message":"Review this"}',
    )
    tool_response = ModelResponse()
    tool_response.set_response_type("tool_call")
    tool_response.data = tool_calls
    tool_response.metadata = {"usage": {"input_tokens": 10, "output_tokens": 2}}
    final_response = ModelResponse()
    final_response.set_response_type("text_generation")
    final_response.add("root complete")
    final_response.metadata = {"usage": {"input_tokens": 20, "output_tokens": 3}}
    root_model.acall = AsyncMock(side_effect=[tool_response, final_response])

    child_model = Mock()
    child_model.model_type = "chat_completion"
    child_response = ModelResponse()
    child_response.set_response_type("text_generation")
    child_response.add("review complete")
    child_response.metadata = {"usage": {"input_tokens": 30, "output_tokens": 4}}
    child_model.acall = AsyncMock(return_value=child_response)

    reviewer = Agent(name="reviewer", model=child_model)
    root = Agent(
        name="root",
        model=root_model,
        tools=[AgentTool(), reviewer],
    )

    events = [event async for event in root.stream_events("Delegate the review")]

    root_start = next(
        event
        for event in events
        if event.type == EventType.RUN_START and event.source_path[-1] == "agent:root"
    )
    nested_start = next(
        event
        for event in events
        if event.type == EventType.RUN_START
        and event.source_path[-1] == "agent:reviewer"
    )
    child_request = next(
        event
        for event in events
        if event.type == EventType.MODEL_REQUEST
        and event.source_path[-1] == "agent:reviewer"
    )
    assert nested_start.source_path[-1] == "agent:reviewer"
    assert nested_start.data["namespace"] == "reviewer"
    assert nested_start.run_id != root_start.run_id
    assert nested_start.run_id == f"{root_start.run_id}:call_reviewer:reviewer"
    assert nested_start.data["root_run_id"] == root_start.run_id
    assert nested_start.data["parent_run_id"] == root_start.run_id
    assert "agent:root" in child_request.source_path
    model_responses = [
        event for event in events if event.type == EventType.MODEL_RESPONSE
    ]
    assert [event.data["usage"]["input_tokens"] for event in model_responses] == [
        10,
        30,
        20,
    ]
    assert [event.source_path[-1] for event in model_responses] == [
        "agent:root",
        "agent:reviewer",
        "agent:root",
    ]


@pytest.mark.asyncio
async def test_agent_streams_reasoning_from_intermediate_tool_model_response():
    def multiply(left: int, right: int) -> int:
        """Multiply two integers."""
        return left * right

    tool_calls = ToolCallAggregator()
    tool_calls.process(
        0,
        "call_multiply",
        "multiply",
        '{"left":27,"right":43}',
    )
    tool_response = ModelStreamResponse(mode="async")
    tool_response.set_response_type("tool_call")
    tool_response.add_reasoning_summary("select multiplication tool")
    tool_response.data = tool_calls
    tool_response.metadata = {"usage": {"input_tokens": 10, "output_tokens": 2}}
    tool_response.finish()

    final_response = ModelStreamResponse(mode="async")
    final_response.set_response_type("text_generation")
    final_response.add_reasoning_summary("format final answer")
    final_response.add("1161")
    final_response.metadata = {"usage": {"input_tokens": 15, "output_tokens": 3}}
    final_response.finish()

    model = Mock()
    model.model_type = "chat_completion"
    model.acall = AsyncMock(side_effect=[tool_response, final_response])
    agent = Agent(
        name="reasoning_tool_agent",
        model=model,
        tools=[multiply],
        config={"stream": True},
    )

    events = [event async for event in agent.stream_events("What is 27 * 43?")]
    summaries = [
        event for event in events if event.type == EventType.REASONING_SUMMARY_DELTA
    ]
    model_responses = [
        event for event in events if event.type == EventType.MODEL_RESPONSE
    ]
    tool_start = next(event for event in events if event.type == EventType.TOOL_START)

    assert [event.data["delta"] for event in summaries] == [
        "select multiplication tool",
        "format final answer",
    ]
    assert len(model_responses) == 2
    assert events.index(summaries[0]) < events.index(model_responses[0])
    assert events.index(model_responses[0]) < events.index(tool_start)
    assert events.index(summaries[1]) < events.index(model_responses[1])


def test_before_run_can_replace_fresh_agent_input():
    agent, model = make_agent(
        hooks=[
            Hook(
                event="before_run",
                handler=lambda event: replace(event, message="replacement"),
            )
        ]
    )

    agent("original")

    sent_messages = model.call_args.kwargs["messages"]
    assert sent_messages[-1]["content"] == "replacement"


def test_before_run_receives_typed_payload():
    received = []

    def capture(event):
        received.append(event)

    agent, _ = make_agent(hooks=[Hook(event="before_run", handler=capture)])

    agent("hello")

    assert isinstance(received[0], BeforeRun)


def test_before_and_after_tool_hooks_modify_validated_tool_call():
    def double(value: int) -> int:
        """Double a value."""
        return value * 2

    owner = EchoModule()
    Hook(
        event="before_tool",
        handler=lambda event: replace(event, arguments={"value": 4}),
    ).register(owner)
    Hook(
        event="after_tool",
        handler=lambda event: replace(event, result=event.result + 1),
    ).register(owner)
    library = ToolLibrary(name="math", tools=[double])
    library.set_lifecycle_owner(owner)

    response = library(tool_callings=[("call_1", "double", {"value": 3})])

    assert response.tool_calls[0].parameters == {"value": 4}
    assert response.tool_calls[0].result == 9


def test_before_tool_preserves_injected_tool_context():
    seen = {}

    def inspect(value: int, messages: list) -> int:
        """Inspect a value and the inherited conversation."""
        seen["messages"] = messages
        return value

    inspect.tool_config = {"inject_messages": True}
    owner = EchoModule()
    Hook(
        event="before_tool",
        handler=lambda event: replace(event, arguments={"value": 7}),
    ).register(owner)
    library = ToolLibrary(name="inspection", tools=[inspect])
    library.set_lifecycle_owner(owner)
    messages = [{"role": "user", "content": "context"}]

    response = library(
        tool_callings=[("call_1", "inspect", {"value": 3})],
        messages=messages,
    )

    assert response.tool_calls[0].result == 7
    assert seen["messages"] is messages


def test_before_tool_hook_failure_blocks_execution():
    calls = []

    def double(value: int) -> int:
        """Double a value."""
        calls.append(value)
        return value * 2

    def fail(_event):
        raise RuntimeError("policy unavailable")

    owner = EchoModule()
    Hook(event="before_tool", handler=fail).register(owner)
    library = ToolLibrary(name="math", tools=[double])
    library.set_lifecycle_owner(owner)

    response = library(tool_callings=[("call_1", "double", {"value": 3})])

    assert calls == []
    assert "failed closed" in response.tool_calls[0].error
