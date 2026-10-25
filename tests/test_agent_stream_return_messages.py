"""Direct Agent streams wrapped by ``return_messages`` stay consumable."""

from __future__ import annotations

import asyncio
import threading
from dataclasses import replace
from unittest.mock import AsyncMock, Mock

import pytest

from msgflux.chat_messages import ChatMessages
from msgflux.data.stores import InMemoryCheckpointStore
from msgflux.models.response import ModelStreamResponse
from msgflux.nn import Agent
from msgflux.nn.extensions import AgentExtension
from msgflux.nn.hooks import Hook, OutputContext
from msgflux.runtime.context import ExecutionScope


def _stream(mode):
    stream = ModelStreamResponse(mode=mode)
    stream.set_response_type("text_generation")
    stream.add("raw answer")
    return stream


def _agent(*, asynchronous, transform_output=None):
    model = Mock(model_type="chat_completion")
    model.aclose = AsyncMock()
    hooks = (
        [Hook(event="transform_output", handler=transform_output)]
        if transform_output is not None
        else None
    )
    agent = Agent(
        name="stream-envelope",
        model=model,
        config={"stream": True, "return_messages": True},
        checkpoint_store=InMemoryCheckpointStore(),
        hooks=hooks,
    )
    stream = _stream("async" if asynchronous else "sync")
    if asynchronous:

        async def aforward(**_kwargs):
            return stream

        agent.generator.aforward = aforward
    else:
        agent.generator.forward = Mock(return_value=stream)
    return agent, stream


async def _call(agent, asynchronous, *, messages, scope):
    if asynchronous:
        return await agent.acall("hi", messages=messages, scope=scope)
    return agent("hi", messages=messages, scope=scope)


async def _await_event_finalization(stream):
    finalization = stream._msgflux_event_finalization_future
    assert finalization is not None
    if isinstance(finalization, asyncio.Future):
        await asyncio.wait_for(asyncio.shield(finalization), timeout=5)
    else:
        await asyncio.wait_for(asyncio.wrap_future(finalization), timeout=5)


def _assistant_content(messages):
    assistant = [
        item for item in messages.to_chatml() if item.get("role") == "assistant"
    ]
    assert assistant
    return assistant[-1]["content"]


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("with_transform", [False, True], ids=["plain", "transform"])
async def test_direct_stream_return_messages_keeps_response_stream(
    asynchronous, with_transform
):
    transformed_inputs = []

    def transform_output(context: OutputContext):
        transformed_inputs.append(context.output)
        return replace(context, output=f"transformed:{context.output}")

    agent, stream = _agent(
        asynchronous=asynchronous,
        transform_output=transform_output if with_transform else None,
    )
    history = ChatMessages()
    scope = ExecutionScope(
        namespace="stream-envelope",
        thread_id=f"thread-{asynchronous}-{with_transform}",
        run_id=f"run-{asynchronous}-{with_transform}",
    )

    try:
        result = await _call(agent, asynchronous, messages=history, scope=scope)

        assert isinstance(result, dict)
        assert result["messages"] is history
        assert result["response"] is stream
        if with_transform:
            # A wrapped stream is not a settled value: the direct call must not
            # pass the presentation envelope through transform_output.
            assert transformed_inputs == []

        stream.finish()
        await _await_event_finalization(stream)

        # The detached event finalizer may publish a transformed event payload,
        # but it must not replace the stream handle returned to the caller.
        assert result["response"] is stream
        chunks = [chunk async for chunk in result["response"].consume()]
        assert chunks == ["raw answer"]
        assert _assistant_content(result["messages"]) == "raw answer"
        persisted = agent.checkpoint_store.load_state(
            scope.namespace, scope.thread_id, scope.run_id
        )
        assert persisted["status"] == "completed"
        assert [
            item["content"]
            for item in persisted["messages"]["items"]
            if item.get("type") == "message" and item.get("role") == "assistant"
        ] == ["raw answer"]
        if with_transform:
            assert transformed_inputs == ["raw answer"]
    finally:
        stream.finish()
        await agent.aclose()


class _FailingStreamTransformer:
    def __init__(self, primary, seen, raised):
        self._primary = primary
        self._seen = seen
        self._raised = raised

    def feed(self, _chunk):
        self._raised.append(self._primary)
        self._seen.set()
        raise self._primary

    def finish(self):
        return ""


class _FailingTransformExtension(AgentExtension):
    """Public extension whose incremental stream transformer fails."""

    def __init__(self, primary, seen, raised):
        super().__init__("failing-transform")
        self._primary = primary
        self._seen = seen
        self._raised = raised

    def hooks(self):
        return (Hook(event="transform_output", handler=lambda context: context),)

    def create_output_transformer(self):
        return _FailingStreamTransformer(self._primary, self._seen, self._raised)


class _TerminalCheckpointStore(InMemoryCheckpointStore):
    """In-memory store that signals when a terminal status is persisted."""

    def __init__(self):
        super().__init__()
        self.terminal = threading.Event()

    def _note(self, state):
        if isinstance(state, dict) and state.get("status") in {
            "completed",
            "failed",
            "interrupted",
        }:
            self.terminal.set()

    def save_state(self, *args, **kwargs):
        self._note(args[3] if len(args) >= 4 else kwargs.get("state"))
        return super().save_state(*args, **kwargs)

    def commit_state(self, *args, **kwargs):
        self._note(args[3] if len(args) >= 4 else kwargs.get("state"))
        return super().commit_state(*args, **kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_direct_stream_surfaces_detached_event_failure(asynchronous):
    primary = RuntimeError("incremental transform failed")
    seen = threading.Event()
    raised = []
    store = _TerminalCheckpointStore()
    model = Mock(model_type="chat_completion")
    model.aclose = AsyncMock()
    agent = Agent(
        name="stream-event-failure",
        model=model,
        config={"stream": True, "return_messages": True},
        checkpoint_store=store,
        extensions=[_FailingTransformExtension(primary, seen, raised)],
    )
    stream = ModelStreamResponse(mode="async" if asynchronous else "sync")
    stream.set_response_type("text_generation")
    stream.add("hello")
    if asynchronous:

        async def aforward(**_kwargs):
            return stream

        agent.generator.aforward = aforward
    else:
        agent.generator.forward = Mock(return_value=stream)
    history = ChatMessages()
    scope = ExecutionScope(
        namespace="stream-event-failure",
        thread_id=f"thread-{asynchronous}",
        run_id=f"run-{asynchronous}",
    )

    try:
        if asynchronous:
            result = await agent.acall("hi", messages=history, scope=scope)
        else:
            result = agent("hi", messages=history, scope=scope)

        assert result["response"] is stream
        assert await asyncio.wait_for(asyncio.to_thread(seen.wait, 5), timeout=6)
        assert await asyncio.wait_for(
            asyncio.to_thread(store.terminal.wait, 5), timeout=6
        )
        assert raised == [primary]

        chunks = []
        with pytest.raises(RuntimeError) as caught:
            async for chunk in result["response"].consume():
                chunks.append(chunk)

        assert caught.value is primary
        assert chunks == ["hello"]
        persisted = store.load_state(scope.namespace, scope.thread_id, scope.run_id)
        assert persisted["status"] == "interrupted"
        assert [
            item["content"]
            for item in persisted["messages"]["items"]
            if item.get("type") == "message" and item.get("role") == "assistant"
        ] == ["hello"]
        assert history.get_active_turn() is None
    finally:
        stream.finish()
        await agent.aclose()
