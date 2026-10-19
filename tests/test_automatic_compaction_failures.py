"""Automatic compaction failures do not publish candidate history."""

import asyncio

import pytest

from msgflux.chat_messages import ChatMessages
from msgflux.data.stores import InMemoryCheckpointStore
from msgflux.models.compaction import ContextTokenEstimate, ModelCompaction
from msgflux.models.response import ModelResponse
from msgflux.nn import Agent, CompactionExtension, CompactionPolicy
from msgflux.runtime import ExecutionScope


class _CompactingModel:
    model_type = "chat_completion"
    context_capacity = 100
    provider = "test"
    api_mode = "messages"
    model_id = "test-model"

    def count_context_tokens(self, messages, *, system_prompt=None, tool_catalog=None):
        return ContextTokenEstimate(input_tokens=90, source="heuristic")

    async def acount_context_tokens(
        self, messages, *, system_prompt=None, tool_catalog=None
    ):
        return self.count_context_tokens(
            messages, system_prompt=system_prompt, tool_catalog=tool_catalog
        )

    def compact_context(self, messages, *, system_prompt=None, native=True):
        return ModelCompaction(
            format="messages",
            items=[{"role": "system", "content": "SUMMARY"}],
            provider=self.provider,
            api_mode=self.api_mode,
            model_id=self.model_id,
        )

    async def acompact_context(self, messages, *, system_prompt=None, native=True):
        return self.compact_context(
            messages, system_prompt=system_prompt, native=native
        )

    def __call__(self, **_kwargs):
        return _text_response("ok")

    async def acall(self, **_kwargs):
        return _text_response("ok")


def _text_response(text):
    response = ModelResponse()
    response.set_response_type("text_generation")
    response.add(text)
    response.reasoning = None
    response.metadata = {}
    return response


def _history():
    messages = ChatMessages(thread_id="thread", namespace="agent")
    messages.begin_turn(turn_id="previous")
    messages.add_user("Old question")
    messages.add_assistant("Old answer")
    messages.end_turn()
    return messages


@pytest.mark.asyncio
@pytest.mark.parametrize("async_call", [False, True])
async def test_automatic_compaction_checkpoint_failure_does_not_publish_history(
    async_call,
):
    store = InMemoryCheckpointStore()
    failure = RuntimeError("injected compaction checkpoint failure")
    original_commit = store.commit_state
    failed_once = False

    def fail_compaction_commit(namespace, thread_id, run_id, state, **kwargs):
        nonlocal failed_once
        if not failed_once and any(
            item.get("type") == "compaction" for item in state["messages"]["items"]
        ):
            failed_once = True
            raise failure
        return original_commit(namespace, thread_id, run_id, state, **kwargs)

    store.commit_state = fail_compaction_commit
    messages = _history()
    agent = Agent(
        name="agent",
        model=_CompactingModel(),
        checkpoint_store=store,
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

    try:
        async with agent.watch("thread") as watcher:
            with pytest.raises(RuntimeError) as raised:
                if async_call:
                    await agent.acall(
                        "New question",
                        messages=messages,
                        scope=ExecutionScope(
                            thread_id="thread", namespace="agent", run_id="failed"
                        ),
                    )
                else:
                    agent(
                        "New question",
                        messages=messages,
                        scope=ExecutionScope(
                            thread_id="thread", namespace="agent", run_id="failed"
                        ),
                    )

            assert raised.value is failure
            observed = []
            while not any(event.type == "compaction.end" for event in observed):
                observed.append(await asyncio.wait_for(anext(watcher), timeout=2))

        end = next(event for event in observed if event.type == "compaction.end")
        assert end.data["status"] == "failed"
        assert end.data["error"] == str(failure)
        assert messages.latest_compaction() is None
        failed_state = store.load_state("agent", "thread", "failed")
        assert failed_state["status"] == "failed"
        assert not any(
            item.get("type") == "compaction"
            for item in failed_state["messages"]["items"]
        )

        # A fresh call on the same thread can acquire the operation after failure.
        await agent.acall(
            "Next question",
            messages=ChatMessages(thread_id="thread", namespace="agent"),
            scope=ExecutionScope(thread_id="thread", namespace="agent", run_id="next"),
        )
    finally:
        await agent.aclose()


@pytest.mark.asyncio
async def test_cancelled_automatic_compaction_emits_terminal_event_and_releases_scope(
    monkeypatch,
):
    model = _CompactingModel()
    entered = asyncio.Event()
    original_acompact = model.acompact_context

    async def slow_compaction(*_args, **_kwargs):
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(model, "acompact_context", slow_compaction)
    store = InMemoryCheckpointStore()
    agent = Agent(
        name="agent",
        model=model,
        checkpoint_store=store,
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
    messages = _history()
    scope = ExecutionScope(thread_id="thread", namespace="agent", run_id="cancelled")

    try:
        async with agent.watch("thread") as watcher:
            task = asyncio.create_task(
                agent.acall("New question", messages=messages, scope=scope)
            )
            await asyncio.wait_for(entered.wait(), timeout=2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

            observed = []
            while not any(event.type == "compaction.end" for event in observed):
                observed.append(await asyncio.wait_for(anext(watcher), timeout=2))

        end = next(event for event in observed if event.type == "compaction.end")
        assert end.data["status"] == "failed"
        assert end.data["error"] == "CancelledError"
        assert messages.latest_compaction() is None
        interrupted = store.load_state("agent", "thread", "cancelled")
        assert interrupted["status"] == "interrupted"
        assert not any(
            item.get("type") == "compaction"
            for item in interrupted["messages"]["items"]
        )

        # Cancellation releases per-thread admission even when compaction is active.
        monkeypatch.setattr(model, "acompact_context", original_acompact)
        await agent.acall(
            "Next question",
            messages=ChatMessages(thread_id="thread", namespace="agent"),
            scope=ExecutionScope(thread_id="thread", namespace="agent", run_id="next"),
        )
    finally:
        await agent.aclose()
