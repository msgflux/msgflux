"""Automatic compaction hooks inspect a snapshot of canonical history."""

import pytest

from msgflux.chat_messages import ChatMessages
from msgflux.data.stores import InMemoryCheckpointStore
from msgflux.exceptions import TaskInterruptRequestedError
from msgflux.models.compaction import ContextTokenEstimate, ModelCompaction
from msgflux.models.response import ModelResponse
from msgflux.nn import Agent, CompactionExtension, CompactionPolicy
from msgflux.nn.hooks import Hook
from msgflux.runtime import ExecutionScope

_HOOK_MARKER = "must-not-enter-canonical-history"


class _CompactingModel:
    model_type = "chat_completion"
    context_capacity = 100
    provider = "test"
    api_mode = "messages"
    model_id = "test-model"

    def __init__(self):
        self.compaction_calls = 0

    def count_context_tokens(self, messages, *, system_prompt=None, tool_catalog=None):
        return ContextTokenEstimate(input_tokens=90, source="heuristic")

    async def acount_context_tokens(
        self, messages, *, system_prompt=None, tool_catalog=None
    ):
        return self.count_context_tokens(
            messages, system_prompt=system_prompt, tool_catalog=tool_catalog
        )

    def compact_context(self, messages, *, system_prompt=None, native=True):
        self.compaction_calls += 1
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
async def test_automatic_compaction_hook_mutation_does_not_reach_history(async_call):
    store = InMemoryCheckpointStore()
    model = _CompactingModel()
    messages = _history()
    failure = TaskInterruptRequestedError("task")

    def mutate_then_interrupt(context):
        context.messages.add_user(_HOOK_MARKER)
        raise failure

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
        hooks=[Hook(event="before_compaction", handler=mutate_then_interrupt)],
    )

    try:
        with pytest.raises(TaskInterruptRequestedError) as raised:
            if async_call:
                await agent.acall(
                    "New question",
                    messages=messages,
                    scope=ExecutionScope(
                        thread_id="thread", namespace="agent", run_id="interrupted"
                    ),
                )
            else:
                agent(
                    "New question",
                    messages=messages,
                    scope=ExecutionScope(
                        thread_id="thread", namespace="agent", run_id="interrupted"
                    ),
                )

        assert raised.value is failure
        assert all(item.get("content") != _HOOK_MARKER for item in messages.to_items())
        interrupted = store.load_state("agent", "thread", "interrupted")
        assert interrupted["status"] == "interrupted"
        assert all(
            item.get("content") != _HOOK_MARKER
            for item in interrupted["messages"]["items"]
        )
        assert model.compaction_calls == 0
    finally:
        await agent.aclose()
