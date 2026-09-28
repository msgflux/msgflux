"""Opt-in OpenAI cache check for conversation reasoning updates."""

import os

import pytest

import msgflux as mf
from msgflux.chat_messages import ChatMessages
from msgflux.data.stores import InMemoryCheckpointStore
from msgflux.nn import Agent
from msgflux.runtime import ExecutionScope


@pytest.mark.skipif(
    os.getenv("MSGFLUX_LIVE_OPENAI_CACHE") != "1",
    reason="set MSGFLUX_LIVE_OPENAI_CACHE=1 to run the live cache check",
)
def test_gpt6_reasoning_update_preserves_cached_prefix():
    mf.load_dotenv(os.getenv("MSGFLUX_TEST_DOTENV", ".env"))
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY is unavailable")

    model = mf.Model.chat_completion(
        "openai/gpt-6-astra",
        reasoning_effort="low",
        max_tokens=256,
        return_reasoning=False,
        reasoning_in_tool_call=False,
        extra_body={"prompt_cache_options": {"mode": "implicit"}},
    )
    store = InMemoryCheckpointStore()
    agent = Agent(
        name="reasoning_cache_probe",
        model=model,
        system_prompt=(
            "You are auditing a deterministic queue worker. Preserve all stated "
            "facts, answer briefly, and do not call tools. The worker reads a "
            "job, validates its revision, records a claim, applies the change, "
            "and records a completion. Each stage must be idempotent. "
        )
        * 70,
        checkpoint_store=store,
    )
    history = ChatMessages()
    first_scope = ExecutionScope(
        namespace="reasoning_cache_probe", thread_id="cache-thread", run_id="first"
    )
    agent("Name the first stage.", messages=history, scope=first_scope)
    agent.agent_inbox.set_reasoning_effort("high")
    second_scope = ExecutionScope(
        namespace="reasoning_cache_probe", thread_id="cache-thread", run_id="second"
    )
    agent(
        "What must be checked before applying a change?",
        messages=history,
        scope=second_scope,
    )

    state = store.load_state("reasoning_cache_probe", "cache-thread", "second")
    items = state["messages"]["items"]
    assert any(item.get("type") == "model_configuration" for item in items)
    audited = [
        item["metadata"]
        for item in items
        if item.get("role") == "assistant"
        and isinstance(item.get("metadata"), dict)
        and "model" in item["metadata"]
    ]
    assert audited[-1]["model"]["reasoning_effort"] == "high"
    assert audited[-1]["model"]["provider"] == "openai"
    assert audited[-1]["model"]["model_id"] == "gpt-6-astra"
    assert audited[-1]["usage"]["cached_input_tokens"] > 0
    identity = {
        "provider": "openai",
        "model_id": "gpt-6-astra",
        "api_mode": "responses",
    }
    assert (
        store.get_last_model_metadata("reasoning_cache_probe", "cache-thread")
        == identity
    )
    assert agent.get_last_model_metadata(scope=second_scope) == identity
