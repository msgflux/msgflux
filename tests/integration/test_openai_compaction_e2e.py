"""Live OpenAI validation for native Responses compaction.

Requires ``OPENAI_API_KEY`` in the environment or repository ``.env``.
"""

import os

import pytest

import msgflux as mf
from msgflux import nn
from msgflux.data.stores import InMemoryCheckpointStore

mf.load_dotenv()

pytestmark = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY is required for live OpenAI integration tests.",
)


@pytest.mark.asyncio
async def test_openai_agent_compacts_and_continues_from_native_view():
    history = mf.ChatMessages(
        thread_id="openai_compaction_e2e",
        namespace="incident_analyst",
    )
    history.begin_turn(turn_id="initial_report")
    history.add_user("Scanner A stopped publishing updates at 09:02.")
    history.add_assistant("Orders may reserve stale inventory until recovery.")
    history.end_turn()
    store = InMemoryCheckpointStore()
    agent = nn.Agent(
        name="incident_analyst",
        model=mf.Model.chat_completion(
            "openai/gpt-5.6-luna",
            api_mode="responses",
            store=False,
        ),
        system_prompt="Preserve incident facts and answer briefly.",
        checkpoint_store=store,
        extensions=[
            nn.CompactionExtension(
                nn.CompactionPolicy(
                    context_capacity=64,
                    trigger_ratio=0.8,
                    reserved_output_tokens=0,
                    safety_margin_tokens=0,
                )
            )
        ],
    )
    scope = mf.ExecutionScope(
        thread_id="openai_compaction_e2e",
        namespace="incident_analyst",
        run_id="follow_up",
    )

    answer = await agent.acall(
        "Scanner A recovered. Name one verification step.",
        messages=history,
        scope=scope,
    )

    operation = history.latest_compaction()
    assert isinstance(answer, str) and answer
    assert operation is not None
    assert operation["views"][0]["format"] == "provider"
    assert operation["views"][0]["provider"] == "openai"
    assert operation["views"][0]["api_mode"] == "responses"
    assert operation["metadata"]["estimate_source"] == "provider"
    assert set(operation["metadata"]["usage"] or ()) <= {
        "input_tokens",
        "output_tokens",
        "cached_input_tokens",
    }
    state = store.load_state(agent.name, scope.thread_id, scope.run_id)
    assert any(item.get("type") == "compaction" for item in state["messages"]["items"])
