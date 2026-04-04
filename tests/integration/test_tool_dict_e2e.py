"""E2E test: agent calls a tool with dict[K, V] parameter (transport lowering)."""

import pytest

import msgflux as mf
from msgflux import nn

mf.load_dotenv()


model = mf.Model.chat_completion("openai/gpt-4.1-mini")


# ── Tool with dict[str, str] parameter ───────────────────────────────────────

_store: dict = {}


def store_fields(fields: dict[str, str]) -> dict:
    """Store key-value pairs into the session store.

    Args:
        fields: A mapping of field names to their string values.
            Example: {"city": "Austin", "country": "USA"}
    """
    _store.update(fields)
    return {"stored": list(fields.keys())}


# ── Agent ─────────────────────────────────────────────────────────────────────

class StoreAgent(nn.Agent):
    model = model
    system_message = "You are a data entry assistant."
    instructions = (
        "When the user gives you key-value pairs, call store_fields once "
        "with all the pairs together."
    )
    message_fields = {"task": "user.text"}
    tools = [store_fields]
    response_mode = "response"


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestDictToolE2E:

    def setup_method(self):
        _store.clear()

    def test_agent_calls_tool_with_dict_param(self):
        agent = StoreAgent()
        msg = mf.Message()
        msg.set("user.text", 'Store these: city="Austin", country="USA"')
        agent(msg, messages=[])
        assert "city" in _store
        assert _store["city"] == "Austin"
        assert "country" in _store
        assert _store["country"] == "USA"

    @pytest.mark.asyncio
    async def test_agent_calls_tool_with_dict_param_async(self):
        agent = StoreAgent()
        msg = mf.Message()
        msg.set("user.text", 'Store these: name="Alice", role="engineer"')
        await agent.acall(msg, messages=[])
        assert "name" in _store
        assert _store["name"] == "Alice"
        assert "role" in _store
        assert _store["role"] == "engineer"
