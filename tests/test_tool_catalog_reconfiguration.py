"""Deferred tool activation follows the current registry after durable reopen."""

from unittest.mock import AsyncMock, Mock

import pytest

from msgflux.data.stores import SQLiteCheckpointStore
from msgflux.chat_messages import ChatMessages
from msgflux.models.response import ModelResponse
from msgflux.models.tool_call_agg import ToolCallAggregator
from msgflux.nn import Agent
from msgflux.nn.modules.tool.library import ToolLibrary
from msgflux.runtime import ExecutionScope


def _text_response(value="done"):
    response = ModelResponse()
    response.set_response_type("text_generation")
    response.add(value)
    return response


def _search_response():
    calls = ToolCallAggregator()
    calls.process(0, "search", "tool_search", '{"select":["lookup"]}')
    response = ModelResponse()
    response.set_response_type("tool_call")
    response.add(calls)
    return response


def _lookup(*, deferred):
    def lookup(query: str) -> str:
        """Look up a value."""
        return query

    lookup.tool_config = {"defer_loading": deferred}
    return lookup


def _agent(store, *, deferred, asynchronous):
    model = Mock()
    model.model_type = "chat_completion"
    agent = Agent(
        name="main",
        model=model,
        tools=[_lookup(deferred=deferred)],
        checkpoint_store=store,
    )
    if asynchronous:
        agent.generator.aforward = AsyncMock(return_value=_text_response())
    else:
        agent.generator.forward = Mock(return_value=_text_response())
    return agent


async def _invoke(agent, asynchronous, message, scope):
    if asynchronous:
        return await agent.acall(message, scope=scope)
    return agent(message, scope=scope)


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_loaded_deferred_tool_can_become_active_after_reopen(
    tmp_path, asynchronous
):
    path = tmp_path / "checkpoints.sqlite3"
    first_store = SQLiteCheckpointStore(path)
    first = _agent(first_store, deferred=True, asynchronous=asynchronous)
    if asynchronous:
        first.generator.aforward = AsyncMock(
            side_effect=[_search_response(), _text_response()]
        )
    else:
        first.generator.forward = Mock(
            side_effect=[_search_response(), _text_response()]
        )

    scope = ExecutionScope(thread_id="thread", run_id="first")
    try:
        assert await _invoke(first, asynchronous, "load lookup", scope) == "done"
        saved = first_store.load_state("main", "thread", "first")
        assert saved["messages"]["metadata"]["tool_catalogs"] == {
            "main_tool_library": {"loaded_tools": ["lookup"]}
        }
    finally:
        await first.aclose()
        first_store.close()

    second_store = SQLiteCheckpointStore(path)
    second = _agent(second_store, deferred=False, asynchronous=asynchronous)
    original_state = second_store.load_state("main", "thread", "first")
    try:
        result = await _invoke(
            second,
            asynchronous,
            "continue",
            ExecutionScope(thread_id="thread", run_id="second"),
        )
        assert result == "done"
        if asynchronous:
            assert second.generator.aforward.await_count == 1
        else:
            assert second.generator.forward.call_count == 1
            catalog = second.generator.forward.call_args.kwargs["tool_catalog"]
        if asynchronous:
            catalog = second.generator.aforward.await_args.kwargs["tool_catalog"]
        entries = catalog.visible_entries()
        assert {entry.name for entry in entries} == {"lookup"}
        assert entries[0].input_schema["properties"].keys() == {"query"}
        state = second_store.load_state("main", "thread", "second")
        assert state["status"] == "completed"
        assert state["messages"]["metadata"]["tool_catalogs"] == {
            "main_tool_library": {"loaded_tools": ["lookup"]}
        }
        assert second_store.load_state("main", "thread", "first") == original_state
    finally:
        await second.aclose()
        second_store.close()


def test_catalog_filters_only_loaded_tools_that_are_no_longer_deferred():
    active = _lookup(deferred=False)
    removed = _lookup(deferred=True)
    removed.__name__ = "removed"
    retained = _lookup(deferred=True)
    retained.__name__ = "retained"
    library = ToolLibrary("catalog", [active, removed, retained])
    messages = ChatMessages()
    messages.load_tools(library.name, ["lookup", "removed", "retained"])
    library.remove("removed")

    entries = library.get_tool_catalog_view(messages, thread_id="thread").entries
    by_name = {entry.name: entry for entry in entries}
    assert set(by_name) == {"lookup", "retained", "tool_search"}
    assert not by_name["lookup"].deferred
    assert not by_name["lookup"].loaded
    assert by_name["retained"].deferred
    assert by_name["retained"].loaded
    assert messages.get_loaded_tools(library.name) == {"lookup", "removed", "retained"}
    with pytest.raises(ValueError, match="Deferred tools are not available: lookup"):
        library.load_tools(messages, ["lookup"])
