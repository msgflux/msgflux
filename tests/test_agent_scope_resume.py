"""Scope transitions retain investigation and budgets through real Agent resume."""

import json
from unittest.mock import AsyncMock, Mock

import pytest

from msgflux.data.stores import InMemoryCheckpointStore
from msgflux.generation.reasoning.react import ReAct
from msgflux.models.response import ModelResponse
from msgflux.models.tool_call_agg import ToolCallAggregator
from msgflux.nn import Agent, ToolTurnLimitExtension
from msgflux.runtime.context import ExecutionScope
from msgflux.tools.builtin import close_context_scope, open_context_scope


def text_response(text, *, flow=False):
    response = ModelResponse()
    response.set_response_type("structured" if flow else "text_generation")
    response.add({"thought": "done", "final_answer": text} if flow else text)
    return response


def scope_agent(store, responses, *, flow=False, extensions=None):
    agent = Agent(
        name="scopes",
        model=Mock(model_type="chat_completion"),
        tools=[open_context_scope, close_context_scope],
        checkpoint_store=store,
        generation_schema=ReAct if flow else None,
        extensions=extensions,
    )
    agent.generator.forward = Mock(side_effect=responses)
    agent.generator.aforward = AsyncMock(side_effect=responses)
    return agent


def scope_response(name, args, flow):
    result = ModelResponse()
    if flow:
        result.set_response_type("structured")
        result.add(
            {"thought": "transition", "actions": [{"name": name, "arguments": args}]}
        )
    else:
        calls = ToolCallAggregator()
        calls.process(0, name, name, json.dumps(args))
        result.set_response_type("tool_call")
        result.add(calls)
    return result


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("flow", [False, True])
async def test_resume_closes_scope_without_resetting_budget(asynchronous, flow):
    store = InMemoryCheckpointStore()
    scope = ExecutionScope(namespace="scopes", thread_id="thread", run_id="run")

    def make(responses):
        return scope_agent(
            store,
            responses,
            flow=flow,
            extensions=[ToolTurnLimitExtension(2)],
        )

    first = make(
        [
            scope_response(
                "open_context_scope",
                {"name": "research", "summary": "Start research"},
                flow,
            ),
            RuntimeError("provider unavailable"),
        ]
    )
    with pytest.raises(RuntimeError, match="provider unavailable"):
        if asynchronous:
            await first.acall("Investigate", scope=scope)
        else:
            first("Investigate", scope=scope)
    checkpoint = store.load_state("scopes", "thread", "run")
    assert checkpoint["runtime"]["branch_id"] == "research"
    second = make(
        [scope_response("close_context_scope", {"summary": "Research result"}, flow)]
    )
    output = await second.acall(scope=scope) if asynchronous else second(scope=scope)
    assert output["completed_tool_turns"] == 2
    checkpoint = store.load_state("scopes", "thread", "run")
    assert checkpoint["runtime"]["branch_id"] == "root"
    assert checkpoint["status"] == "completed"
    state = checkpoint["messages"]
    assert "Start research" in str(state["items"])
    assert "Research result" in str(state["items"])
    branches = state["metadata"]["runtime"]["context_scopes"]["branches"]
    assert branches["research"]["closed"] is True
    assert "close_context_scope" in str(branches["research"]["items"])


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("flow,nested", [(False, False), (False, True), (True, False)])
@pytest.mark.parametrize("mode", ["fork", "prefix", "continue"])
async def test_scope_close_keeps_destination_turn(
    checkpoint_store, asynchronous, flow, mode, nested
):
    from msgflux.chat_messages import ChatMessages

    store = checkpoint_store
    responses = [scope_response("open_context_scope", {"name": "research"}, flow)]
    if nested:
        responses.append(scope_response("open_context_scope", {"name": "source"}, flow))
    responses.append(text_response("private investigation", flow=flow))
    source_scope = ExecutionScope(
        namespace="scopes", thread_id="source", run_id="original"
    )
    first = scope_agent(store, responses, flow=flow)
    if asynchronous:
        await first.acall("Investigate", scope=source_scope)
    else:
        first("Investigate", scope=source_scope)
    source = store.load_state("scopes", "source", "original")
    fork_source = "original"
    fork_source_state = source
    if mode == "prefix":
        later = scope_agent(
            store, [text_response("excluded future", flow=flow)], flow=flow
        )
        later_scope = ExecutionScope(
            namespace="scopes", thread_id="source", run_id="later"
        )
        if asynchronous:
            await later.acall("Future request", scope=later_scope)
        else:
            later("Future request", scope=later_scope)
        fork_source = "later"
        fork_source_state = store.load_state("scopes", "source", "later")
    target_thread = "source" if mode == "continue" else "fork"
    if mode != "continue":
        options = {}
        if mode == "prefix":
            options["at_item_id"] = source["messages"]["items"][-1]["item_id"]
        store.fork_run(
            "scopes",
            "source",
            fork_source,
            target_thread_id=target_thread,
            target_run_id="destination",
            status="paused",
            **options,
        )

    responses = []
    if nested:
        responses.append(
            scope_response("close_context_scope", {"name": "source"}, flow)
        )
    responses.extend(
        [
            scope_response(
                "close_context_scope", {"summary": "Research summary"}, flow
            ),
            text_response("done", flow=flow),
        ]
    )
    second = scope_agent(store, responses, flow=flow)
    target_scope = ExecutionScope(
        namespace="scopes", thread_id=target_thread, run_id="destination"
    )
    if mode == "continue":
        result = (
            await second.acall("Continue", scope=target_scope)
            if asynchronous
            else second("Continue", scope=target_scope)
        )
    else:
        result = (
            await second.acall(scope=target_scope)
            if asynchronous
            else second(scope=target_scope)
        )
    assert result == ({"thought": "done", "final_answer": "done"} if flow else "done")
    target = store.load_state("scopes", target_thread, "destination")
    assert target["status"] == "completed"
    assert target["runtime"]["run_id"] == "destination"
    assert target["scope"]["run_id"] == "destination"
    assert target["runtime"]["branch_id"] == "root"
    restored = ChatMessages()
    restored._hydrate_state(target["messages"])
    assert [(turn["turn_id"], turn["status"]) for turn in restored.turns] == [
        ("original", "completed"),
        ("destination", "completed"),
    ]
    assert restored.get_active_turn() is None
    source_events = [
        item for item in source["messages"]["items"] if item.get("type") == "turn"
    ]
    assert [item for item in restored if item.get("type") == "turn"][
        :2
    ] == source_events
    assert len({item["item_id"] for item in restored}) == len(restored)
    assert "private investigation" not in str(target["messages"]["items"])
    assert "excluded future" not in str(target["messages"]["items"])
    assert "Future request" not in str(target["messages"]["items"])
    assert "Research summary" in str(target["messages"]["items"])
    assert "private investigation" in str(
        target["messages"]["metadata"]["runtime"]["context_scopes"]["branches"]
    )
    assert store.load_state("scopes", "source", "original") == source
    assert store.load_state("scopes", "source", fork_source) == fork_source_state
    if mode != "continue":
        assert store.load_state("scopes", target_thread, "original") is None
        assert target["_checkpoint"]["fork_of"]["run_id"] == fork_source


@pytest.mark.parametrize("position", ["before", "at"])
@pytest.mark.parametrize("nested", [False, True])
def test_fork_rejects_prefix_before_active_scope_origin(
    checkpoint_store, position, nested
):
    store = checkpoint_store
    first_responses = (
        [scope_response("open_context_scope", {"name": "outer"}, False)]
        if nested
        else []
    )
    first = scope_agent(store, [*first_responses, text_response("first")])
    first(
        "First",
        scope=ExecutionScope(namespace="scopes", thread_id="source", run_id="first"),
    )
    second = scope_agent(
        store,
        [
            scope_response("open_context_scope", {"name": "research"}, False),
            text_response("inside research"),
        ],
    )
    second(
        "Investigate",
        scope=ExecutionScope(namespace="scopes", thread_id="source", run_id="second"),
    )
    source = store.load_state("scopes", "source", "second")
    boundary = next(
        item
        for item in source["messages"]["items"]
        if item.get("type") == "turn"
        and item.get("event") == ("complete" if position == "at" else "start")
        and item.get("turn_id") == ("first" if position == "at" else "second")
    )
    with pytest.raises(ValueError, match="active context scope origin"):
        store.fork_run(
            "scopes",
            "source",
            "second",
            target_thread_id="fork",
            target_run_id="destination",
            at_item_id=boundary["item_id"],
            position=position,
            status="paused",
        )
    assert store.load_state("scopes", "fork", "destination") is None
    assert store.list_runs("scopes", "fork") == []
    assert store.load_state("scopes", "source", "second") == source
