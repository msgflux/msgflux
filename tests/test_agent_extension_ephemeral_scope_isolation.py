"""Ephemeral extension state is isolated by fresh Agent invocation."""

import asyncio
import json
from unittest.mock import Mock

from msgflux.data.stores import SQLiteCheckpointStore
from msgflux.models.response import ModelResponse
from msgflux.models.tool_call_agg import ToolCallAggregator
from msgflux.nn import Agent, AgentExtension
from msgflux.nn.hooks import Hook
from msgflux.runtime.context import ExecutionScope, get_execution_context


def _scope(thread_id, run_id):
    return ExecutionScope(
        namespace="ephemeral_state_agent", thread_id=thread_id, run_id=run_id
    )


def _text_response(text):
    response = ModelResponse()
    response.set_response_type("text_generation")
    response.add(text)
    return response


def _tool_response(name, call_id):
    calls = ToolCallAggregator()
    calls.process(0, call_id, name, json.dumps({}))
    response = ModelResponse()
    response.set_response_type("tool_call")
    response.add(calls)
    return response


class _RememberMessage(AgentExtension):
    def __init__(self):
        super().__init__("remember_message")

    def hooks(self):
        def remember(payload):
            self.state()["message"] = payload.message
            return payload

        return (Hook(event="before_run", handler=remember),)


def _agent(store):
    extension = _RememberMessage()
    agent = Agent(
        name="ephemeral_state_agent",
        model=Mock(model_type="chat_completion"),
        checkpoint_store=store,
        extensions=[extension],
    )
    return agent, extension


def test_nested_sync_invocation_restores_parent_extension_state(tmp_path):
    database = tmp_path / "nested-state.sqlite3"
    store = SQLiteCheckpointStore(database)
    agent, extension = _agent(store)
    observed = []

    def nested_call():
        return agent("child", scope=_scope("child-thread", "child-run"))

    agent.tool_library.add(nested_call)

    def forward(**kwargs):
        scope = get_execution_context()["scope"]
        observed.append((scope.run_id, extension.state()["message"]))
        if scope.run_id == "outer-run" and len(observed) == 1:
            return _tool_response("nested_call", "nested-call")
        return _text_response(f"answer:{scope.run_id}")

    agent.generator.forward = forward
    try:
        result = agent("outer", scope=_scope("outer-thread", "outer-run"))
    finally:
        store.close()

    assert result == "answer:outer-run"
    assert observed == [
        ("outer-run", "outer"),
        ("child-run", "child"),
        ("outer-run", "outer"),
    ]

    reopened = SQLiteCheckpointStore(database)
    try:
        for thread_id, run_id in (
            ("outer-thread", "outer-run"),
            ("child-thread", "child-run"),
        ):
            state = reopened.load_state("ephemeral_state_agent", thread_id, run_id)
            assert state["status"] == "completed"
    finally:
        reopened.close()


def test_concurrent_async_child_invocations_keep_parent_and_sibling_state(tmp_path):
    database = tmp_path / "concurrent-state.sqlite3"
    store = SQLiteCheckpointStore(database)
    agent, extension = _agent(store)
    observed = []
    child_entries = 0
    both_children_entered = asyncio.Event()

    async def nested_pair():
        first, second = await asyncio.gather(
            agent.acall("child-A", scope=_scope("thread-A", "child-A")),
            agent.acall("child-B", scope=_scope("thread-B", "child-B")),
        )
        return f"{first}|{second}"

    agent.tool_library.add(nested_pair)

    async def forward(**kwargs):
        nonlocal child_entries
        scope = get_execution_context()["scope"]
        if scope.run_id == "outer-run" and not observed:
            observed.append((scope.run_id, extension.state()["message"]))
            return _tool_response("nested_pair", "nested-pair")
        if scope.run_id in {"child-A", "child-B"}:
            child_entries += 1
            if child_entries == 2:
                both_children_entered.set()
            await asyncio.wait_for(both_children_entered.wait(), timeout=5)
        observed.append((scope.run_id, extension.state()["message"]))
        return _text_response(f"answer:{scope.run_id}")

    agent.generator.aforward = forward

    async def run():
        return await agent.acall("outer", scope=_scope("outer-thread", "outer-run"))

    try:
        result = asyncio.run(asyncio.wait_for(run(), timeout=10))
    finally:
        store.close()

    assert result == "answer:outer-run"
    assert sorted(observed) == sorted(
        [
            ("outer-run", "outer"),
            ("child-A", "child-A"),
            ("child-B", "child-B"),
            ("outer-run", "outer"),
        ]
    )

    reopened = SQLiteCheckpointStore(database)
    try:
        for thread_id, run_id in (
            ("outer-thread", "outer-run"),
            ("thread-A", "child-A"),
            ("thread-B", "child-B"),
        ):
            state = reopened.load_state("ephemeral_state_agent", thread_id, run_id)
            assert state["status"] == "completed"
    finally:
        reopened.close()
