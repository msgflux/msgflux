"""Same-Agent nested calls must keep independent invocation state."""

import asyncio
from unittest.mock import Mock

from msgflux.data.stores import SQLiteCheckpointStore
from msgflux.models.response import ModelResponse
from msgflux.models.tool_call_agg import ToolCallAggregator
from msgflux.nn import Agent
from msgflux.runtime.context import ExecutionScope, get_execution_context
from msgflux.runtime.agent_run import get_agent_run
from msgflux.runtime.workspace.receipts import get_command_receipt_persist


def _text_response(text):
    response = ModelResponse()
    response.set_response_type("text_generation")
    response.add(text)
    return response


def _tool_response(name, call_id):
    calls = ToolCallAggregator()
    calls.process(0, call_id, name, "{}")
    response = ModelResponse()
    response.set_response_type("tool_call")
    response.add(calls)
    return response


def _scope(thread_id, run_id):
    return ExecutionScope(
        namespace="nested_scope_agent", thread_id=thread_id, run_id=run_id
    )


def _checkpoint_state(path, thread_id, run_id):
    store = SQLiteCheckpointStore(path)
    try:
        state = store.load_state("nested_scope_agent", thread_id, run_id)
    finally:
        store.close()
    return state


def test_sync_nested_call_on_same_agent_keeps_outer_checkpoint(tmp_path):
    path = tmp_path / "nested-sync.sqlite3"
    calls = []
    restored_context = []
    checkpoints = SQLiteCheckpointStore(path)
    agent = Agent(
        name="nested_scope_agent",
        model=Mock(model_type="chat_completion"),
        checkpoint_store=checkpoints,
    )

    def nested_call() -> str:
        parent_run = get_agent_run()
        parent_receipt_persist = get_command_receipt_persist()
        result = agent("child marker", scope=_scope("child-thread", "child-run"))
        restored_context.append(
            (
                get_agent_run() is parent_run,
                get_command_receipt_persist() is parent_receipt_persist,
            )
        )
        return result

    agent.tool_library.add(nested_call)

    def forward(**kwargs):
        scope = get_execution_context()["scope"]
        calls.append(scope)
        if scope.run_id == "outer-run":
            if sum(call.run_id == "outer-run" for call in calls) == 1:
                return _tool_response("nested_call", "nested-call")
            return _text_response("outer answer")
        if scope.run_id == "child-run":
            return _text_response("child answer")
        raise AssertionError(f"Unexpected run {scope.run_id}")

    agent.generator.forward = forward
    try:
        assert agent("outer marker", scope=_scope("outer-thread", "outer-run")) == (
            "outer answer"
        )
    finally:
        checkpoints.close()

    assert restored_context == [(True, True)]
    assert [scope.run_id for scope in calls] == [
        "outer-run",
        "child-run",
        "outer-run",
    ]
    for thread_id, run_id, marker in (
        ("outer-thread", "outer-run", "outer answer"),
        ("child-thread", "child-run", "child answer"),
    ):
        state = _checkpoint_state(path, thread_id, run_id)
        assert state["status"] == "completed"
        assert state["runtime"]["thread_id"] == thread_id
        assert state["runtime"]["run_id"] == run_id
        contents = [item.get("content") for item in state["messages"]["items"]]
        assert marker in contents


def test_async_gathered_calls_on_same_agent_keep_checkpoints_isolated(tmp_path):
    path = tmp_path / "nested-async.sqlite3"
    checkpoints = SQLiteCheckpointStore(path)
    agent = Agent(
        name="nested_scope_agent",
        model=Mock(model_type="chat_completion"),
        checkpoint_store=checkpoints,
    )
    child_calls = 0
    both_children_entered = asyncio.Event()
    calls = []

    async def nested_pair() -> str:
        results = await asyncio.gather(
            agent.acall(
                "child A marker", scope=_scope("child-A-thread", "child-A-run")
            ),
            agent.acall(
                "child B marker", scope=_scope("child-B-thread", "child-B-run")
            ),
        )
        return "|".join(results)

    agent.tool_library.add(nested_pair)

    async def aforward(**kwargs):
        nonlocal child_calls
        scope = get_execution_context()["scope"]
        calls.append((scope.thread_id, scope.run_id))
        if scope.run_id == "outer-run":
            if len(calls) == 1:
                return _tool_response("nested_pair", "nested-pair")
            return _text_response("outer answer")
        child_calls += 1
        if child_calls == 2:
            both_children_entered.set()
        await asyncio.wait_for(both_children_entered.wait(), timeout=5)
        return _text_response(f"answer for {scope.run_id}")

    agent.generator.aforward = aforward
    try:
        result = asyncio.run(
            agent.acall("outer marker", scope=_scope("outer-thread", "outer-run"))
        )
        assert result == "outer answer"
    finally:
        checkpoints.close()

    assert len(calls) == 4
    for thread_id, run_id in (
        ("outer-thread", "outer-run"),
        ("child-A-thread", "child-A-run"),
        ("child-B-thread", "child-B-run"),
    ):
        state = _checkpoint_state(path, thread_id, run_id)
        assert state["status"] == "completed"
        assert state["runtime"]["thread_id"] == thread_id
        assert state["runtime"]["run_id"] == run_id
        contents = [item.get("content") for item in state["messages"]["items"]]
        expected = "outer answer" if run_id == "outer-run" else f"answer for {run_id}"
        assert expected in contents
