"""Execution authorization for per-run Agent tool filters."""

import json
from itertools import count
from unittest.mock import AsyncMock, Mock

import pytest

from msgflux.chat_messages import ChatMessages
from msgflux.data.stores import InMemoryCheckpointStore, SQLiteCheckpointStore
from msgflux.exceptions import TaskPauseRequestedError
from msgflux.generation.reasoning.react import ReAct
from msgflux.models.response import ModelResponse
from msgflux.models.tool_call_agg import ToolCallAggregator
from msgflux.nn import Agent
from msgflux.runtime import AgentApprovals, ExecutionScope, InMemoryApprovalStore
from msgflux.tasks import InMemoryTaskStore
from msgflux.tools.config import tool_config

_CALL_IDS = count()


def _tool_call(*calls):
    aggregator = ToolCallAggregator()
    for index, (name, arguments) in enumerate(calls):
        aggregator.process(
            index, f"call-{next(_CALL_IDS)}", name, json.dumps(arguments)
        )
    response = ModelResponse()
    response.set_response_type("tool_call")
    response.add(aggregator)
    response.metadata = {}
    return response


def _text(value="done"):
    response = ModelResponse()
    response.set_response_type("text_generation")
    response.add(value)
    response.reasoning = None
    response.metadata = {}
    return response


def _flow_call(name, arguments):
    response = ModelResponse()
    response.set_response_type("structured")
    response.add(
        {"thought": "call", "actions": [{"name": name, "arguments": arguments}]}
    )
    return response


def _flow_final(value="done"):
    response = ModelResponse()
    response.set_response_type("structured")
    response.add({"thought": None, "actions": None, "final_answer": value})
    return response


def _agent(name, tools, responses, **kwargs):
    agent = Agent(
        name=name, model=Mock(model_type="chat_completion"), tools=tools, **kwargs
    )
    agent.generator.forward = Mock(side_effect=responses)
    agent.generator.aforward = AsyncMock(side_effect=responses)
    return agent


def _scope(name, run_id):
    return ExecutionScope(namespace=name, thread_id="thread", run_id=run_id)


def _outputs_for(history, tool_name):
    call_names = {
        item["call_id"]: item["name"]
        for item in history
        if item.get("type") == "function_call"
    }
    return [
        item["output"]
        for item in history
        if item.get("type") == "function_call_output"
        and call_names.get(item.get("call_id")) == tool_name
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_native_filter_denies_only_blocked_members_of_a_batch(asynchronous):
    calls = []

    def allowed_tool():
        calls.append("allowed")
        return "allowed-result"

    def blocked_tool():
        calls.append("blocked")
        return "blocked-result"

    agent = _agent(
        "native_filter",
        [allowed_tool, blocked_tool],
        [_tool_call(("blocked_tool", {}), ("allowed_tool", {})), _text()],
    )
    try:
        history = ChatMessages()
        kwargs = {
            "messages": history,
            "scope": _scope("native_filter", "native-run"),
            "tool_filter": {"allow": "allowed_tool"},
        }
        if asynchronous:
            await agent.acall("run", **kwargs)
        else:
            agent("run", **kwargs)
        assert calls == ["allowed"]
        outcomes = [
            item["output"]
            for item in history
            if item.get("type") == "function_call_output"
        ]
        assert len(outcomes) == 2
        assert "blocked_tool` is blocked by" in outcomes[0]
        assert "allowed-result" in outcomes[1]
    finally:
        await agent.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_flow_control_calls_are_filtered_at_dispatch(asynchronous):
    calls = []

    def blocked_tool():
        calls.append("blocked")
        return "bad"

    agent = _agent(
        "flow_filter",
        [blocked_tool],
        [_flow_call("blocked_tool", {}), _flow_final()],
        generation_schema=ReAct,
    )
    try:
        if asynchronous:
            await agent.acall("run", tool_filter={"block": "*"})
        else:
            agent("run", tool_filter={"block": "*"})
        assert calls == []
    finally:
        await agent.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_search_container_does_not_authorize_deferred_resources(asynchronous):
    calls = []

    def safe_deferred(query: str) -> str:
        calls.append("safe")
        return query.upper()

    def other_deferred(query: str) -> str:
        calls.append("other")
        return query.upper()

    safe_deferred.tool_config = {"defer_loading": True}
    other_deferred.tool_config = {"defer_loading": True}
    agent = _agent(
        "search_filter",
        [safe_deferred, other_deferred],
        [
            _tool_call(("tool_search", {"select": ["other_deferred"]})),
            _tool_call(("tool_search", {"query": "safe_deferred"})),
            _tool_call(("other_deferred", {"query": "denied"})),
            _tool_call(("safe_deferred", {"query": "allowed"})),
            _text(),
        ],
    )
    try:
        history = ChatMessages()
        kwargs = {
            "messages": history,
            "scope": _scope("search_filter", f"search-{asynchronous}"),
            "tool_filter": {"allow": ["tool_search", "safe_deferred"]},
        }
        if asynchronous:
            await agent.acall("run", **kwargs)
        else:
            agent("run", **kwargs)
        assert calls == ["safe"]
        assert history.get_loaded_tools(agent.tool_library.name) == {"safe_deferred"}
        search_outputs = _outputs_for(history, "tool_search")
        assert json.loads(search_outputs[0])["matches"] == []
        assert json.loads(search_outputs[0])["loaded"] == []
        search_output = search_outputs[1]
        assert json.loads(search_output)["matches"] == ["safe_deferred"]
    finally:
        await agent.aclose()


@pytest.mark.asyncio
async def test_allowing_search_alone_exposes_and_loads_no_deferred_tools():
    calls = []

    def private_deferred(query: str) -> str:
        calls.append(query)
        return query

    private_deferred.tool_config = {"defer_loading": True}
    agent = _agent(
        "search_only",
        [private_deferred],
        [
            _tool_call(("tool_search", {"query": "private"})),
            _tool_call(("private_deferred", {"query": "guessed"})),
            _text(),
        ],
    )
    try:
        history = ChatMessages()
        await agent.acall(
            "run",
            messages=history,
            scope=_scope("search_only", "search-only"),
            tool_filter={"allow": "tool_search"},
        )
        assert calls == []
        assert history.get_loaded_tools(agent.tool_library.name) == set()
        search_output = _outputs_for(history, "tool_search")[0]
        assert json.loads(search_output)["matches"] == []
    finally:
        await agent.aclose()


@pytest.mark.asyncio
async def test_blocking_search_does_not_block_an_authorized_deferred_tool():
    calls = []

    def safe_deferred(query: str) -> str:
        calls.append(query)
        return query

    safe_deferred.tool_config = {"defer_loading": True}
    agent = _agent(
        "search_blocked_resource_allowed",
        [safe_deferred],
        [_tool_call(("safe_deferred", {"query": "ok"})), _text()],
    )
    try:
        await agent.acall(
            "run",
            scope=_scope("search_blocked_resource_allowed", "direct-resource"),
            tool_filter={"block": "tool_search"},
        )
        assert calls == ["ok"]
    finally:
        await agent.aclose()


@pytest.mark.asyncio
async def test_grouped_bucket_and_captured_member_filters_are_distinct():
    @tool_config(background=True)
    def worker(value: str) -> str:
        return value

    store = InMemoryTaskStore()
    list_calls = []
    get_calls = []
    original_list = store.list
    original_get = store.get
    store.list = lambda *args, **kwargs: (
        list_calls.append(True),
        original_list(*args, **kwargs),
    )[1]
    store.get = lambda *args, **kwargs: (
        get_calls.append(True),
        original_get(*args, **kwargs),
    )[1]
    agent = _agent(
        "task_bucket_filter",
        [worker],
        [_tool_call(("task_list", {})), _text()],
    )
    agent.tool_library._task_store = store
    agent.tool_library.extensions["background_tasks"].enable_grouping()
    try:
        await agent.acall(
            "run",
            scope=_scope("task_bucket_filter", "blocked-bucket"),
            tool_filter={"block": "task"},
        )
        assert not list_calls

        agent.generator.aforward = AsyncMock(
            side_effect=[
                _tool_call(("task", {"request": {"operation": "list"}})),
                _tool_call(
                    ("task", {"request": {"operation": "output", "task_id": "x"}})
                ),
                _text(),
            ]
        )
        await agent.acall(
            "run",
            scope=_scope("task_bucket_filter", "member-filter"),
            tool_filter={"allow": "task_list"},
        )
        assert list_calls
        assert not get_calls

        agent.generator.aforward = AsyncMock(
            side_effect=[_tool_call(("task_output", {"task_id": "x"})), _text()]
        )
        await agent.acall(
            "run",
            scope=_scope("task_bucket_filter", "direct-member-filter"),
            tool_filter={"allow": "task_list"},
        )
        assert not get_calls
    finally:
        await agent.aclose()


@pytest.mark.asyncio
async def test_filtered_call_never_creates_an_approval_request():
    checkpoint = InMemoryCheckpointStore()
    journal = InMemoryApprovalStore()
    calls = []

    def protected():
        calls.append("executed")
        return "secret"

    policy = AgentApprovals(journal, {"protected": "v1"}, "policy-v1")
    agent = _agent(
        "approval_filter",
        [protected],
        [_tool_call(("protected", {})), _text()],
        checkpoint_store=checkpoint,
        approvals=policy,
    )
    try:
        assert (
            await agent.acall(
                "run",
                scope=_scope("approval_filter", "approval-denial"),
                tool_filter={"block": "protected"},
            )
            == "done"
        )
        assert calls == []
        assert journal.pending("approval_filter", "thread", "approval-denial") == []
        state = checkpoint.load_state("approval_filter", "thread", "approval-denial")
        assert "pending_approvals" not in state["runtime"]["extensions"]
    finally:
        await agent.aclose()


@pytest.mark.asyncio
async def test_approval_replay_cannot_execute_after_resume_filter_denies_call():
    checkpoint = InMemoryCheckpointStore()
    journal = InMemoryApprovalStore()
    calls = []

    def protected():
        calls.append("executed")
        return "secret"

    approvals = AgentApprovals(journal, {"protected": "v1"}, "policy-v1")
    agent = _agent(
        "approval_replay_filter",
        [protected],
        [_tool_call(("protected", {})), _text()],
        checkpoint_store=checkpoint,
        approvals=approvals,
    )
    scope = ExecutionScope(
        namespace="approval_replay_filter",
        thread_id="thread",
        run_id="approval-replay",
        principal="user",
    )
    try:
        with pytest.raises(TaskPauseRequestedError):
            await agent.acall("start", scope=scope)
        request = journal.pending("approval_replay_filter", "thread", scope.run_id)[0]
        agent.decide_approval(
            request.request_id,
            approved=True,
            decided_by="host",
            approvals=approvals,
        )
        await agent.acall("resume", scope=scope, tool_filter={"block": "protected"})
        assert calls == []
        state = checkpoint.load_state("approval_replay_filter", "thread", scope.run_id)
        assert "pending_approvals" not in state["runtime"]["extensions"]
    finally:
        await agent.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("explicit_override", ["inherit", "replace", "clear"])
async def test_checkpoint_resume_inherits_or_replaces_filter(
    tmp_path, explicit_override
):
    store_path = tmp_path / "filter-checkpoint.sqlite3"
    store = SQLiteCheckpointStore(store_path)
    calls = []

    def dangerous():
        calls.append("dangerous")
        return "ran"

    first = _agent(
        "filter_resume",
        [dangerous],
        [RuntimeError("fake model failure")],
        checkpoint_store=store,
    )
    scope = _scope("filter_resume", f"resume-{explicit_override}")
    try:
        with pytest.raises(RuntimeError, match="fake model failure"):
            await first.acall(
                "start",
                scope=scope,
                tool_filter={"block": "dangerous"},
            )
    finally:
        await first.aclose()
        store.close()

    reopened = SQLiteCheckpointStore(store_path)
    second = _agent(
        "filter_resume",
        [dangerous],
        [_tool_call(("dangerous", {})), _text()],
        checkpoint_store=reopened,
    )
    try:
        resume_kwargs = {
            "inherit": {},
            "replace": {"tool_filter": {"allow": "dangerous"}},
            "clear": {"tool_filter": None},
        }[explicit_override]
        await second.acall("resume", scope=scope, **resume_kwargs)
        allowed = explicit_override != "inherit"
        assert calls == (["dangerous"] if allowed else [])
        checkpoint = reopened.load_state("filter_resume", "thread", scope.run_id)
        extensions = checkpoint["runtime"]["extensions"]
        if explicit_override == "clear":
            assert "tool_filter_policy" not in extensions
        else:
            expected = ["dangerous"] if allowed else []
            assert extensions["tool_filter_policy"]["allowed_names"] == expected

        second.generator.aforward = AsyncMock(
            side_effect=[_tool_call(("dangerous", {})), _text()]
        )
        await second.acall(
            "new turn",
            scope=_scope("filter_resume", f"new-{explicit_override}"),
        )
        assert len(calls) == (2 if allowed else 1)
    finally:
        await second.aclose()
        reopened.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid_policy",
    [None, [], "unrestricted", False, 1, {}, {"allowed_names": "*"}],
)
async def test_invalid_saved_policy_cannot_restore_unrestricted_access(invalid_policy):
    store = InMemoryCheckpointStore()
    effects = []

    def protected():
        effects.append("executed")
        return "private"

    first = _agent(
        "invalid_policy",
        [protected],
        [RuntimeError("provider unavailable")],
        checkpoint_store=store,
    )
    scope = _scope("invalid_policy", "failed-run")
    try:
        with pytest.raises(RuntimeError, match="provider unavailable"):
            await first.acall("start", scope=scope, tool_filter={"block": "*"})
    finally:
        await first.aclose()

    state = store.load_state("invalid_policy", "thread", scope.run_id)
    state["runtime"]["extensions"]["tool_filter_policy"] = invalid_policy
    store.save_state("invalid_policy", "thread", scope.run_id, state)
    second = _agent(
        "invalid_policy",
        [protected],
        [_tool_call(("protected", {})), _text()],
        checkpoint_store=store,
    )
    try:
        with pytest.raises(ValueError, match="Invalid checkpointed tool-filter policy"):
            await second.acall(scope=scope)
        assert not effects
        second.generator.aforward.assert_not_called()
    finally:
        await second.aclose()
