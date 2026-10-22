"""Black-box adversarial regression tests for ``tool_filter`` enforcement.

These tests exercise only public Agent/tool APIs with deterministic scripted
models. They challenge the gap between presenting a filtered tool catalog to the
model and actually enforcing the filter at tool-resolution time:

* a blocked tool must never run, even when the model still emits a call for it;
* denied calls must still be paired with a ``function_call_output``;
* denied calls must not create host approvals;
* the effective filter must be an immutable per-invocation snapshot;
* concurrent invocations on one Agent must keep their own filters;
* a plain ``transform_tool_catalog`` hook is presentation-only and never grants
  or denies execution;
* a nested child Agent must not inherit an unrelated parent name filter.

Each case is deterministic: scripted responses plus bounded ``asyncio`` events,
no sleeps and no unbounded waits.
"""

import asyncio
from dataclasses import replace
from unittest.mock import AsyncMock, Mock

import pytest

from msgflux.chat_messages import ChatMessages
from msgflux.data.stores import InMemoryCheckpointStore
from msgflux.models.response import ModelResponse
from msgflux.models.tool_call_agg import ToolCallAggregator
from msgflux.nn import Agent
from msgflux.nn.hooks import Hook
from msgflux.runtime import AgentApprovals, ExecutionScope, InMemoryApprovalStore
from msgflux.runtime.context import get_execution_scope
from msgflux.runtime.events import EventType, _EventSink, _capture_events


def _tool_response(*calls):
    """Build a tool-call response from ``(call_id, name, arguments)`` tuples."""
    aggregator = ToolCallAggregator()
    for index, (call_id, name, arguments) in enumerate(calls):
        aggregator.process(index, call_id, name, arguments)
    response = ModelResponse()
    response.set_response_type("tool_call")
    response.add(aggregator)
    response.reasoning = None
    return response


def _text_response(text):
    response = ModelResponse()
    response.set_response_type("text_generation")
    response.add(text)
    response.reasoning = None
    return response


def _install(agent, responses, asynchronous):
    if asynchronous:
        agent.generator.aforward = AsyncMock(side_effect=responses)
    else:
        agent.generator.forward = Mock(side_effect=responses)


async def _invoke(agent, message, *, asynchronous, **kwargs):
    if asynchronous:
        return await agent.acall(message, **kwargs)
    return agent(message, **kwargs)


def _denied_call_ids(events):
    """Collect call ids that a run reported as blocked or failed."""
    denied = set()
    for event in events:
        if event.type in {EventType.TOOL_BLOCKED, EventType.TOOL_PERMISSION_DENIED}:
            denied.add(event.data.get("tool_call_id"))
        elif event.type == EventType.TOOL_END and event.data.get("error"):
            denied.add(event.data.get("tool_call_id"))
    denied.discard(None)
    return denied


def _tool_output_call_ids(messages):
    return {
        item["call_id"]
        for item in messages
        if item.get("type") == "function_call_output"
    }


def _agent(tools, *, name="filter_agent", **kwargs):
    model = Mock()
    model.model_type = "chat_completion"
    return Agent(name=name, model=model, tools=list(tools), **kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_blocked_tool_in_mixed_batch_is_denied_but_still_paired(asynchronous):
    effects = []

    def alpha() -> str:
        """Allowed tool."""
        effects.append("alpha")
        return "alpha-result"

    def beta() -> str:
        """Blocked tool."""
        effects.append("beta")
        return "beta-result"

    agent = _agent([alpha, beta])
    batch = _tool_response(("call-alpha", "alpha", "{}"), ("call-beta", "beta", "{}"))
    _install(agent, [batch, _text_response("done")], asynchronous)
    messages = ChatMessages()
    events = []

    with _capture_events(_EventSink(events.append)):
        result = await _invoke(
            agent,
            "go",
            asynchronous=asynchronous,
            messages=messages,
            tool_filter={"block": ["beta"]},
        )

    assert result == "done"
    # Enforcement: the blocked tool must not run; the allowed one must.
    assert effects == ["alpha"]
    # Pairing: every model tool call still gets a result entry.
    assert _tool_output_call_ids(messages) == {"call-alpha", "call-beta"}
    # The denied call is reported through an existing blocked/error event.
    assert "call-beta" in _denied_call_ids(events)


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_allow_filter_denies_unlisted_tool(asynchronous):
    effects = []

    def alpha() -> str:
        """Allowed tool."""
        effects.append("alpha")
        return "alpha-result"

    def beta() -> str:
        """Not on the allow list."""
        effects.append("beta")
        return "beta-result"

    agent = _agent([alpha, beta])
    _install(
        agent,
        [_tool_response(("call-beta", "beta", "{}")), _text_response("done")],
        asynchronous,
    )

    result = await _invoke(
        agent,
        "go",
        asynchronous=asynchronous,
        tool_filter={"allow": ["alpha"]},
    )

    assert result == "done"
    assert effects == []


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_block_wildcard_denies_every_tool_call(asynchronous):
    effects = []

    def alpha() -> str:
        """Tool that must not run under a wildcard block."""
        effects.append("alpha")
        return "alpha-result"

    agent = _agent([alpha])
    _install(
        agent,
        [_tool_response(("call-alpha", "alpha", "{}")), _text_response("done")],
        asynchronous,
    )

    result = await _invoke(
        agent,
        "go",
        asynchronous=asynchronous,
        tool_filter={"block": "*"},
    )

    assert result == "done"
    assert effects == []


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_denied_tool_does_not_create_an_approval(asynchronous):
    effects = []

    def guarded() -> str:
        """Host-gated tool that is filtered out of this run."""
        effects.append("guarded")
        return "guarded-result"

    journal = InMemoryApprovalStore()
    agent = _agent(
        [guarded],
        checkpoint_store=InMemoryCheckpointStore(),
        approvals=AgentApprovals(journal, {"guarded": "v1"}, "p1"),
    )
    _install(
        agent,
        [_tool_response(("call-guarded", "guarded", "{}")), _text_response("done")],
        asynchronous,
    )

    result = await _invoke(
        agent,
        "go",
        asynchronous=asynchronous,
        scope=ExecutionScope(
            namespace="filter_agent",
            thread_id="thread",
            run_id="run",
            principal="user",
        ),
        tool_filter={"block": ["guarded"]},
    )

    assert result == "done"
    assert effects == []
    # A denied call must not become a pending host approval.
    assert not journal.pending("filter_agent", "thread", "run")


@pytest.mark.asyncio
async def test_caller_mutation_after_entry_cannot_expand_allow_filter():
    effects = []
    filter_value = {"allow": ["alpha"]}

    def alpha() -> str:
        """Allowed tool."""
        effects.append("alpha")
        return "alpha-result"

    def beta() -> str:
        """Tool the caller will try to smuggle in after entry."""
        effects.append("beta")
        return "beta-result"

    agent = _agent([alpha, beta])
    mutated = asyncio.Event()

    async def generate(**kwargs):
        if not mutated.is_set():
            # The run has already entered; expand the caller-owned allow list.
            filter_value["allow"].append("beta")
            mutated.set()
            return _tool_response(("call-beta", "beta", "{}"))
        return _text_response("done")

    agent.generator.aforward = generate

    result = await agent.acall(
        "go",
        tool_filter=filter_value,
        scope=ExecutionScope(
            namespace="filter_agent", thread_id="thread", run_id="run"
        ),
    )

    assert mutated.is_set()
    assert filter_value == {"allow": ["alpha", "beta"]}
    assert result == "done"
    # The effective filter was snapshotted before the mutation, so beta stays out.
    assert effects == []


@pytest.mark.asyncio
async def test_concurrent_runs_keep_their_own_filters():
    effects = []
    both_entered = asyncio.Event()
    entered = set()

    def alpha() -> str:
        """Allowed only in run-b."""
        effects.append("alpha")
        return "alpha-result"

    def beta() -> str:
        """Allowed only in run-a."""
        effects.append("beta")
        return "beta-result"

    agent = _agent([alpha, beta])

    async def generate(**kwargs):
        run_id = get_execution_scope().run_id
        if run_id in entered:
            return _text_response(f"done-{run_id}")
        entered.add(run_id)
        if len(entered) == 2:
            both_entered.set()
        await asyncio.wait_for(both_entered.wait(), timeout=5)
        # Each run tries the other run's allowed tool, which it does not permit.
        attempted = "beta" if run_id == "run-a" else "alpha"
        return _tool_response((f"call-{run_id}", attempted, "{}"))

    agent.generator.aforward = generate

    results = await asyncio.gather(
        agent.acall(
            "a",
            tool_filter={"allow": ["alpha"]},
            scope=ExecutionScope(
                namespace="filter_agent", thread_id="ta", run_id="run-a"
            ),
        ),
        agent.acall(
            "b",
            tool_filter={"allow": ["beta"]},
            scope=ExecutionScope(
                namespace="filter_agent", thread_id="tb", run_id="run-b"
            ),
        ),
    )

    assert sorted(results) == ["done-run-a", "done-run-b"]
    # Neither run may execute a tool that belongs to the other run's filter.
    assert effects == []


def test_catalog_transform_is_presentation_only():
    effects = []

    def alpha() -> str:
        """Tool kept in the presented catalog."""
        effects.append("alpha")
        return "alpha-result"

    def beta() -> str:
        """Tool hidden from the catalog but still callable by the model."""
        effects.append("beta")
        return "beta-result"

    def hide_beta(ctx):
        return replace(ctx, catalog=ctx.catalog.with_tools(("alpha",)))

    agent = _agent(
        [alpha, beta],
        hooks=[Hook(event="transform_tool_catalog", handler=hide_beta)],
    )
    agent.generator.forward = Mock(
        side_effect=[
            _tool_response(("call-beta", "beta", "{}")),
            _text_response("done"),
        ]
    )

    assert agent("go") == "done"
    # Presentation-only hooks must never deny execution.
    assert effects == ["beta"]


def test_nested_child_agent_library_does_not_inherit_parent_filter():
    child_effects = []
    parent_effects = []

    def child_tool() -> str:
        """Unrelated child library tool."""
        child_effects.append("child")
        return "child-result"

    child = _agent([child_tool], name="child_agent")
    child.generator.forward = Mock(
        side_effect=[
            _tool_response(("call-child", "child_tool", "{}")),
            _text_response("child-done"),
        ]
    )

    def call_child() -> str:
        """Run the nested child Agent without a filter."""
        parent_effects.append("call_child")
        return child(
            "child request",
            scope=ExecutionScope(
                namespace="child_agent", thread_id="child-thread", run_id="child-run"
            ),
        )

    parent = _agent([call_child])
    parent.generator.forward = Mock(
        side_effect=[
            _tool_response(("call-parent", "call_child", "{}")),
            _text_response("parent-done"),
        ]
    )

    result = parent(
        "parent request",
        scope=ExecutionScope(
            namespace="filter_agent", thread_id="parent-thread", run_id="parent-run"
        ),
        tool_filter={"allow": ["call_child"]},
    )

    assert result == "parent-done"
    assert parent_effects == ["call_child"]
    # The child library name is unrelated to the parent's allow list.
    assert child_effects == ["child"]
