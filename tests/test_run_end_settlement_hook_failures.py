"""A failing pre-end hook must not replace an existing terminal error."""

import asyncio

import pytest

from msgflux.data.stores import InMemoryCheckpointStore
from msgflux.exceptions import (
    AbortRequestedError,
    TaskInterruptRequestedError,
    TaskPauseRequestedError,
)
from msgflux.nn import Agent
from msgflux.nn.hooks import Hook
from msgflux.runtime import ExecutionScope


class _FailingModel:
    model_type = "chat_completion"

    def __init__(self, error):
        self.error = error

    def __call__(self, **_kwargs):
        raise self.error

    async def acall(self, **_kwargs):
        raise self.error


def _primary_error(kind):
    if kind == "failed":
        return RuntimeError("primary model failure")
    if kind == "interrupted":
        return TaskInterruptRequestedError("task", "primary interruption")
    if kind == "aborted":
        return AbortRequestedError("primary abort")
    if kind == "cancelled":
        return asyncio.CancelledError("primary cancellation")
    return TaskPauseRequestedError("task", "primary pause")


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "kind", ["failed", "interrupted", "paused", "aborted", "cancelled"]
)
async def test_before_run_end_error_during_terminal_settlement_is_secondary(
    asynchronous, kind
):
    primary = _primary_error(kind)
    original_cause = LookupError("original cause")
    primary.__cause__ = original_cause
    secondary = RuntimeError("secondary before_run_end failure")
    expected_status = "interrupted" if kind in {"aborted", "cancelled"} else kind
    expected_type = TaskInterruptRequestedError if kind == "aborted" else type(primary)
    before_calls = []
    after_calls = []
    store = InMemoryCheckpointStore()

    def fail_before_run_end(context):
        before_calls.append(context.outcome)
        raise secondary

    def observe_after_run_end(context):
        after_calls.append((context.outcome, context.error))

    agent = Agent(
        name="agent",
        model=_FailingModel(primary),
        checkpoint_store=store,
        hooks=[
            Hook(event="before_run_end", handler=fail_before_run_end),
            Hook(event="after_run_end", handler=observe_after_run_end),
        ],
    )
    scope = ExecutionScope(
        thread_id="thread",
        namespace="agent",
        run_id=f"{kind}-run",
    )

    try:
        with pytest.raises(expected_type) as raised:
            if asynchronous:
                await agent.acall("hello", scope=scope)
            else:
                agent("hello", scope=scope)

        if kind == "aborted":
            assert raised.value.__cause__ is primary
        else:
            assert raised.value is primary
        assert primary.__cause__ is original_cause
        assert any(
            "secondary before_run_end failure" in note for note in primary.__notes__
        )
        assert before_calls == [expected_status]
        assert after_calls == [(expected_status, primary)]
        state = store.load_state("agent", "thread", scope.run_id)
        assert state["status"] == expected_status
    finally:
        await agent.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "secondary",
    [
        pytest.param(AbortRequestedError("secondary abort"), id="abort"),
        pytest.param(
            TaskInterruptRequestedError("task", "secondary interrupt"),
            id="interrupt",
        ),
        pytest.param(TaskPauseRequestedError("task", "secondary pause"), id="pause"),
        pytest.param(asyncio.CancelledError("secondary cancellation"), id="cancel"),
    ],
)
async def test_before_run_end_control_exception_during_settlement_still_propagates(
    asynchronous, secondary
):
    primary = RuntimeError("primary model failure")
    store = InMemoryCheckpointStore()

    def raise_secondary(_context):
        raise secondary

    agent = Agent(
        name="agent",
        model=_FailingModel(primary),
        checkpoint_store=store,
        hooks=[Hook(event="before_run_end", handler=raise_secondary)],
    )
    scope = ExecutionScope(
        thread_id="thread",
        namespace="agent",
        run_id="secondary-control",
    )

    try:
        with pytest.raises(type(secondary)) as raised:
            if asynchronous:
                await agent.acall("hello", scope=scope)
            else:
                agent("hello", scope=scope)

        assert raised.value is secondary
    finally:
        await agent.aclose()
