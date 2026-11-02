"""Thread steering: a live run queues, a genuinely idle thread admits a prompt."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from msgflux.data.stores import InMemoryCheckpointStore
from msgflux.models.response import ModelResponse
from msgflux.models.tool_call_agg import ToolCallAggregator
from msgflux.nn import Agent
from msgflux.nn.hooks import Hook
from msgflux.runtime.service import (
    AgentSession,
    ServiceBusyError,
    ServiceRecoveryRequiredError,
    ThreadSteerResult,
)
from tests.test_agent_service import _agent, _response, _service


def _tool_call_response(tool_name):
    calls = ToolCallAggregator()
    calls.process(0, "call_1", tool_name, "{}")
    response = ModelResponse()
    response.set_response_type("tool_call")
    response.add(calls)
    return response


def _tool_call_agent(name, tool, answer):
    model = Mock()
    model.model_type = "chat_completion"
    agent = Agent(name=name, model=model, tools=[tool])
    agent.generator.aforward = AsyncMock(side_effect=answer)
    return agent


@pytest.mark.asyncio
async def test_steer_on_idle_thread_admits_exactly_one_new_prompt():
    checkpoints = InMemoryCheckpointStore()
    agent = _agent("idle-steer")
    agent.checkpoint_store = checkpoints
    service = _service(
        lambda _thread_id: AgentSession(agent, checkpoint_store=checkpoints)
    )
    thread = await service.open_thread("main", thread_id="idle-steer")
    try:
        result = await service.steer_thread(thread.thread_id, "follow up")

        assert isinstance(result, ThreadSteerResult)
        assert result.status == "started"
        assert result.thread_id == thread.thread_id
        assert result.request_id and result.notification_id is None

        settled = await service.wait(thread.thread_id, result.request_id)
        assert settled.status == "completed"
        assert settled.run_id == result.run_id
        assert [run.run_id for run in await service.runs(thread.thread_id)] == [
            result.run_id
        ]
        assert agent.generator.aforward.await_count == 1
        assert (await service.run_result(thread.thread_id, result.run_id)).output == (
            "done"
        )
    finally:
        await service.aclose()


@pytest.mark.parametrize("status", ["completed", "interrupted"])
@pytest.mark.parametrize("content", ["hello", "  spaced  ", "multi\nline"])
@pytest.mark.asyncio
async def test_steer_starts_prompt_after_settled_runs(status, content):
    checkpoints = InMemoryCheckpointStore()
    agent = _agent("idle-after-run")
    agent.checkpoint_store = checkpoints
    service = _service(
        lambda _thread_id: AgentSession(agent, checkpoint_store=checkpoints)
    )
    thread = await service.open_thread("main", thread_id="idle-after-run")
    checkpoints.save_state(
        agent.get_module_name(), thread.thread_id, "settled-run", {"status": status}
    )
    try:
        result = await service.steer_thread(thread.thread_id, content)

        assert result.status == "started"
        assert result.run_id != "settled-run"
        settled = await service.wait(thread.thread_id, result.request_id)
        assert settled.status == "completed"
    finally:
        await service.aclose()


@pytest.mark.asyncio
async def test_steer_on_active_thread_queues_same_run_without_admission():
    tool_started, finish_tool = asyncio.Event(), asyncio.Event()
    observed_model_inputs = []

    async def wait_for_release() -> str:
        tool_started.set()
        await finish_tool.wait()
        return "tool finished"

    async def generate(**_kwargs):
        observed_model_inputs.append(_kwargs["messages"].to_chatml())
        if len(observed_model_inputs) == 1:
            return _tool_call_response("wait_for_release")
        return _response("done")

    checkpoints = InMemoryCheckpointStore()
    agent = _tool_call_agent("active-steer", wait_for_release, generate)
    agent.checkpoint_store = checkpoints
    service = _service(
        lambda _thread_id: AgentSession(agent, checkpoint_store=checkpoints)
    )
    thread = await service.open_thread("main", thread_id="active-steer")
    try:
        receipt = await service.prompt(thread.thread_id, "start", request_id="req-1")
        await asyncio.wait_for(tool_started.wait(), timeout=5)
        result = await service.steer_thread(thread.thread_id, "new constraint")

        assert result.status == "queued"
        assert result.run_id == receipt.run_id
        assert result.notification_id and result.request_id is None

        finish_tool.set()
        settled = await service.wait(thread.thread_id, "req-1")
        assert settled.status == "completed"
        assert "new constraint" in str(observed_model_inputs[1])
        # The queued message never became a second admission or run.
        assert [run.run_id for run in await service.runs(thread.thread_id)] == [
            receipt.run_id
        ]
        # The run-scoped entry point keeps its strict identity requirement.
        with pytest.raises(ServiceBusyError, match="not active"):
            await service.steer(thread.thread_id, receipt.run_id, "after settlement")
    finally:
        finish_tool.set()
        await service.aclose()


@pytest.mark.asyncio
async def test_concurrent_steers_on_idle_thread_start_at_most_one_run():
    release = asyncio.Event()

    async def answer(**_kwargs):
        await release.wait()
        return _response("done")

    checkpoints = InMemoryCheckpointStore()
    agent = _agent("racing-steer")
    agent.checkpoint_store = checkpoints
    agent.generator.aforward = AsyncMock(side_effect=answer)
    service = _service(
        lambda _thread_id: AgentSession(agent, checkpoint_store=checkpoints)
    )
    thread = await service.open_thread("main", thread_id="racing-steer")
    try:
        first, second = await asyncio.gather(
            service.steer_thread(thread.thread_id, "first"),
            service.steer_thread(thread.thread_id, "second"),
        )

        assert {first.status, second.status} == {"started", "queued"}
        started = first if first.status == "started" else second
        queued = second if first.status == "started" else first
        assert queued.run_id == started.run_id

        release.set()
        settled = await service.wait(thread.thread_id, started.request_id)
        assert settled.status == "completed"
        # One admission only: the queued content rode the started run's inbox.
        assert [run.run_id for run in await service.runs(thread.thread_id)] == [
            started.run_id
        ]
    finally:
        release.set()
        await service.aclose()


@pytest.mark.asyncio
async def test_steer_never_reports_queued_once_finalization_sealed_the_inbox():
    hook_entered, release_hook = asyncio.Event(), asyncio.Event()

    async def after_run_end(_context):
        hook_entered.set()
        await release_hook.wait()

    checkpoints = InMemoryCheckpointStore()
    agent = _agent("sealed-steer")
    agent.checkpoint_store = checkpoints
    agent.register_lifecycle_hook(
        "after_run_end", Hook(event="after_run_end", handler=after_run_end)
    )
    service = _service(
        lambda _thread_id: AgentSession(agent, checkpoint_store=checkpoints)
    )
    thread = await service.open_thread("main", thread_id="sealed-steer")
    try:
        receipt = await service.prompt(thread.thread_id, "finish", request_id="req-1")
        await asyncio.wait_for(hook_entered.wait(), timeout=5)

        with pytest.raises(ServiceBusyError, match="no longer accepts steering"):
            await service.steer_thread(thread.thread_id, "too late")
        assert agent.generator.aforward.await_count == 1

        release_hook.set()
        assert (await service.wait(thread.thread_id, "req-1")).status == "completed"
        assert [run.run_id for run in await service.runs(thread.thread_id)] == [
            receipt.run_id
        ]
        # Only after the run settles may a retry become a new prompt.
        started = await service.steer_thread(thread.thread_id, "next turn")
        assert started.status == "started"
        assert started.run_id != receipt.run_id
        assert (await service.wait(thread.thread_id, started.request_id)).status == (
            "completed"
        )
    finally:
        release_hook.set()
        await service.aclose()


@pytest.mark.asyncio
async def test_steer_while_run_drains_interruption_is_retryable_busy():
    entered, release = asyncio.Event(), asyncio.Event()

    async def delayed(*_args, **_kwargs):
        entered.set()
        await release.wait()
        return _response()

    agent = _agent("draining-steer")
    agent.generator.aforward = AsyncMock(side_effect=delayed)
    service = _service(lambda _thread_id: AgentSession(agent))
    thread = await service.open_thread("main", thread_id="draining-steer")
    try:
        receipt = await service.prompt(thread.thread_id, "hello", request_id="req-1")
        await asyncio.wait_for(entered.wait(), timeout=5)
        assert await service.interrupt(thread.thread_id, receipt.run_id) is True

        with pytest.raises(ServiceBusyError, match="interruption"):
            await service.steer_thread(thread.thread_id, "too late")

        release.set()
        settled = await service.wait(thread.thread_id, "req-1")
        assert settled.status == "interrupted"
        assert service.store.latest_admission(thread.thread_id).receipt.run_id == (
            receipt.run_id
        )
    finally:
        release.set()
        await service.aclose()


@pytest.mark.asyncio
async def test_steer_refuses_ambiguous_thread_instead_of_choosing_a_run():
    release = asyncio.Event()

    async def delayed(*_args, **_kwargs):
        await release.wait()
        return _response()

    agent = _agent("ambiguous-steer")
    agent.generator.aforward = AsyncMock(side_effect=delayed)
    service = _service(lambda _thread_id: AgentSession(agent))
    thread = await service.open_thread("main", thread_id="ambiguous-steer")
    try:
        receipt = await service.prompt(thread.thread_id, "hello", request_id="req-1")
        # An unexpected second live worker must never be steered silently.
        ghost_task = asyncio.ensure_future(asyncio.sleep(3600))
        service._workers[(thread.thread_id, "ghost")] = SimpleNamespace(
            receipt=SimpleNamespace(thread_id=thread.thread_id, run_id="ghost-run"),
            task=ghost_task,
        )
        try:
            with pytest.raises(ServiceBusyError, match="more than one active run"):
                await service.steer_thread(thread.thread_id, "hello")
        finally:
            service._workers.pop((thread.thread_id, "ghost"))
            ghost_task.cancel()
            await asyncio.gather(ghost_task, return_exceptions=True)

        release.set()
        settled = await service.wait(thread.thread_id, "req-1")
        assert settled.status == "completed"
        assert settled.run_id == receipt.run_id
    finally:
        release.set()
        await service.aclose()


@pytest.mark.asyncio
async def test_steer_on_unrecovered_paused_thread_still_requires_recovery():
    checkpoints = InMemoryCheckpointStore()
    agent = _agent("paused-steer")
    agent.checkpoint_store = checkpoints
    service = _service(
        lambda _thread_id: AgentSession(agent, checkpoint_store=checkpoints)
    )
    thread = await service.open_thread("main", thread_id="paused-steer")
    checkpoints.save_state(
        agent.get_module_name(), thread.thread_id, "paused-run", {"status": "paused"}
    )
    try:
        with pytest.raises(ServiceRecoveryRequiredError):
            await service.steer_thread(thread.thread_id, "retry")
        assert agent.generator.aforward.await_count == 0
        assert service.store.latest_admission(thread.thread_id) is None
    finally:
        await service.aclose()


@pytest.mark.asyncio
async def test_steer_on_unknown_thread_loads_no_agent():
    loaded = []

    def factory(thread):
        loaded.append(thread)
        return AgentSession(_agent())

    service = _service(factory)
    try:
        with pytest.raises(KeyError):
            await service.steer_thread("missing", "hello")
        assert loaded == []
    finally:
        await service.aclose()


@pytest.mark.parametrize(
    "content, error",
    [("", ValueError), ("   ", ValueError), ("\n\t", ValueError), (None, TypeError)],
)
@pytest.mark.asyncio
async def test_steer_requires_non_empty_text_content(content, error):
    loaded = []

    def factory(thread):
        loaded.append(thread)
        return AgentSession(_agent())

    service = _service(factory)
    thread = await service.open_thread("main", thread_id="invalid-steer")
    try:
        with pytest.raises(error):
            await service.steer_thread(thread.thread_id, content)
        assert loaded == []
        assert service.store.latest_admission(thread.thread_id) is None
    finally:
        await service.aclose()
