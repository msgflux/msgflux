"""Shutdown ordering for Agent-owned durable thread resources."""

import asyncio
import stat
from threading import Event
from unittest.mock import AsyncMock, Mock

import msgflux as mf
import pytest

from msgflux.models.response import ModelResponse
from msgflux.models.tool_call_agg import ToolCallAggregator
from msgflux.nn import Agent
from msgflux.runtime.service import AgentService, AgentSession, SQLiteServiceStore
from msgflux.tools.builtin import AgentTool


def _text(content: str) -> ModelResponse:
    response = ModelResponse()
    response.set_response_type("text_generation")
    response.add(content)
    return response


def _tool_call(name: str, arguments: str) -> ModelResponse:
    calls = ToolCallAggregator()
    calls.process(0, "call-1", name, arguments)
    response = ModelResponse()
    response.set_response_type("tool_call")
    response.add(calls)
    return response


def _agent(name, agent_dir, answer):
    model = Mock()
    model.model_type = "chat_completion"
    result = Agent(name=name, model=model, agent_dir=agent_dir)
    result.generator.aforward = AsyncMock(side_effect=answer)
    return result


@pytest.mark.asyncio
async def test_service_shutdown_interrupts_nested_background_work_before_host_closes_dependencies(
    tmp_path,
):
    entered = Event()
    child_model_called = Event()
    interrupted = Event()
    host_closed = Event()
    session = None

    @mf.tool_config(runtime_inputs=("handle",))
    def wait_for_shutdown(handle):
        task = handle.get_task()
        entered.set()
        while not task.is_interrupt_requested():
            entered.wait(0.01)
        interrupted.set()
        task.raise_if_interrupted()

    def make_session(_thread):
        nonlocal session

        child_calls = 0

        def child_answer(**_kwargs):
            nonlocal child_calls
            child_calls += 1
            child_model_called.set()
            if child_calls == 1:
                return _tool_call("wait_for_shutdown", "{}")
            return _text("done")

        child = _agent("child", None, child_answer)
        child.generator.forward = Mock(side_effect=child_answer)
        child.tool_library.add(wait_for_shutdown)

        root_calls = 0

        async def root_answer(**_kwargs):
            nonlocal root_calls
            root_calls += 1
            if root_calls == 1:
                return _tool_call(
                    "agent",
                    '{"name":"child","message":"work","run_in_background":true}',
                )
            return _text("delegated")

        root = _agent("root", tmp_path / "agent-state", root_answer)
        root.tool_library.add(mf.tool_config(allow_background=True)(AgentTool()))
        root.tool_library.add(child)

        async def host_close():
            # The callback must run after cooperative cancellation settles,
            # while the owned task store is still open.
            assert interrupted.is_set()
            task_store = root._owned_threads[_thread.thread_id].resources.task_store
            assert task_store.list()
            host_closed.set()

        session = AgentSession(root, on_close=host_close)
        return session

    service = AgentService(store=SQLiteServiceStore())
    service.register("root", make_session)
    thread = await service.open_thread("root")
    receipt = await service.prompt(
        thread.thread_id, "delegate", request_id="request-shutdown"
    )
    try:
        settled = await asyncio.wait_for(
            service.wait(thread.thread_id, receipt.request_id), 5
        )
        assert settled.status == "completed"
        assert await asyncio.to_thread(child_model_called.wait, 2)
        assert await asyncio.to_thread(entered.wait, 2)
    except TimeoutError:
        raise AssertionError(root_task_state(service, thread.thread_id)) from None
    finally:
        await asyncio.wait_for(service.aclose(), 5)
    assert interrupted.is_set()
    assert host_closed.is_set()
    assert session is not None

    # The thread store was closed after the host callback; persisted state can
    # be reopened and confirms that the task reached a terminal state.
    from msgflux.runtime.agent_resources import AgentResources

    resources = AgentResources(tmp_path / "agent-state").bind(
        thread.thread_id, namespace="root"
    )
    try:
        task = resources.task_store.list()[0]
        assert task.status in {"completed", "interrupted", "failed"}
    finally:
        resources.close()


def test_service_journal_is_opened_with_private_permissions(tmp_path):
    from msgflux.runtime.agent_resources import AgentResources

    resources = AgentResources(tmp_path / "agent-state")
    store = resources.service_store()
    try:
        journal = resources.agent_dir / "runtime" / "service.sqlite3"
        assert journal.is_file()
        assert stat.S_IMODE(journal.stat().st_mode) == 0o600
        assert stat.S_IMODE(journal.parent.stat().st_mode) == 0o700
    finally:
        store.close()


def root_task_state(service, thread_id):
    session = service._sessions[thread_id]
    resources = session.agent._owned_threads[thread_id].resources
    return [task.to_dict() for task in resources.task_store.list()]


@pytest.mark.asyncio
async def test_thread_close_retry_does_not_repeat_successful_host_callback(tmp_path):
    agent = _agent("retry-close", tmp_path / "retry-close", lambda **_kwargs: _text())
    thread_id = "retry-close-thread"
    resources = agent._bind_resources(thread_id)
    original_close = resources.checkpoint_store.close
    store_close_calls = 0
    callback_calls = 0

    def fail_first_store_close():
        nonlocal store_close_calls
        store_close_calls += 1
        if store_close_calls == 1:
            raise RuntimeError("checkpoint close failed")
        original_close()

    def host_close():
        nonlocal callback_calls
        callback_calls += 1

    resources.checkpoint_store.close = fail_first_store_close
    try:
        with pytest.raises(ExceptionGroup, match="resource stores"):
            await agent._close_thread_resources(thread_id, before_close=host_close)
        assert callback_calls == 1
        assert not resources._closed

        await agent._close_thread_resources(thread_id, before_close=host_close)
        assert callback_calls == 1
        assert store_close_calls == 2
        assert resources._closed
    finally:
        if not resources._closed:
            resources.close()


@pytest.mark.asyncio
async def test_failed_host_close_callback_is_quarantined_without_silent_retry(
    tmp_path,
):
    agent = _agent(
        "failed-host-close",
        tmp_path / "failed-host-close",
        lambda **_kwargs: _text(),
    )
    thread_id = "failed-host-close-thread"
    resources = agent._bind_resources(thread_id)
    callback_calls = 0

    def host_close():
        nonlocal callback_calls
        callback_calls += 1
        raise RuntimeError("host cleanup requires review")

    try:
        with pytest.raises(RuntimeError, match="requires review"):
            await agent._close_thread_resources(thread_id, before_close=host_close)
        with pytest.raises(RuntimeError, match="requires review"):
            await agent._close_thread_resources(thread_id, before_close=host_close)
        assert callback_calls == 1
        assert not resources._closed
    finally:
        resources.close()
