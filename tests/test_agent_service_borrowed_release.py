"""Borrowed TaskStore bindings prevent unsafe AgentSession release."""

from __future__ import annotations

import asyncio
from threading import Event
from unittest.mock import AsyncMock, Mock

import msgflux as mf
import pytest

from msgflux.models.response import ModelResponse
from msgflux.models.tool_call_agg import ToolCallAggregator
from msgflux.nn import Agent
from msgflux.runtime.service import (
    AgentService,
    AgentSession,
    ServiceBusyError,
    SQLiteServiceStore,
)
from msgflux.tasks import InMemoryTaskStore
from msgflux.tools.builtin import AgentTool


def _text(content="done"):
    response = ModelResponse()
    response.set_response_type("text_generation")
    response.add(content)
    return response


def _tool_call(name, arguments):
    calls = ToolCallAggregator()
    calls.process(0, "call-1", name, arguments)
    response = ModelResponse()
    response.set_response_type("tool_call")
    response.add(calls)
    return response


def _agent(name):
    model = Mock(model_type="chat_completion")
    model.close = Mock()
    model.aclose = AsyncMock()
    return Agent(name=name, model=model)


class BorrowedTaskStore(InMemoryTaskStore):
    def __init__(self):
        super().__init__()
        self.close_calls = 0

    def close(self):
        self.close_calls += 1


@pytest.mark.asyncio
async def test_borrowed_task_store_blocks_release_until_background_child_finishes():
    child_entered = Event()
    finish_child = Event()
    task_store = BorrowedTaskStore()
    host_close_calls = 0

    @mf.tool_config(runtime_inputs=("handle",))
    def wait_for_finish(handle):
        task = handle.get_task()
        child_entered.set()
        while not finish_child.wait(0.01):
            task.raise_if_interrupted()
        return "child finished"

    child = _agent("borrowed-child")
    child.tool_library.add(wait_for_finish)
    child.generator.aforward = AsyncMock(
        side_effect=[_tool_call("wait_for_finish", "{}"), _text("finished")]
    )

    root = _agent("borrowed-root")
    root.tool_library.add(mf.tool_config(allow_background=True)(AgentTool()))
    root.tool_library.add(child)
    root.generator.aforward = AsyncMock(
        side_effect=[
            _tool_call(
                "agent",
                '{"name":"borrowed-child","message":"background",'
                '"run_in_background":true}',
            ),
            _text("delegated"),
        ]
    )

    def host_close():
        nonlocal host_close_calls
        host_close_calls += 1

    service = AgentService(store=SQLiteServiceStore())
    service.register(
        "borrowed-root",
        lambda _thread: AgentSession(root, task_store=task_store, on_close=host_close),
    )
    thread = await service.open_thread("borrowed-root", thread_id="borrowed-child")
    try:
        receipt = await service.prompt(
            thread.thread_id, "delegate", request_id="borrowed-request"
        )
        assert (
            await asyncio.wait_for(
                service.wait(thread.thread_id, receipt.request_id), 5
            )
        ).status == "completed"
        assert await asyncio.to_thread(child_entered.wait, 3)
        assert task_store.has_unfinished_for_thread(thread_id=thread.thread_id)

        with pytest.raises(ServiceBusyError):
            await service.release_session(thread.thread_id)
        assert host_close_calls == 0
        assert task_store.close_calls == 0

        finish_child.set()
        deadline = asyncio.get_running_loop().time() + 5
        while (
            task_store.has_unfinished_for_thread(thread_id=thread.thread_id)
            and asyncio.get_running_loop().time() < deadline
        ):
            await asyncio.sleep(0.01)
        assert not task_store.has_unfinished_for_thread(thread_id=thread.thread_id)

        assert await service.release_session(thread.thread_id) is True
        assert host_close_calls == 1
        assert task_store.close_calls == 0
    finally:
        finish_child.set()
        await service.aclose()
        await root.aclose()
        await child.aclose()


@pytest.mark.asyncio
async def test_borrowed_store_without_quiescence_query_fails_closed():
    delegate = InMemoryTaskStore()

    class UnknownTaskStore:
        def __getattr__(self, name):
            if name == "has_unfinished_for_thread":
                raise AttributeError(name)
            return getattr(delegate, name)

    task_store = UnknownTaskStore()
    agent = _agent("unknown-store")
    agent.generator.aforward = AsyncMock(return_value=_text())
    host_close_calls = 0

    def host_close():
        nonlocal host_close_calls
        host_close_calls += 1

    service = AgentService(store=SQLiteServiceStore())
    service.register(
        "unknown-store",
        lambda _thread: AgentSession(agent, task_store=task_store, on_close=host_close),
    )
    thread = await service.open_thread("unknown-store", thread_id="unknown-store")
    try:
        receipt = await service.prompt(thread.thread_id, "hello", request_id="request")
        assert (await service.wait(thread.thread_id, receipt.request_id)).status == (
            "completed"
        )
        with pytest.raises(ServiceBusyError, match="confirm thread quiescence"):
            await service.release_session(thread.thread_id)
        assert host_close_calls == 0
    finally:
        await service.aclose()
        await agent.aclose()
