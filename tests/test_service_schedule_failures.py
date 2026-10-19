"""Admission scheduling failure recovery through AgentService's public API."""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from msgflux.models.response import ModelResponse
from msgflux.nn import Agent
from msgflux.runtime.service import AgentService, AgentSession
from msgflux.runtime.service.store import SQLiteServiceStore


def _agent():
    model = Mock()
    model.model_type = "chat_completion"
    agent = Agent(name="service-schedule-failure", model=model)
    response = ModelResponse()
    response.set_response_type("text_generation")
    response.add("done")
    response.reasoning = None
    agent.generator.aforward = AsyncMock(return_value=response)
    return agent


@pytest.mark.asyncio
async def test_prompt_retry_recovers_after_worker_task_creation_failure(monkeypatch):
    store = SQLiteServiceStore()
    agent = _agent()
    service = AgentService(store=store)
    service.register("main", lambda _thread: AgentSession(agent))
    thread = await service.open_thread("main", thread_id="schedule-failure")
    lease = await service.acquire_session(thread.thread_id)
    await lease.aclose()

    original_create_task = asyncio.create_task
    failed_coroutines = []
    failure = RuntimeError("injected worker task creation failure")

    def fail_worker_once(coroutine, **kwargs):
        if coroutine.cr_code.co_qualname == "AgentService._produce":
            failed_coroutines.append(coroutine)
            monkeypatch.setattr(asyncio, "create_task", original_create_task)
            raise failure
        return original_create_task(coroutine, **kwargs)

    monkeypatch.setattr(asyncio, "create_task", fail_worker_once)
    try:
        with pytest.raises(RuntimeError) as raised:
            await service.prompt(
                thread.thread_id, "hello", request_id="retry-after-schedule-failure"
            )
        assert raised.value is failure
        assert failed_coroutines[0].cr_frame is None
        assert service._workers == {}
        assert service._producer_tasks == {}

        admitted = service.store.get(thread.thread_id, "retry-after-schedule-failure")
        assert admitted.receipt.status == "accepted"

        retried = await service.prompt(
            thread.thread_id, "hello", request_id="retry-after-schedule-failure"
        )
        assert retried.run_id == admitted.receipt.run_id
        assert (await service.wait(thread.thread_id, retried.request_id)).status == (
            "completed"
        )
        assert agent.generator.aforward.await_count == 1
    finally:
        monkeypatch.setattr(asyncio, "create_task", original_create_task)
        await service.aclose()
        await agent.aclose()
        store.close()
