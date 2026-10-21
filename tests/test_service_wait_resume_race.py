"""Waiting follows a replacement worker for the same admission."""

import asyncio
from unittest.mock import Mock

import pytest

from msgflux.data.stores import SQLiteCheckpointStore
from msgflux.exceptions import TaskPauseRequestedError
from msgflux.models.response import ModelResponse
from msgflux.nn import Agent
from msgflux.runtime.service import AgentService, AgentSession, SQLiteServiceStore


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["completed", "paused"])
async def test_wait_follows_run_resumed_before_receipt_is_observed(tmp_path, outcome):
    first_entered, pause, second_entered, finish = (asyncio.Event() for _ in range(4))
    checkpoints = SQLiteCheckpointStore(tmp_path / "checkpoints.sqlite3")
    journal = SQLiteServiceStore(tmp_path / "service.sqlite3")
    agent = Agent(
        name="wait-resume-agent",
        model=Mock(model_type="chat_completion"),
        checkpoint_store=checkpoints,
    )
    calls = 0

    async def answer(**_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            first_entered.set()
            await asyncio.wait_for(pause.wait(), timeout=5)
            raise TaskPauseRequestedError(message="Pause before resuming")
        second_entered.set()
        await asyncio.wait_for(finish.wait(), timeout=5)
        if outcome == "paused":
            raise TaskPauseRequestedError(message="Pause again")
        response = ModelResponse()
        response.set_response_type("text_generation")
        response.add("resumed answer")
        return response

    agent.generator.aforward = answer
    service = AgentService(store=journal, cache_policy=None)
    service.register("main", lambda _thread: AgentSession(agent))
    resumptions = []
    waiter = None
    try:
        thread = await service.open_thread("main")
        receipt = await service.prompt(thread.thread_id, "start", request_id="first")
        await asyncio.wait_for(first_entered.wait(), timeout=3)
        worker = service._workers[(thread.thread_id, receipt.request_id)]

        # Order the public resume before wait's completion callback reads the
        # journal. No admission or checkpoint state is fabricated by the test.
        def resume_on_completion(_task):
            resumptions.append(
                asyncio.create_task(
                    service.resume(thread.thread_id, receipt.request_id)
                )
            )

        worker.task.add_done_callback(resume_on_completion)
        waiter = asyncio.create_task(service.wait(thread.thread_id, receipt.request_id))
        await asyncio.sleep(0)
        pause.set()
        await asyncio.wait_for(second_entered.wait(), timeout=3)
        assert not waiter.done(), waiter.result()
        finish.set()
        settled = await asyncio.wait_for(waiter, timeout=3)
        assert settled.status == outcome
        assert settled.run_id == receipt.run_id
        assert calls == 2
        checkpoint = checkpoints.load_state(
            agent.get_module_name(), thread.thread_id, receipt.run_id
        )
        assert checkpoint["status"] == outcome
    finally:
        pause.set()
        finish.set()
        await asyncio.wait_for(service.aclose(), timeout=5)
        pending = [*resumptions, *([waiter] if waiter is not None else [])]
        if pending:
            await asyncio.wait_for(
                asyncio.gather(*pending, return_exceptions=True), timeout=3
            )
        journal.close()
        checkpoints.close()
