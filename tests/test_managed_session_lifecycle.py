"""Managed CodingSession and AgentService resource lifecycle integration tests."""

import asyncio
import sqlite3
from unittest.mock import AsyncMock, Mock

import pytest

from msgflux.coding import CodingSession
from msgflux.models.response import ModelResponse
from msgflux.nn import Agent
from msgflux.runtime.context import get_execution_scope
from msgflux.runtime.service import AgentService, AgentSession, SQLiteServiceStore


def _response(content: str) -> ModelResponse:
    response = ModelResponse()
    response.set_response_type("text_generation")
    response.add(content)
    return response


def _managed_agent(agent_dir, answer):
    model = Mock(model_type="chat_completion")
    model.close = Mock()
    model.aclose = AsyncMock()
    agent = Agent(name="main", model=model, agent_dir=agent_dir)
    agent.generator.aforward = AsyncMock(side_effect=answer)
    return agent, model


@pytest.mark.asyncio
async def test_managed_coding_session_cancelled_close_drains_before_reopen(tmp_path):
    agent_dir = tmp_path / "agent-state"
    entered = asyncio.Event()
    shutdown_requested = asyncio.Event()
    release = asyncio.Event()
    worker_exiting = asyncio.Event()

    async def block_until_shutdown(**_kwargs):
        entered.set()
        scope = get_execution_scope()
        try:
            await scope.abort_signal.wait()
            scope.abort_signal.raise_if_aborted()
        finally:
            # Model cancellation enters its cleanup path; keep that cleanup
            # pending until the test proves stores remain open during the drain.
            shutdown_requested.set()
            try:
                await asyncio.wait_for(release.wait(), timeout=5)
            finally:
                worker_exiting.set()

    agent, model = _managed_agent(agent_dir, block_until_shutdown)
    session = CodingSession(agent, thread_id="managed-active")
    receipt = await session.prompt("remember durable input", request_id="first")
    assert receipt.request_id == "first"
    await asyncio.wait_for(entered.wait(), timeout=3)

    resources = agent._owned_threads[session.thread_id].resources
    checkpoints = resources.checkpoint_store
    journal = session._service_store
    assert (agent_dir / "runtime" / "service.sqlite3").is_file()

    closing = asyncio.create_task(session.aclose())
    try:
        await asyncio.wait_for(shutdown_requested.wait(), timeout=3)
        assert not closing.done()
        assert journal.threads()
        checkpoints.list_runs("main", session.thread_id)

        closing.cancel()
        with pytest.raises(asyncio.CancelledError):
            await closing

        release.set()
        await asyncio.wait_for(session.aclose(), timeout=5)
        assert worker_exiting.is_set()
        assert resources._closed
        with pytest.raises(sqlite3.ProgrammingError):
            journal.threads()
        with pytest.raises(sqlite3.ProgrammingError):
            checkpoints.list_runs("main", session.thread_id)

        history = {}

        async def answer_after_reopen(**kwargs):
            history["messages"] = kwargs["messages"].to_chatml()
            return _response("reopened")

        reopened_agent, reopened_model = _managed_agent(agent_dir, answer_after_reopen)
        reopened = CodingSession(reopened_agent, thread_id=session.thread_id)
        try:
            next_receipt = await reopened.prompt(
                "what was the input?", request_id="second"
            )
            settled = await asyncio.wait_for(
                reopened.wait(next_receipt.request_id), timeout=3
            )
            assert settled.status == "completed"
            assert any(
                item.get("role") == "user"
                and "remember durable input" in item.get("content", "")
                for item in history["messages"]
            )
        finally:
            await reopened.aclose()
            await reopened_agent.aclose()
            assert not reopened_model.close.called
        assert not reopened_model.aclose.await_count
    finally:
        release.set()
        if not closing.done():
            await asyncio.gather(closing, return_exceptions=True)
        await session.aclose()
        await agent.aclose()
        assert not model.close.called
        assert not model.aclose.await_count


@pytest.mark.asyncio
async def test_service_shutdown_aggregates_binding_cleanup_failure_and_keeps_borrowed_store(
    tmp_path,
):
    agent_dir = tmp_path / "agents"
    journal = SQLiteServiceStore(tmp_path / "service.sqlite3")
    service = AgentService(store=journal)
    callbacks = []
    agents = {}
    resources = {}
    diagnostics = tmp_path / "cleanup-diagnostics"
    diagnostics.mkdir()

    def factory(thread):
        thread_id = thread.thread_id

        async def answer(**_kwargs):
            return _response(f"result for {thread_id}")

        agent, model = _managed_agent(agent_dir, answer)
        agents[thread_id] = (agent, model)

        async def on_close():
            callbacks.append(thread_id)
            (diagnostics / thread_id).write_text("cleanup attempted")
            if thread_id == "cleanup-fails":
                raise RuntimeError("cleanup failed for cleanup-fails")

        return AgentSession(agent, on_close=on_close)

    service.register("main", factory)
    failing_thread = await service.open_thread("main", thread_id="cleanup-fails")
    succeeding_thread = await service.open_thread("main", thread_id="cleanup-succeeds")
    failing_receipt = await service.prompt(
        failing_thread.thread_id, "persist failing binding", request_id="one"
    )
    succeeding_receipt = await service.prompt(
        succeeding_thread.thread_id, "persist succeeding binding", request_id="two"
    )
    assert (
        await service.wait(failing_thread.thread_id, failing_receipt.request_id)
    ).status == "completed"
    assert (
        await service.wait(succeeding_thread.thread_id, succeeding_receipt.request_id)
    ).status == "completed"

    for thread_id in (failing_thread.thread_id, succeeding_thread.thread_id):
        resources[thread_id] = agents[thread_id][0]._owned_threads[thread_id].resources

    try:
        with pytest.raises(ExceptionGroup) as shutdown_error:
            await service.aclose()

        assert any(
            isinstance(error, RuntimeError)
            and "cleanup failed for cleanup-fails" in str(error)
            for error in shutdown_error.value.exceptions
        )
        assert callbacks == ["cleanup-fails", "cleanup-succeeds"]
        assert (diagnostics / "cleanup-fails").read_text() == "cleanup attempted"
        assert (diagnostics / "cleanup-succeeds").read_text() == "cleanup attempted"
        assert not resources["cleanup-fails"]._closed
        assert resources["cleanup-fails"].checkpoint_store.list_runs(
            "main", failing_thread.thread_id
        )
        assert resources["cleanup-succeeds"]._closed

        # The service borrows its journal, and the host still owns each model.
        assert {thread.thread_id for thread in journal.threads()} == {
            failing_thread.thread_id,
            succeeding_thread.thread_id,
        }
        for _agent, model in agents.values():
            assert not model.close.called
            assert not model.aclose.await_count
    finally:
        for agent, _model in agents.values():
            await agent.aclose()
        journal.close()
