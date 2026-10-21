"""Run finalization failures must retain the execution failure they mask."""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from msgflux.models.response import ModelResponse
from msgflux.nn import Agent
from msgflux.runtime.service import (
    AgentService,
    AgentSession,
    ServiceRecoveryRequiredError,
    SQLiteServiceStore,
)
from msgflux.runtime.service import api as service_api


def _response():
    response = ModelResponse()
    response.set_response_type("text_generation")
    response.add("unused")
    return response


async def _service(name, model_side_effect):
    agent = Agent(name=name, model=Mock(model_type="chat_completion"))
    agent.generator.aforward = AsyncMock(side_effect=model_side_effect)
    store = SQLiteServiceStore()
    service = AgentService(store=store)
    service.register("main", lambda _thread: AgentSession(agent))
    thread = await service.open_thread("main", thread_id=name)
    return agent, store, service, thread


async def _close(agent, store, service, finish):
    store.finish = finish
    await service.aclose()
    store.close()
    await agent.aclose()


@pytest.mark.asyncio
async def test_finish_failure_preserves_execution_error_and_releases_session_lease():
    execution_error = RuntimeError("model execution failed")
    finish_error = OSError("journal finish failed")
    agent, store, service, thread = await _service("finish-error-audit", None)
    entered = asyncio.Event()
    release = asyncio.Event()

    async def fail_model(**_kwargs):
        entered.set()
        await release.wait()
        raise execution_error

    agent.generator.aforward = AsyncMock(side_effect=fail_model)
    original_finish = store.finish
    store.finish = Mock(side_effect=finish_error)
    wait_task = None

    try:
        await service.prompt(thread.thread_id, "run", request_id="request")
        wait_task = asyncio.create_task(service.wait(thread.thread_id, "request"))
        await asyncio.wait_for(entered.wait(), timeout=5)
        release.set()
        with pytest.raises(ExceptionGroup) as raised:
            await asyncio.wait_for(wait_task, timeout=5)

        assert any(error is execution_error for error in raised.value.exceptions)
        assert any(error is finish_error for error in raised.value.exceptions)
        assert store.get(thread.thread_id, "request").receipt.status == "running"
        assert service._cache._entries[thread.thread_id].pins == 0
        await asyncio.sleep(0)
        assert not service._workers
        assert not service._producer_tasks
    finally:
        release.set()
        if wait_task is not None and not wait_task.done():
            await asyncio.gather(wait_task, return_exceptions=True)
        await _close(agent, store, service, original_finish)


@pytest.mark.asyncio
async def test_cancellation_remains_cancellation_when_journal_settlement_fails(
    monkeypatch,
):
    cancellation = asyncio.CancelledError("worker cancelled")
    finish_error = OSError("journal finish failed")
    agent, store, service, thread = await _service("cancel-finish-audit", None)
    logged_error = Mock()
    monkeypatch.setattr(service_api.logger, "error", logged_error)
    entered = asyncio.Event()
    release = asyncio.Event()

    async def cancel_execution(_record, _worker):
        entered.set()
        await release.wait()
        raise cancellation

    service._execute = cancel_execution
    original_finish = store.finish
    store.finish = Mock(side_effect=finish_error)

    try:
        await service.prompt(thread.thread_id, "run", request_id="request")
        wait_task = asyncio.create_task(service.wait(thread.thread_id, "request"))
        await asyncio.wait_for(entered.wait(), timeout=5)
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(wait_task, timeout=5)

        assert wait_task.cancelled()
        logged_error.assert_called_once()
        assert logged_error.call_args.kwargs["exc_info"] is finish_error
        assert store.get(thread.thread_id, "request").receipt.status == "running"
        assert service._cache._entries[thread.thread_id].pins == 0
        await asyncio.sleep(0)
        assert not service._workers
        assert not service._producer_tasks
    finally:
        release.set()
        await _close(agent, store, service, original_finish)


@pytest.mark.asyncio
async def test_finish_only_failure_is_not_reclassified_or_marked_settled():
    finish_error = OSError("journal finish failed")
    agent, store, service, thread = await _service("finish-only-audit", None)
    entered = asyncio.Event()
    release = asyncio.Event()

    async def succeed_model(**_kwargs):
        entered.set()
        await release.wait()
        return _response()

    agent.generator.aforward = AsyncMock(side_effect=succeed_model)
    original_finish = store.finish
    store.finish = Mock(side_effect=finish_error)
    wait_task = None

    try:
        await service.prompt(thread.thread_id, "run", request_id="request")
        wait_task = asyncio.create_task(service.wait(thread.thread_id, "request"))
        await asyncio.wait_for(entered.wait(), timeout=5)
        release.set()
        with pytest.raises(OSError) as raised:
            await asyncio.wait_for(wait_task, timeout=5)

        assert raised.value is finish_error
        assert raised.value.__cause__ is None
        assert store.get(thread.thread_id, "request").receipt.status == "running"
        assert service._cache._entries[thread.thread_id].pins == 0
    finally:
        release.set()
        if wait_task is not None and not wait_task.done():
            await asyncio.gather(wait_task, return_exceptions=True)
        await _close(agent, store, service, original_finish)


@pytest.mark.asyncio
async def test_claim_failure_is_logged_without_settling_accepted_admission(monkeypatch):
    claim_error = OSError("journal claim failed")
    storage_cause = RuntimeError("underlying storage failure")
    claim_error.__cause__ = storage_cause
    agent, store, service, thread = await _service("claim-error-audit", None)
    original_claim = store.claim
    original_finish = store.finish
    logged_error = Mock()
    logged = asyncio.Event()
    logged_error.side_effect = lambda *_args, **_kwargs: logged.set()
    monkeypatch.setattr(service_api.logger, "error", logged_error)
    store.claim = Mock(side_effect=claim_error)
    store.finish = Mock(wraps=original_finish)

    try:
        receipt = await service.prompt(thread.thread_id, "run", request_id="request")
        assert receipt.status == "accepted"
        await asyncio.wait_for(logged.wait(), timeout=5)

        logged_error.assert_called_once()
        assert logged_error.call_args.kwargs["exc_info"] is claim_error
        assert claim_error.__cause__ is storage_cause
        store.finish.assert_not_called()
        assert store.get(thread.thread_id, "request").receipt.status == "accepted"
        assert service._cache._entries[thread.thread_id].pins == 0
        with pytest.raises(ServiceRecoveryRequiredError):
            await asyncio.wait_for(service.wait(thread.thread_id, "request"), timeout=5)
    finally:
        store.claim = original_claim
        await _close(agent, store, service, original_finish)


@pytest.mark.asyncio
async def test_unclaimed_worker_exit_does_not_settle_or_report_storage_failure(
    monkeypatch,
):
    agent, store, service, thread = await _service("claim-contended-audit", None)
    original_claim = store.claim
    original_finish = store.finish
    logged_error = Mock()
    monkeypatch.setattr(service_api.logger, "error", logged_error)
    store.claim = Mock(return_value=False)
    store.finish = Mock(wraps=original_finish)

    try:
        receipt = await service.prompt(thread.thread_id, "run", request_id="request")
        assert receipt.status == "accepted"
        with pytest.raises(ServiceRecoveryRequiredError):
            await asyncio.wait_for(service.wait(thread.thread_id, "request"), timeout=5)

        logged_error.assert_not_called()
        store.finish.assert_not_called()
        assert store.get(thread.thread_id, "request").receipt.status == "accepted"
        assert service._cache._entries[thread.thread_id].pins == 0
    finally:
        store.claim = original_claim
        await _close(agent, store, service, original_finish)
