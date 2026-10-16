"""Manual compaction uses service ownership and the same local/HTTP contract."""

import asyncio
from copy import deepcopy

import httpx2
import pytest

from msgflux.chat_messages import ChatMessages
from msgflux.nn import CompactionExtension
from msgflux.runtime import CompactionResult
from msgflux.runtime.service import (
    AgentService,
    AgentSession,
    ServiceBusyError,
    SQLiteServiceStore,
)
from msgflux.runtime.service.http.client import (
    AgentServiceClient,
    AgentServiceHTTPError,
)
from msgflux.runtime.service.http.factory import create_service_app
from msgflux.runtime.service.http.session import AgentSessionClient
from msgflux.runtime.service.records import ServiceRecoveryRequiredError
from tests.test_agent_extensions import _CompactingModel
from tests.test_agent_service_approvals import _close_service, _paused_review_fixture


async def approval_compaction_fixture():
    (
        service,
        agent,
        approvals,
        calls,
        _scope_options,
        thread,
        receipt,
        review,
    ) = await _paused_review_fixture()
    agent.register_extension("compaction", CompactionExtension())
    compactor = _CompactingModel()
    for name in (
        "count_context_tokens",
        "acount_context_tokens",
        "compact_context",
        "acompact_context",
    ):
        setattr(agent.model, name, getattr(compactor, name))
    agent.model.context_capacity = 100_000
    store = agent.checkpoint_store
    state = store.load_state(agent.get_module_name(), thread.thread_id, receipt.run_id)
    previous = ChatMessages(
        thread_id=thread.thread_id, namespace=agent.get_module_name()
    )
    previous.begin_turn(turn_id="previous")
    previous.add_user("Prior request")
    previous.add_assistant("Prior decision")
    previous.end_turn()
    state["messages"]["items"] = previous.to_items() + state["messages"]["items"]
    store.commit_state(
        agent.get_module_name(),
        thread.thread_id,
        receipt.run_id,
        state,
        expected_revision=state["_checkpoint"]["revision"],
    )
    return service, agent, approvals, calls, thread, receipt, review, compactor


@pytest.mark.asyncio
async def test_http_compaction_preserves_approval_and_resume_executes_once():
    (
        service,
        agent,
        _approvals,
        calls,
        thread,
        receipt,
        review,
        compactor,
    ) = await approval_compaction_fixture()
    app = create_service_app(service, token="test-token")
    before = agent.checkpoint_store.load_state(
        agent.get_module_name(), thread.thread_id, receipt.run_id
    )
    try:
        async with httpx2.AsyncClient(
            transport=httpx2.ASGITransport(app=app)
        ) as transport:
            client = AgentServiceClient(
                "http://test", token="test-token", client=transport
            )
            session = await AgentSessionClient.open(client, thread_id=thread.thread_id)
            result = await session.compact()
            assert isinstance(result, CompactionResult)
            assert result.status == "completed"
            assert result.run_id == receipt.run_id
            assert (
                service.receipt(thread.thread_id, receipt.request_id).status == "paused"
            )
            after = agent.checkpoint_store.load_state(
                agent.get_module_name(), thread.thread_id, receipt.run_id
            )
            assert after["runtime"]["extensions"] == before["runtime"]["extensions"]
            assert after["status"] == "paused"
            assert (await session.approval_reviews(receipt.run_id))[0] == review
            assert calls == []
            assert (await session.compact()).reason == "already_compacted"
            await session.decide_approval(
                receipt.run_id,
                review.request_id,
                approved=True,
                expected_revision=review.revision,
            )
            await session.resume(receipt.run_id)
            finished = await session.wait(receipt.request_id)
            assert finished.status == "completed"
            assert calls == ["sensitive query"]
            assert len(compactor.compaction_calls) == 1
            with pytest.raises(AgentServiceHTTPError) as error:
                await session.compact()
            assert error.value.status_code == 422
    finally:
        await _close_service(service)


@pytest.mark.asyncio
async def test_compaction_excludes_same_thread_without_blocking_other_threads(
    monkeypatch,
):
    (
        service,
        agent,
        _approvals,
        _calls,
        thread,
        receipt,
        _review,
        _compactor,
    ) = await approval_compaction_fixture()
    entered, release = asyncio.Event(), asyncio.Event()
    original = agent.model.acompact_context

    async def slow(*args, **kwargs):
        entered.set()
        await release.wait()
        return await original(*args, **kwargs)

    monkeypatch.setattr(agent.model, "acompact_context", slow)
    task = asyncio.create_task(service.compact(thread.thread_id))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        with pytest.raises(ServiceBusyError):
            await service.compact(thread.thread_id)
        with pytest.raises(ServiceBusyError):
            await service.resume_checkpoint(thread.thread_id, receipt.run_id)
        with pytest.raises(ServiceBusyError):
            await service.prompt(thread.thread_id, "Do not admit", request_id="new")
        assert service.store.get(thread.thread_id, "new") is None
        from tests.test_manual_compaction import paused_agent

        other_agent, _store, _scope = paused_agent()
        service.register("other", lambda _thread: AgentSession(other_agent))
        other = await asyncio.wait_for(service.open_thread("other"), 1)
        assert other.agent_id == "other"
    finally:
        release.set()
        assert (await task).status == "completed"
        await _close_service(service)


@pytest.mark.asyncio
async def test_service_shutdown_cancels_compaction_before_closing_resources(
    monkeypatch,
):
    (
        service,
        agent,
        _approvals,
        _calls,
        thread,
        receipt,
        _review,
        _compactor,
    ) = await approval_compaction_fixture()
    before = agent.checkpoint_store.load_state(
        agent.get_module_name(), thread.thread_id, receipt.run_id
    )
    entered = asyncio.Event()

    async def slow(*args, **kwargs):
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(agent.model, "acompact_context", slow)
    task = asyncio.create_task(service.compact(thread.thread_id))
    await asyncio.wait_for(entered.wait(), 2)
    await asyncio.wait_for(service.aclose(), 2)
    with pytest.raises(asyncio.CancelledError):
        await task
    assert (
        agent.checkpoint_store.load_state(
            agent.get_module_name(), thread.thread_id, receipt.run_id
        )
        == before
    )
    assert not service._maintenance_tasks
    service.store.close()


@pytest.mark.asyncio
async def test_unauthorized_compaction_and_unknown_body_fields_do_not_call_provider():
    (
        service,
        _agent,
        _approvals,
        _calls,
        thread,
        _receipt,
        _review,
        compactor,
    ) = await approval_compaction_fixture()
    try:
        app = create_service_app(service, token="test-token")
        async with httpx2.AsyncClient(
            transport=httpx2.ASGITransport(app=app), base_url="http://test"
        ) as http:
            path = f"/v1/threads/{thread.thread_id}/compact"
            assert (await http.post(path, json={})).status_code == 401
            response = await http.post(
                path,
                json={"messages": []},
                headers={"Authorization": "Bearer test-token"},
            )
            assert response.status_code == 422
            assert not compactor.compaction_calls
    finally:
        await _close_service(service)


@pytest.mark.asyncio
async def test_sqlite_restart_compaction_requires_quiescence_and_keeps_approval(
    tmp_path,
):
    from unittest.mock import AsyncMock

    from msgflux.data.stores import SQLiteCheckpointStore
    from msgflux.runtime import SQLiteApprovalStore
    from msgflux.runtime.service.local.leases import LocalOwnerLease
    from tests.test_service_approval_recovery import (
        _agent,
        _service,
        _text_response,
        _tool_response,
    )

    effects = []

    def record_effect(value: str) -> str:
        """Record an approved effect."""
        effects.append(value)
        return value

    def build():
        checkpoints = SQLiteCheckpointStore(tmp_path / "checkpoints.sqlite3")
        approvals = SQLiteApprovalStore(tmp_path / "approvals.sqlite3")
        journal = SQLiteServiceStore(tmp_path / "service.sqlite3")
        agent = _agent(checkpoints, approvals, record_effect)
        compactor = _CompactingModel()
        agent.model.acount_context_tokens = compactor.acount_context_tokens
        agent.model.acompact_context = compactor.acompact_context
        service = _service(agent, checkpoints, approvals, journal)
        service._owner_lease = LocalOwnerLease(service)
        return service, agent, checkpoints, approvals, compactor

    first, agent, checkpoints, approvals, _compactor = build()
    agent.generator.aforward = AsyncMock(
        side_effect=[_text_response("Prior decision"), _tool_response()]
    )
    second = None
    try:
        thread = await first.open_thread("main", thread_id="restart-compaction")
        await first.prompt(thread.thread_id, "First request", request_id="previous")
        assert (await first.wait(thread.thread_id, "previous")).status == "completed"
        receipt = await first.prompt(
            thread.thread_id, "Record effect", request_id="current"
        )
        assert (await first.wait(thread.thread_id, "current")).status == "paused"
        (review,) = await first.approval_reviews(thread.thread_id, receipt.run_id)
        second, recovered, saved, saved_approvals, compactor = build()
        with pytest.raises(ServiceRecoveryRequiredError, match="still held"):
            await second.compact(thread.thread_id)
        assert not compactor.compaction_calls
        await first.aclose()
        first._owner_lease.close()
        checkpoints.close()
        approvals.close()
        first.store.close()
        result = await second.compact(thread.thread_id)
        assert result.status == "completed"
        assert result.run_id == receipt.run_id
        assert len(compactor.compaction_calls) == 1
        assert effects == []
        assert (await second.approval_reviews(thread.thread_id, receipt.run_id)) == (
            review,
        )
        recovered.generator.aforward = AsyncMock(return_value=_text_response("done"))
        await second.decide_approval(
            thread.thread_id,
            receipt.run_id,
            review.request_id,
            approved=True,
            expected_revision=review.revision,
        )
        await second.resume_checkpoint(thread.thread_id, receipt.run_id)
        assert (await second.wait(thread.thread_id, "current")).status == "completed"
        assert effects == ["one"]
    finally:
        await first.aclose()
        first._owner_lease.close()
        first.store.close()
        checkpoints.close()
        approvals.close()
        if second is not None:
            await second.aclose()
            second._owner_lease.close()
            second.store.close()
            saved.close()
            saved_approvals.close()
