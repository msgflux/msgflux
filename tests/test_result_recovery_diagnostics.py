"""CLI recovery diagnostics preserve a failure settled by another live service."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from msgflux.data.stores import InMemoryCheckpointStore
from msgflux.nn import Agent
from msgflux.runtime.service import (
    AgentService,
    AgentSession,
    ServiceRecoveryRequiredError,
    SQLiteServiceStore,
)
from msgflux.runtime.service.local.results import show_result


class ObserverClient:
    def __init__(self, service, on_recovery=None):
        self.service = service
        self.on_recovery = on_recovery

    async def run_result(self, thread_id, run_id):
        return await self.service.run_result(thread_id, run_id)

    async def inspect_run(self, thread_id, run_id):
        return await self.service.inspect_run(thread_id, run_id)

    async def receipt(self, thread_id, request_id, *, require_waitable):
        assert require_waitable
        try:
            return self.service._wait_receipt(thread_id, request_id)
        except ServiceRecoveryRequiredError:
            if self.on_recovery is not None:
                await self.on_recovery(thread_id, request_id)
            raise


def _agent():
    return Agent(
        name="result-agent",
        model=Mock(model_type="chat_completion"),
        checkpoint_store=InMemoryCheckpointStore(),
    )


def _service(journal, agent):
    service = AgentService(store=journal)
    service.register("main", lambda _thread: AgentSession(agent))
    return service


def _args(thread_id, run_id, as_json):
    return SimpleNamespace(
        resource="runs", thread_id=thread_id, run_id=run_id, wait=True, json=as_json
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("as_json", [False, True])
async def test_result_preserves_failure_settled_during_recovery(
    tmp_path, capsys, as_json
):
    journals = [SQLiteServiceStore(tmp_path / "service.sqlite3") for _ in range(2)]
    agent = _agent()
    owner, observer = [_service(journal, agent) for journal in journals]
    entered, fail = asyncio.Event(), asyncio.Event()

    async def answer(**_kwargs):
        entered.set()
        await fail.wait()
        raise RuntimeError("provider unavailable")

    agent.generator.aforward = AsyncMock(side_effect=answer)

    async def settle_failure(thread_id, request_id):
        fail.set()
        final = await asyncio.wait_for(owner.wait(thread_id, request_id), 2)
        assert final.status == "failed"
        assert final.error == "provider unavailable"

    try:
        thread = await owner.open_thread("main")
        receipt = await owner.prompt(thread.thread_id, "hello", request_id="first")
        await asyncio.wait_for(entered.wait(), 2)
        client = ObserverClient(observer, settle_failure)
        with pytest.raises(SystemExit) as caught:
            await show_result(client, _args(thread.thread_id, receipt.run_id, as_json))
        assert caught.value.code == 1
        captured = capsys.readouterr()
        if as_json:
            result = json.loads(captured.out)
            assert result["status"] == "failed"
            assert result["error"] == "provider unavailable"
        else:
            assert "failed" in captured.err
            assert "provider unavailable" in captured.err
            assert "does not own" not in captured.err
    finally:
        fail.set()
        await owner.aclose()
        await observer.aclose()
        for journal in journals:
            journal.close()


@pytest.mark.asyncio
async def test_result_without_recorded_error_keeps_recovery_guidance(capsys):
    journal = SQLiteServiceStore()
    agent = _agent()
    service = _service(journal, agent)
    try:
        thread = await service.open_thread("main")
        admission = journal.admit(thread.thread_id, "orphan", "hello", agent.name)
        with pytest.raises(SystemExit) as caught:
            await show_result(
                ObserverClient(service),
                _args(thread.thread_id, admission.receipt.run_id, True),
            )
        assert caught.value.code == 1
        result = json.loads(capsys.readouterr().out)
        assert result["status"] == "accepted"
        assert result["error"] == "This service does not own that attempt"
    finally:
        await service.aclose()
        journal.close()
