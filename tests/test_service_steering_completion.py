"""Steering acceptance follows the Agent run's durable inbox gate."""

import asyncio

import pytest

pytest.importorskip("litestar")
pytest.importorskip("uvicorn")

from msgflux.data.stores import SQLiteCheckpointStore
from msgflux.nn.hooks import Hook
from msgflux.runtime.agent_inbox import AgentInbox, SQLiteAgentInboxStore
from msgflux.runtime.service import AgentSession, ServiceBusyError
from msgflux.runtime.service.http import AgentServiceClient
from tests.test_agent_service_http_integration import TOKEN, _agent, _response, _server


@pytest.mark.asyncio
async def test_steer_during_final_model_request_continues_same_sqlite_run(tmp_path):
    entered, release = asyncio.Event(), asyncio.Event()
    requests = []
    checkpoint_store = SQLiteCheckpointStore(str(tmp_path / "checkpoints.sqlite3"))
    inbox_store = SQLiteAgentInboxStore(str(tmp_path / "inbox.sqlite3"))

    async def answer(**kwargs):
        requests.append(kwargs["messages"].to_chatml())
        if len(requests) == 1:
            entered.set()
            await release.wait()
            return _response("first answer")
        return _response("newest answer")

    agent = _agent(answer, checkpoints=checkpoint_store)
    inbox = AgentInbox(store=inbox_store)
    async with _server(
        lambda _thread: AgentSession(
            agent, checkpoint_store=checkpoint_store, agent_inbox=inbox
        )
    ) as (service, url):
        client = AgentServiceClient(url, token=TOKEN)
        try:
            thread = await client.open_thread("main")
            receipt = await client.prompt(
                thread.thread_id, "start", request_id="steer-during-final-model"
            )
            await asyncio.wait_for(entered.wait(), 5)
            notice = await client.steer(
                thread.thread_id, receipt.run_id, "Include the new requirement"
            )
            assert notice["notification_id"]
            release.set()

            finished = await asyncio.wait_for(
                service.wait(thread.thread_id, receipt.request_id), 5
            )
            assert finished.status == "completed"
            assert finished.run_id == receipt.run_id
            assert len(requests) == 2
            assert str(requests[1]).count("Include the new requirement") == 1
            result = await client.run_result(thread.thread_id, receipt.run_id)
            assert result.output == "newest answer"
            checkpoint = checkpoint_store.load_state(
                "main", thread.thread_id, receipt.run_id
            )
            assert checkpoint["status"] == "completed"
            assert str(checkpoint).count("Include the new requirement") == 1
            assert (
                inbox_store.load_notifications("main", thread.thread_id, receipt.run_id)
                == []
            )
        finally:
            release.set()
            await client.aclose()
    checkpoint_store.close()
    inbox_store.close()


@pytest.mark.asyncio
async def test_steer_after_final_checkpoint_is_busy_and_does_not_publish(tmp_path):
    hook_entered, release_hook = asyncio.Event(), asyncio.Event()
    checkpoint_store = SQLiteCheckpointStore(str(tmp_path / "checkpoints.sqlite3"))
    inbox_store = SQLiteAgentInboxStore(str(tmp_path / "inbox.sqlite3"))

    async def after_run_end(_context):
        hook_entered.set()
        await release_hook.wait()

    agent = _agent(
        lambda **_kwargs: _response("finished"), checkpoints=checkpoint_store
    )
    agent.register_lifecycle_hook(
        "after_run_end", Hook(event="after_run_end", handler=after_run_end)
    )
    inbox = AgentInbox(store=inbox_store)
    async with _server(
        lambda _thread: AgentSession(
            agent, checkpoint_store=checkpoint_store, agent_inbox=inbox
        )
    ) as (service, url):
        client = AgentServiceClient(url, token=TOKEN)
        try:
            thread = await client.open_thread("main")
            receipt = await client.prompt(
                thread.thread_id, "finish", request_id="steer-after-seal"
            )
            await asyncio.wait_for(hook_entered.wait(), 5)
            with pytest.raises(ServiceBusyError) as caught:
                await client.steer(
                    thread.thread_id, receipt.run_id, "must not be accepted"
                )
            assert caught.value.__cause__.status_code == 409
            assert caught.value.__cause__.code == "service_busy"
            assert "no longer accepts steering" in str(caught.value)
            assert (
                inbox_store.load_notifications("main", thread.thread_id, receipt.run_id)
                == []
            )
            assert agent.generator.aforward.await_count == 1
            release_hook.set()
            assert (
                await asyncio.wait_for(
                    service.wait(thread.thread_id, receipt.request_id), 5
                )
            ).status == "completed"
            assert agent.generator.aforward.await_count == 1
        finally:
            release_hook.set()
            await client.aclose()
    checkpoint_store.close()
    inbox_store.close()
