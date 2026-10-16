"""Latest admission selection survives persistence and stays independent of history."""

import asyncio

import pytest

pytest.importorskip("litestar")
pytest.importorskip("uvicorn")

from msgflux.runtime.service import AgentSession, SQLiteServiceStore
from msgflux.runtime.service.http import AgentServiceClient, AgentServiceHTTPError
from msgflux.runtime.service.records import ServiceThread
from tests.test_agent_service_http_integration import TOKEN, _agent, _response, _server


def test_latest_admission_preserves_insert_order_after_updates_and_reopen(tmp_path):
    path = tmp_path / "service.sqlite3"
    store = SQLiteServiceStore(path)
    try:
        store.bind_thread(ServiceThread("thread", "main"))
        assert store.latest_admission("thread") is None
        first = store.admit("thread", "first", "one", "main")
        assert store.claim(first, "owner")
        first = store.get("thread", "first")
        store.finish(first.receipt, "owner", "completed")
        latest = store.admit("thread", "second", "two", "main")
        # Replaying an older input must not move it ahead of the new admission.
        assert store.admit("thread", "first", "one", "main").receipt.status == (
            "completed"
        )
        assert store.latest_admission("thread") == latest
        assert store.claim(latest, "owner")
        running = store.get("thread", "second")
        store.finish(running.receipt, "owner", "failed", "transport broke")
        assert store.latest_admission("thread").receipt.run_id == latest.receipt.run_id
        # Recovery changes revision and status, never admission order.
        store.prepare_resume(store.get("thread", "second"))
        assert store.latest_admission("thread").receipt.run_id == latest.receipt.run_id
    finally:
        store.close()
    reopened = SQLiteServiceStore(path)
    try:
        persisted = reopened.latest_admission("thread")
        assert persisted.receipt.run_id == latest.receipt.run_id
        assert persisted.receipt.status == "accepted"
        assert reopened._connection.execute("PRAGMA integrity_check").fetchone()[0] == (
            "ok"
        )
    finally:
        reopened.close()


@pytest.mark.asyncio
async def test_thread_result_tracks_latest_run_without_waiting_or_hiding_history():
    entered, release = asyncio.Event(), asyncio.Event()
    count = 0

    async def answer(**_kwargs):
        nonlocal count
        count += 1
        if count == 2:
            entered.set()
            await release.wait()
        return _response(f"answer {count}")

    agent = _agent(answer)
    async with _server(lambda _thread: AgentSession(agent)) as (service, url):
        client = AgentServiceClient(url, token=TOKEN)
        try:
            thread = await client.open_thread("main")
            for thread_id in (thread.thread_id, "unknown"):
                with pytest.raises(AgentServiceHTTPError) as caught:
                    await client.thread_result(thread_id)
                assert caught.value.status_code == 404
            first = await client.prompt(thread.thread_id, "one", request_id="first")
            await service.wait(thread.thread_id, "first")
            assert (await client.thread_result(thread.thread_id)).output == "answer 1"
            second = await client.prompt(thread.thread_id, "two", request_id="second")
            await asyncio.wait_for(entered.wait(), 5)
            current = await asyncio.wait_for(client.thread_result(thread.thread_id), 2)
            assert current.run_id == second.run_id
            assert current.status == "running" and current.output is None
            assert (await client.run_result(thread.thread_id, first.run_id)).output == (
                "answer 1"
            )
            await client.steer(thread.thread_id, second.run_id, "keep working")
            duplicate = await client.prompt(thread.thread_id, "one", request_id="first")
            assert duplicate.run_id == first.run_id
            assert (
                await client.thread_result(thread.thread_id)
            ).run_id == second.run_id
            release.set()
            await service.wait(thread.thread_id, "second")
            final = await client.thread_result(thread.thread_id)
            assert final.run_id == second.run_id and final.output == "answer 2"
            assert count == 2
        finally:
            release.set()
            await client.aclose()
