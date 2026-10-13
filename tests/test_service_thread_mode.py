import sqlite3

import pytest

from msgflux.runtime.service import (
    AgentService,
    SQLiteServiceStore,
    ServiceConflictError,
)


@pytest.mark.asyncio
async def test_thread_mode_persists_and_reopening_cannot_change_it(tmp_path):
    path = tmp_path / "service.sqlite3"
    store = SQLiteServiceStore(path)
    service = AgentService(store=store, default_agent="coding")
    service.register("coding", lambda _thread: None)
    try:
        thread = await service.open_thread(thread_id="print", interactive=False)
        assert not thread.interactive
        with pytest.raises(ServiceConflictError):
            await service.open_thread(thread_id="print", interactive=True)
    finally:
        await service.aclose()
        store.close()
    reopened_store = SQLiteServiceStore(path)
    restarted = AgentService(store=reopened_store, default_agent="coding")
    restarted.register("coding", lambda _thread: None)
    try:
        assert not (await restarted.open_thread(thread_id="print")).interactive
        assert (await restarted.open_thread(thread_id="normal")).interactive
    finally:
        await restarted.aclose()
        reopened_store.close()


def test_existing_sqlite_threads_migrate_to_interactive_default(tmp_path):
    path = tmp_path / "service.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE service_threads (thread_id TEXT PRIMARY KEY, agent_id TEXT NOT NULL, cwd TEXT)"
        )
        connection.execute(
            "INSERT INTO service_threads VALUES ('existing', 'coding', NULL)"
        )
    store = SQLiteServiceStore(path)
    try:
        assert store.thread("existing").interactive is True
    finally:
        store.close()
