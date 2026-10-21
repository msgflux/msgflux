"""Inbox dedupe replacements preserve notification lease ownership."""

import pytest

from msgflux.runtime.agent_inbox import (
    AgentInbox,
    InMemoryAgentInboxStore,
    SQLiteAgentInboxStore,
)


@pytest.fixture(params=["memory", "sqlite"])
def store(request, tmp_path):
    if request.param == "memory":
        yield InMemoryAgentInboxStore()
        return
    inbox_store = SQLiteAgentInboxStore(str(tmp_path / "dedupe-inbox.sqlite"))
    try:
        yield inbox_store
    finally:
        inbox_store.close()


def _inbox(store):
    return AgentInbox(
        store=store,
        namespace="steering-audit",
        thread_id="thread",
        run_id="run",
    )


def test_replaced_reasoning_update_releases_old_lease_without_acknowledging_new(
    store,
):
    writer = _inbox(store)
    old_owner = writer.fork()
    new_owner = writer.fork()

    old = writer.set_reasoning_effort("low")
    assert [item.notification_id for item in old_owner.claim()] == [old.notification_id]
    replacement = writer.set_reasoning_effort("high")
    assert replacement.notification_id != old.notification_id

    old_owner.release()
    old_owner.ack([old.notification_id])

    pending = new_owner.peek()
    assert [item.notification_id for item in pending] == [replacement.notification_id]
    assert pending[0].metadata["reasoning_effort"] == "high"
    claimed = new_owner.claim()
    assert [item.notification_id for item in claimed] == [replacement.notification_id]
    assert claimed[0].metadata["reasoning_effort"] == "high"


def test_same_id_dedupe_update_stays_under_original_lease(store):
    writer = _inbox(store)
    old_owner = writer.fork()
    new_owner = writer.fork()

    original = writer.publish(
        {
            "source": "task",
            "notification_id": "stable-id",
            "status": "started",
            "dedupe_key": "task:stable",
        }
    )
    assert [item.notification_id for item in old_owner.claim()] == [
        original.notification_id
    ]
    writer.publish(
        {
            "source": "task",
            "notification_id": original.notification_id,
            "status": "completed",
            "dedupe_key": "task:stable",
        }
    )

    assert new_owner.claim() == []
    old_owner.release()
    claimed = new_owner.claim()
    assert [item.notification_id for item in claimed] == [original.notification_id]
    assert claimed[0].status == "completed"
