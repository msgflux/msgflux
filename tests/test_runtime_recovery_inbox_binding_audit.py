"""Inbox scope binding preserves destination dedupe updates."""

from unittest.mock import Mock

import pytest

from msgflux.data.stores import InMemoryCheckpointStore
from msgflux.models.response import ModelResponse
from msgflux.nn import Agent
from msgflux.runtime import ExecutionScope
from msgflux.runtime.agent_inbox import (
    AgentInbox,
    InMemoryAgentInboxStore,
    SQLiteAgentInboxStore,
)
from msgflux.runtime.agent_inbox.base import AgentInboxStore


class _BaseMoveStore(AgentInboxStore):
    """Exercise the default move implementation over an in-memory store."""

    def __init__(self):
        self._store = InMemoryAgentInboxStore()

    def load_notifications(self, namespace, thread_id, run_id):
        return self._store.load_notifications(namespace, thread_id, run_id)

    def save_notifications(self, namespace, thread_id, run_id, notifications):
        self._store.save_notifications(namespace, thread_id, run_id, notifications)

    def clear(self, namespace=None, thread_id=None, run_id=None, *, older_than=None):
        return self._store.clear(namespace, thread_id, run_id, older_than=older_than)

    def open_user_messages(self, namespace, thread_id, run_id):
        self._store.open_user_messages(namespace, thread_id, run_id)

    def close_user_messages(
        self, namespace, thread_id, run_id, *, only_if_empty=True, consumed_ids=()
    ):
        return self._store.close_user_messages(
            namespace,
            thread_id,
            run_id,
            only_if_empty=only_if_empty,
            consumed_ids=consumed_ids,
        )


@pytest.fixture(params=["memory", "sqlite"])
def inbox_pair(request, tmp_path):
    if request.param == "memory":
        store = InMemoryAgentInboxStore()
        yield store, store
        return

    path = str(tmp_path / "inbox.sqlite")
    first = SQLiteAgentInboxStore(path)
    second = SQLiteAgentInboxStore(path)
    try:
        yield first, second
    finally:
        first.close()
        second.close()


def test_binding_does_not_resurrect_older_deduped_configuration(inbox_pair):
    source_store, target_store = inbox_pair
    source = AgentInbox(
        store=source_store,
        namespace="assistant",
        thread_id="thread",
        run_id="old-run",
    )
    target = AgentInbox(
        store=target_store,
        namespace="assistant",
        thread_id="thread",
        run_id="new-run",
    )
    source.set_reasoning_effort("low")
    target.set_reasoning_effort("high")

    model = Mock(model_type="chat_completion")
    checkpoint = InMemoryCheckpointStore()
    agent = Agent(
        name="assistant", model=model, agent_inbox=source, checkpoint_store=checkpoint
    )
    response = ModelResponse()
    response.set_response_type("text_generation")
    response.add("done")
    agent.generator.forward = Mock(return_value=response)

    scope = ExecutionScope(namespace="assistant", thread_id="thread", run_id="new-run")
    assert agent("work", scope=scope) == "done"

    assert agent.get_reasoning_effort(scope=scope) == "high"


def test_binding_keeps_target_lease_and_moves_non_deduped_content(inbox_pair):
    source_store, target_store = inbox_pair
    source = AgentInbox(
        store=source_store,
        namespace="assistant",
        thread_id="thread",
        run_id="old-run",
    )
    target = AgentInbox(
        store=target_store,
        namespace="assistant",
        thread_id="thread",
        run_id="new-run",
    )
    source.set_reasoning_effort("low")
    carried = source.user_message("carry this message")
    destination_setting = target.set_reasoning_effort("high")
    target_owner = target.fork()
    assert [item.notification_id for item in target_owner.claim()] == [
        destination_setting.notification_id
    ]

    source.bind(run_id="new-run")

    # The destination's original lease still owns its deduped setting.
    target_owner.ack([destination_setting.notification_id])
    assert target.peek() == [carried]


def test_default_store_move_keeps_destination_dedupe_value():
    store = _BaseMoveStore()
    source = AgentInbox(
        store=store,
        namespace="assistant",
        thread_id="thread",
        run_id="old-run",
    )
    target = AgentInbox(
        store=store,
        namespace="assistant",
        thread_id="thread",
        run_id="new-run",
    )
    source.set_reasoning_effort("low")
    carried = source.user_message("carry this message")
    destination_setting = target.set_reasoning_effort("high")

    source.bind(run_id="new-run")

    assert target.peek() == [destination_setting, carried]
