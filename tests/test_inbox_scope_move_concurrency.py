"""An inbox scope change must not redeliver another run's leased input."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event
from unittest.mock import Mock

import pytest

from msgflux.models.response import ModelResponse
from msgflux.nn.hooks import Hook
from msgflux.nn.hooks.events import NotificationContext
from msgflux.nn.modules.agent import Agent
from msgflux.runtime.agent_inbox import (
    AgentInbox,
    InMemoryAgentInboxStore,
    SQLiteAgentInboxStore,
)
from msgflux.runtime.context import ExecutionScope


@pytest.mark.parametrize("provider", ["memory", "sqlite"])
def test_binding_another_run_does_not_move_a_claimed_message(provider, tmp_path):
    if provider == "memory":
        store_a = store_b = InMemoryAgentInboxStore()
    else:
        path = str(tmp_path / "shared-agent-inbox.sqlite")
        store_a = SQLiteAgentInboxStore(path)
        store_b = SQLiteAgentInboxStore(path)

    inbox_a = AgentInbox(
        store=store_a,
        namespace="agent",
        thread_id="thread",
        run_id="run-a",
    )
    inbox_b = AgentInbox(
        store=store_b,
        namespace="agent",
        thread_id="thread",
        run_id="run-a",
    )
    inbox_a.user_message("accepted for run A")
    entered_transform = Event()
    release_transform = Event()
    requests_a = []
    requests_b = []

    def block_run_a(context: NotificationContext):
        if context.scope.run_id == "run-a":
            entered_transform.set()
            if not release_transform.wait(timeout=5):
                raise TimeoutError("run A notification transform was not released")
        return context

    def answer(requests):
        def complete(**kwargs):
            requests.append(str(kwargs["messages"]))
            response = ModelResponse()
            response.set_response_type("text_generation")
            response.add("done")
            return response

        return complete

    agent_a = Agent(
        name="agent",
        model=Mock(model_type="chat_completion"),
        agent_inbox=inbox_a,
        hooks=[Hook(event="transform_notifications", handler=block_run_a)],
    )
    agent_b = Agent(
        name="agent",
        model=Mock(model_type="chat_completion"),
        agent_inbox=inbox_b,
    )
    agent_a.generator.forward = Mock(side_effect=answer(requests_a))
    agent_b.generator.forward = Mock(side_effect=answer(requests_b))

    def scope(run_id):
        return ExecutionScope(namespace="agent", thread_id="thread", run_id=run_id)

    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            run_a = pool.submit(agent_a, "start A", scope=scope("run-a"))
            assert entered_transform.wait(timeout=5)
            assert agent_b("start B", scope=scope("run-b")) == "done"
            release_transform.set()
            assert run_a.result(timeout=5) == "done"

        assert len(requests_a) == len(requests_b) == 1
        assert "accepted for run A" in requests_a[0]
        assert "accepted for run A" not in requests_b[0]
        assert inbox_a.peek() == []
    finally:
        release_transform.set()
        if provider == "sqlite":
            store_a.close()
            store_b.close()
