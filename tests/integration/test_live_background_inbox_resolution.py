"""Opt-in live provider check for resumed AgentTool inbox routing.

Run with ``MSGFLUX_LIVE_INBOX_RESOLUTION=1`` and ``OPENAI_API_KEY`` set.
The test makes two billable model calls. The model wrapper holds the second
response briefly so ``task_message`` can target a genuinely running subagent.
"""

from __future__ import annotations

import asyncio
import os
import threading
import time

import msgflux as mf
import pytest

from msgflux.data.stores import SQLiteCheckpointStore
from msgflux.models import Model
from msgflux.models.chat_transport import HTTPChatTransport
from msgflux.nn import Agent
from msgflux.nn.modules.tool import ToolLibrary
from msgflux.runtime.agent_inbox import AgentInbox, SQLiteAgentInboxStore
from msgflux.runtime.context import execution_context
from msgflux.tasks import SQLiteTaskStore
from msgflux.tools.builtin import AgentTool
from msgflux.tools.builtin.task_tool import TaskMessageTool
from msgflux.tools.config import tool_config


def _wait_for_status(store, task_id: str, status: str) -> None:
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        task = store.get(task_id)
        if task.status == status:
            return
        if task.status == "failed":
            raise AssertionError("Live background agent failed")
        time.sleep(0.05)
    raise TimeoutError(f"Live task did not reach {status}")


@pytest.mark.skipif(
    os.getenv("MSGFLUX_LIVE_INBOX_RESOLUTION") != "1",
    reason="Set MSGFLUX_LIVE_INBOX_RESOLUTION=1 for billable model calls",
)
def test_live_resumed_agenttool_routes_message_without_inbox_map(
    tmp_path, record_property
):
    dotenv_path = os.getenv("MSGFLUX_LIVE_DOTENV")
    mf.load_dotenv(dotenv_path or ".env")
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY is required")

    started = threading.Event()
    release = threading.Event()
    real_model = Model.chat_completion(
        os.getenv("MSGFLUX_LIVE_INBOX_MODEL", "openai/gpt-6-luna"),
        api_mode="responses",
        max_tokens=96,
        retry=False,
        chat_transport=HTTPChatTransport(timeout=60, max_retries=0),
    )

    class HoldingModel:
        model_type = "chat_completion"
        model_id = real_model.model_id
        provider = real_model.provider

        def __init__(self):
            self.calls = 0

        def __call__(self, **kwargs):
            return real_model(**kwargs)

        async def acall(self, **kwargs):
            response = await real_model.acall(**kwargs)
            self.calls += 1
            if self.calls == 2:
                started.set()
                if not await asyncio.to_thread(release.wait, 30):
                    raise TimeoutError("Second model response was not released")
            return response

    checkpoints = SQLiteCheckpointStore(str(tmp_path / "checkpoints.sqlite"))
    tasks = SQLiteTaskStore(str(tmp_path / "tasks.sqlite"))
    inbox_store = SQLiteAgentInboxStore(str(tmp_path / "inbox.sqlite"))
    root_inbox = AgentInbox(owner="root", store=inbox_store)
    model = HoldingModel()
    child = Agent(
        name="live_child",
        model=model,
        checkpoint_store=checkpoints,
        system_prompt="Answer in one short sentence.",
    )
    library = ToolLibrary(
        name="live_library",
        tools=[tool_config(allow_background=True)(AgentTool()), child],
        task_store=tasks,
    )
    library.set_agent_inbox(root_inbox)
    task_id = None
    try:
        with execution_context(
            thread_id="live_thread",
            namespace="root",
            run_id="root_run",
            root_run_id="root_run",
            checkpoint_store=checkpoints,
            agent_inbox=root_inbox,
        ):
            dispatched = library(
                [
                    (
                        "first",
                        "agent",
                        {
                            "name": "live_child",
                            "message": "Say hello briefly.",
                            "run_in_background": True,
                        },
                    )
                ]
            )
            assert dispatched.tool_calls[0].error is None
            task_id = (
                dispatched.tool_calls[0]
                .result.split("task_id='", 1)[1]
                .split("'", 1)[0]
            )
            _wait_for_status(tasks, task_id, "completed")

            resumed = library(
                [
                    (
                        "resume",
                        "task_message",
                        {"task_id": task_id, "message": "Name one way to save memory."},
                    )
                ]
            )
            assert resumed.tool_calls[0].result["status"] == "resumed"
            assert started.wait(90), "Second real model call did not return"

            current_inbox = library.get_handle().get_task_inbox(task_id)
            sent = TaskMessageTool()(
                task_id=task_id,
                message="Additional instruction while running",
                handle=library.get_handle(),
            )
            assert sent["status"] == "delivered"
            assert (
                current_inbox.run_id == tasks.get(task_id).metadata["checkpoint_run_id"]
            )
            assert [item.metadata["message"] for item in current_inbox.peek()] == [
                "Additional instruction while running"
            ]
            assert not hasattr(library.get_background_dispatcher(), "_task_inboxes")
            record_property("real_model_calls", model.calls)
            assert model.calls == 2
    finally:
        release.set()
        try:
            if task_id is not None:
                _wait_for_status(tasks, task_id, "completed")
        finally:
            real_model.close()
            tasks.close()
            checkpoints.close()
            inbox_store.close()
