"""Unit tests for msgflux.tools.builtin.send_parent_message."""

from __future__ import annotations

from dataclasses import replace

import pytest

from msgflux.nn.modules.tool import ToolLibrary
from msgflux.runtime.agent_inbox import AgentInbox, InMemoryAgentInboxStore
from msgflux.runtime.agent_inbox.base import AgentInboxClosedError
from msgflux.runtime.context import ExecutionScope, execution_context
from msgflux.tasks import InMemoryTaskStore, TaskHandle
from msgflux.tools.builtin import SendParentMessageTool
from msgflux.tools.runtime import ToolIntent

_THREAD_ID = "thread"
_PARENT_RUN_ID = "parent_run"
_CHILD_RUN_ID = "child_run"


def _parent_library() -> ToolLibrary:
    return ToolLibrary(name="parent", tools=[])


def _parent_inbox(library: ToolLibrary) -> AgentInbox:
    return library.get_handle().get_agent_inbox(
        thread_id=_THREAD_ID, run_id=_PARENT_RUN_ID
    )


def _notices(library: ToolLibrary) -> list:
    return _parent_inbox(library).peek()


def _child_scope(
    *,
    namespace: str = "explorer",
    run_id: str = _CHILD_RUN_ID,
) -> ExecutionScope:
    return ExecutionScope(
        namespace=namespace,
        thread_id=_THREAD_ID,
        run_id=run_id,
        parent_run_id=_PARENT_RUN_ID,
    )


def _composed_library(parent_library: ToolLibrary) -> ToolLibrary:
    """Compose the tool with an explicit parent handle, as the factory does."""
    parent = parent_library.get_handle()
    return ToolLibrary(name="explorer", tools=[SendParentMessageTool(parent)])


def _send(library: ToolLibrary, message, call_id: str = "call_1"):
    return library.execute_intents(
        [
            ToolIntent(
                id=call_id,
                name="send_parent_message",
                arguments={"message": message},
            )
        ]
    )[0]


def test_declaration_is_a_normal_handle_tool():
    tool = SendParentMessageTool(_parent_library().get_handle())

    assert tool.name == "send_parent_message"
    assert tool.display_name == "Send Parent Message"
    assert tool.annotations == {"message": str, "return": str}
    assert tool.tool_config == {"runtime_inputs": ("handle",)}
    assert "raw logs" in tool.usage_guidance


def test_tool_requires_a_parent_library_handle():
    with pytest.raises(TypeError, match="ToolLibraryHandle"):
        SendParentMessageTool(object())


def test_schema_exposes_only_the_message_argument():
    library = _composed_library(_parent_library())

    function = library.get_tool_json_schemas()[0]["function"]

    assert function["name"] == "send_parent_message"
    assert function["parameters"]["required"] == ["message"]
    assert set(function["parameters"]["properties"]) == {"message"}
    assert function["parameters"]["properties"]["message"]["type"] == "string"
    assert "raw logs" in function["parameters"]["properties"]["message"]["description"]


def test_publishes_to_the_parent_namespace_run_and_thread():
    parent_library = _parent_library()
    library = _composed_library(parent_library)

    with execution_context(scope=_child_scope()):
        outcome = _send(library, "Cause found in the retry helper.")

    assert outcome.status == "completed"
    assert outcome.error is None
    assert "parent" in outcome.result
    notices = _notices(parent_library)
    assert len(notices) == 1
    notice = notices[0]
    assert notice.source == "incoming_message"
    assert notice.ref == _CHILD_RUN_ID
    assert notice.metadata["content"] == (
        f"[explorer:{_CHILD_RUN_ID}] Cause found in the retry helper."
    )
    assert notice.metadata["description"] == "Message from subagent explorer"
    assert notice.metadata["origin"] == "subagent"
    assert notice.metadata["agent_name"] == "explorer"
    assert notice.metadata["direction"] == "child_to_parent"
    assert notice.metadata["ref_kind"] == "run"
    assert notice.metadata["run_id"] == _CHILD_RUN_ID


def test_message_is_stripped_and_keeps_no_length_cap():
    parent_library = _parent_library()
    library = _composed_library(parent_library)
    body = "x" * 8000

    with execution_context(scope=_child_scope()):
        assert _send(library, f"  \n{body}\t ").status == "completed"

    content = _notices(parent_library)[0].metadata["content"]
    assert content == f"[explorer:{_CHILD_RUN_ID}] {body}"


@pytest.mark.parametrize("message", ["", "   ", "\n\t "])
def test_blank_message_is_rejected_without_publishing(message):
    parent_library = _parent_library()
    library = _composed_library(parent_library)

    with execution_context(scope=_child_scope()):
        outcome = _send(library, message)

    assert outcome.status != "completed"
    assert "non-empty" in outcome.error.message
    assert _notices(parent_library) == []


def test_non_string_message_is_rejected_without_publishing():
    parent_library = _parent_library()
    tool = SendParentMessageTool(parent_library.get_handle())

    with execution_context(scope=_child_scope()):
        with pytest.raises(ValueError, match="non-empty"):
            tool(3, handle=parent_library.get_handle())

    assert _notices(parent_library) == []


@pytest.mark.parametrize(
    "overrides",
    [
        {"thread_id": None},
        {"parent_run_id": None},
        {"run_id": None},
        {"thread_id": ""},
        {"parent_run_id": "  "},
        {"run_id": ""},
    ],
)
def test_scope_identity_is_required(overrides):
    parent_library = _parent_library()
    library = _composed_library(parent_library)
    scope = replace(_child_scope(), **overrides)

    with execution_context(scope=scope):
        outcome = _send(library, "Cause found.")

    assert outcome.status != "completed"
    assert "subagent scope" in outcome.error.message
    assert _notices(parent_library) == []


def test_closed_parent_inbox_raises_and_keeps_no_notice():
    parent_library = _parent_library()
    library = _composed_library(parent_library)
    _parent_inbox(parent_library).close_user_messages(only_if_empty=False)

    with execution_context(scope=_child_scope()):
        outcome = _send(library, "Cause found.")

    assert outcome.status != "completed"
    assert outcome.error.details["exception_type"] == AgentInboxClosedError.__name__
    assert _notices(parent_library) == []


def _background_context(store: InMemoryTaskStore, task_id: str, *, scope):
    return execution_context(
        task_store=store,
        task_handle=TaskHandle(task_id, store, tool_name="agent"),
        scope=scope,
    )


def _record(
    store: InMemoryTaskStore,
    task_id: str,
    *,
    namespace: str = "explorer",
    thread_id: str = _THREAD_ID,
    checkpoint_run_id: str | None = None,
) -> None:
    store.create(
        "agent",
        task_id=task_id,
        metadata={
            "checkpoint_run_id": checkpoint_run_id or task_id,
            "checkpoint_namespace": namespace,
            "thread_id": thread_id,
        },
    )


def test_background_task_reference_uses_the_task_id():
    parent_library = _parent_library()
    library = _composed_library(parent_library)
    store = InMemoryTaskStore()
    _record(store, "task1234")

    with _background_context(store, "task1234", scope=_child_scope(run_id="task1234")):
        assert _send(library, "Done.").status == "completed"

    notice = _notices(parent_library)[0]
    assert notice.ref == "task1234"
    assert notice.metadata["ref_kind"] == "task"
    assert notice.metadata["task_id"] == "task1234"
    assert notice.metadata["content"] == "[explorer:task1234] Done."


def test_resumed_task_keeps_the_stable_task_reference():
    parent_library = _parent_library()
    library = _composed_library(parent_library)
    store = InMemoryTaskStore()
    _record(store, "task1234")

    with _background_context(store, "task1234", scope=_child_scope(run_id="task1234")):
        assert _send(library, "First.", "call_1").status == "completed"

    store.requeue("task1234", run_id="resumed_run")
    assert store.get("task1234").metadata["checkpoint_run_id"] == "resumed_run"

    with _background_context(
        store, "task1234", scope=_child_scope(run_id="resumed_run")
    ):
        assert _send(library, "Second.", "call_2").status == "completed"

    refs = [notice.ref for notice in _notices(parent_library)]
    assert refs == ["task1234", "task1234"]


def test_inline_nested_child_falls_back_to_its_own_run_id():
    parent_library = _parent_library()
    library = _composed_library(parent_library)
    store = InMemoryTaskStore()
    # An inline nested child inherits the ancestor task handle but not its run.
    _record(store, "ancestor_task")

    with _background_context(
        store, "ancestor_task", scope=_child_scope(run_id="leaf_run")
    ):
        assert _send(library, "Leaf done.").status == "completed"

    notice = _notices(parent_library)[0]
    assert notice.ref == "leaf_run"
    assert notice.metadata["ref_kind"] == "run"
    assert notice.metadata["run_id"] == "leaf_run"
    assert notice.metadata["content"] == "[explorer:leaf_run] Leaf done."


@pytest.mark.parametrize(
    "record_kwargs",
    [
        {"namespace": "builder"},
        {"thread_id": "other_thread"},
        {"checkpoint_run_id": "other_run"},
    ],
)
def test_task_reference_requires_matching_checkpoint_identity(record_kwargs):
    parent_library = _parent_library()
    library = _composed_library(parent_library)
    store = InMemoryTaskStore()
    _record(store, "task1234", **record_kwargs)

    with _background_context(store, "task1234", scope=_child_scope(run_id="task1234")):
        assert _send(library, "Done.").status == "completed"

    notice = _notices(parent_library)[0]
    assert notice.ref == "task1234"
    assert notice.metadata["ref_kind"] == "run"


def test_missing_task_record_falls_back_to_the_scope_run_id():
    parent_library = _parent_library()
    library = _composed_library(parent_library)
    store = InMemoryTaskStore()

    with _background_context(store, "task1234", scope=_child_scope(run_id="task1234")):
        assert _send(library, "Done.").status == "completed"

    assert _notices(parent_library)[0].metadata["ref_kind"] == "run"


def test_foreground_call_uses_the_scope_run_id():
    parent_library = _parent_library()
    library = _composed_library(parent_library)
    store = InMemoryTaskStore()
    _record(store, "task1234")

    with execution_context(task_store=store, scope=_child_scope()):
        assert _send(library, "Done.").status == "completed"

    notice = _notices(parent_library)[0]
    assert notice.ref == _CHILD_RUN_ID
    assert notice.metadata["ref_kind"] == "run"


def test_publish_uses_the_effective_parent_backend_not_the_child_run():
    parent_library = _parent_library()
    library = _composed_library(parent_library)
    ambient = AgentInbox(
        store=InMemoryAgentInboxStore(), namespace="explorer", thread_id="other"
    )

    with execution_context(agent_inbox=ambient, scope=_child_scope()):
        assert _send(library, "Done.").status == "completed"

    target = ambient.fork(
        namespace=parent_library.name,
        thread_id=_THREAD_ID,
        run_id=_PARENT_RUN_ID,
    )
    assert [notice.metadata["content"] for notice in target.peek()] == [
        f"[explorer:{_CHILD_RUN_ID}] Done."
    ]
    assert ambient.peek() == []
    assert _notices(parent_library) == []


def test_direction_metadata_is_not_custom_identity():
    parent_library = _parent_library()
    library = _composed_library(parent_library)

    with execution_context(scope=_child_scope()):
        assert _send(library, "First.", "call_1").status == "completed"
        assert _send(library, "Second.", "call_2").status == "completed"

    notices = _notices(parent_library)
    assert len(notices) == 2
    assert len({notice.notification_id for notice in notices}) == 2
    assert all(notice.dedupe_key is None for notice in notices)
    assert [notice.metadata["content"] for notice in notices] == [
        f"[explorer:{_CHILD_RUN_ID}] First.",
        f"[explorer:{_CHILD_RUN_ID}] Second.",
    ]
