import asyncio

import pytest

from msgflux.runtime import ExecutionScope
from msgflux.vulcano import (
    ActivateSessionTab,
    CloseSessionTab,
    DomainEvent,
    EventType,
    SessionStore,
    SessionWorkspace,
    SubmitInput,
    ToggleSessionPin,
    VulcanoRuntime,
)


def test_session_store_round_trips_forks_and_exports(tmp_path):
    store = SessionStore(tmp_path / "sessions")
    store.ensure("thd_source")
    store.append(
        "thd_source",
        DomainEvent(
            type=EventType.MESSAGE_USER,
            sequence=1,
            payload={"content": "inspect this"},
            correlation_id="one",
        ),
    )
    store.append(
        "thd_source",
        DomainEvent(
            type=EventType.ASSISTANT_COMPLETED,
            sequence=2,
            payload={"content": "done", "status": "completed"},
            correlation_id="one",
        ),
    )

    loaded = store.load("thd_source")
    assert [event.type for event in loaded] == [
        EventType.MESSAGE_USER,
        EventType.ASSISTANT_COMPLETED,
    ]

    fork = store.fork("thd_source", target_thread_id="thd_fork")
    assert fork.parent_thread_id == "thd_source"
    assert fork.forked_from_sequence == 2
    assert fork.event_count == 2

    export = store.export_markdown("thd_fork", tmp_path / "exports" / "run.md")
    markdown = export.read_text(encoding="utf-8")
    assert "## User" in markdown
    assert "inspect this" in markdown
    assert "## Assistant" in markdown
    assert "done" in markdown


def test_session_workspace_persists_pins_without_duplicate_tabs(tmp_path):
    path = tmp_path / "workspace.toml"
    workspace = SessionWorkspace(path)
    workspace.start("thd_one", ("thd_one", "thd_two"))
    workspace.toggle_pin("thd_one")
    workspace.activate("thd_two")
    workspace.activate("thd_two")

    assert [tab.thread_id for tab in workspace.tabs] == ["thd_one", "thd_two"]
    assert [tab.status for tab in workspace.tabs] == ["paused", "active"]

    restored = SessionWorkspace(path)
    restored.start(
        "thd_current",
        ("thd_one", "thd_two", "thd_current"),
    )

    assert [tab.thread_id for tab in restored.tabs] == ["thd_one", "thd_current"]
    assert [tab.status for tab in restored.tabs] == ["idle", "active"]
    assert restored.tabs[0].pinned
    contents = path.read_text(encoding="utf-8")
    assert contents.startswith("version = 1\n")
    assert contents.count('status = "active"') == 1


def test_session_replay_closes_interrupted_streams_as_aborted(tmp_path):
    store = SessionStore(tmp_path / "sessions")
    store.ensure("thd_interrupted")
    store.append(
        "thd_interrupted",
        DomainEvent(
            type=EventType.ASSISTANT_STARTED,
            sequence=1,
            correlation_id="run",
        ),
    )
    store.append(
        "thd_interrupted",
        DomainEvent(
            type=EventType.ASSISTANT_DELTA,
            sequence=2,
            payload={"delta": "partial"},
            correlation_id="run",
        ),
    )

    replay = store.replay("thd_interrupted")

    assert replay[-1].type == EventType.ASSISTANT_COMPLETED
    assert replay[-1].payload == {"content": "partial", "status": "aborted"}


def test_session_replay_closes_interrupted_execution_as_aborted(tmp_path):
    store = SessionStore(tmp_path / "sessions")
    store.ensure("thd_execution")
    scope = {
        "thread_id": "thd_execution",
        "namespace": "vulcano",
        "run_id": "run_interrupted",
        "parent_run_id": None,
        "root_run_id": "run_interrupted",
    }
    store.append(
        "thd_execution",
        DomainEvent(
            type=EventType.EXECUTION_STARTED,
            sequence=1,
            payload={"run_id": "run_interrupted", "scope": scope},
            correlation_id="request",
        ),
    )

    replay = store.replay("thd_execution")

    assert [event.type for event in replay] == [
        EventType.EXECUTION_STARTED,
        EventType.EXECUTION_COMPLETED,
    ]
    assert replay[-1].payload == {
        "run_id": "run_interrupted",
        "scope": scope,
        "status": "aborted",
        "final_message_id": None,
    }


def test_session_replay_does_not_restore_client_view_mode(tmp_path):
    store = SessionStore(tmp_path / "sessions")
    store.ensure("thd_view")
    store.append(
        "thd_view",
        DomainEvent(
            type=EventType.CLIENT_ACTION,
            sequence=1,
            payload={"action": "transcript.view", "mode": "compact"},
        ),
    )

    assert len(store.load("thd_view")) == 1
    assert store.replay("thd_view") == ()


def test_session_replay_keeps_permission_decision_without_reopening_prompt(tmp_path):
    store = SessionStore(tmp_path / "sessions")
    store.ensure("thd_permission")
    payload = {
        "request_id": "permission-one",
        "owner": "extension",
        "operation": "shell",
        "description": "Run tests?",
        "resource": "pytest tests/vulcano",
        "scope": {"thread_id": "thd_permission", "run_id": "run_one"},
    }
    store.append(
        "thd_permission",
        DomainEvent(
            type=EventType.PERMISSION_REQUESTED,
            sequence=1,
            payload={**payload, "requires_confirmation": True},
        ),
    )
    store.append(
        "thd_permission",
        DomainEvent(
            type=EventType.PERMISSION_RESOLVED,
            sequence=2,
            payload={
                **payload,
                "decision": "allow_once",
                "source": "user",
                "allowed": True,
            },
        ),
    )

    replay = store.replay("thd_permission")
    export = store.export_markdown(
        "thd_permission",
        tmp_path / "permission.md",
    ).read_text(encoding="utf-8")

    assert [event.type for event in replay] == [EventType.PERMISSION_RESOLVED]
    assert "Permission: `shell`" in export
    assert "pytest tests/vulcano" in export


@pytest.mark.asyncio
async def test_runtime_replays_persisted_session_before_start_event(tmp_path):
    store = SessionStore(tmp_path / "sessions")
    scope = ExecutionScope(thread_id="thd_resume", namespace="vulcano")
    first = VulcanoRuntime(
        scope=scope,
        session_store=store,
        stream_delay=0,
        extensions_enabled=False,
    )
    await first.dispatch(SubmitInput("persist me", correlation_id="persisted"))
    await first.stop()

    resumed = VulcanoRuntime(
        scope=scope,
        session_store=store,
        stream_delay=0,
        extensions_enabled=False,
    )
    subscription = resumed.subscribe()
    await resumed.start()
    received = []
    while True:
        event = await asyncio.wait_for(subscription.__anext__(), timeout=1)
        received.append(event)
        if event.type == EventType.RUNTIME_STARTED:
            break
    await subscription.aclose()

    replay = [event for event in received if event.correlation_id == "persisted"]
    assert replay[0].type == EventType.MESSAGE_USER
    assert replay[-1].type == EventType.ASSISTANT_COMPLETED
    assert received[-1].type == EventType.RUNTIME_STARTED
    assert resumed.history[-1].sequence > first.history[-1].sequence


@pytest.mark.asyncio
async def test_runtime_slash_commands_fork_resume_and_export_sessions(tmp_path):
    store = SessionStore(tmp_path / "sessions")
    runtime = VulcanoRuntime(
        scope=ExecutionScope(thread_id="thd_original", namespace="vulcano"),
        session_store=store,
        export_directory=tmp_path,
        stream_delay=0,
        extensions_enabled=False,
    )
    await runtime.dispatch(SubmitInput("before fork", correlation_id="before"))
    await runtime.dispatch(SubmitInput("/fork", correlation_id="fork"))

    fork_event = next(
        event for event in runtime.history if event.type == EventType.SESSION_SWITCHED
    )
    fork_thread = str(fork_event.payload["thread_id"])
    assert runtime.sessions.current_thread_id == fork_thread
    assert store.info(fork_thread).parent_thread_id == "thd_original"

    await runtime.dispatch(SubmitInput("after fork", correlation_id="after"))
    message = next(
        event
        for event in runtime.history
        if event.type == EventType.MESSAGE_USER and event.correlation_id == "after"
    )
    assert message.payload["scope"]["thread_id"] == fork_thread

    destination = tmp_path / "fork.md"
    await runtime.dispatch(SubmitInput(f"/export {destination}"))
    assert destination.is_file()
    assert "after fork" in destination.read_text(encoding="utf-8")

    await runtime.dispatch(SubmitInput("/resume thd_original"))
    assert runtime.sessions.current_thread_id == "thd_original"
    assert [
        event.payload["kind"]
        for event in runtime.history
        if event.type == EventType.SESSION_SWITCHED
    ] == ["fork", "resume"]


@pytest.mark.asyncio
async def test_runtime_new_command_opens_an_empty_session_tab(tmp_path):
    store = SessionStore(tmp_path / "sessions")
    runtime = VulcanoRuntime(
        scope=ExecutionScope(thread_id="thd_original", namespace="vulcano"),
        session_store=store,
        stream_delay=0,
        extensions_enabled=False,
    )
    await runtime.dispatch(SubmitInput("keep this in the original session"))

    await runtime.dispatch(SubmitInput("/new", correlation_id="new-session"))

    transition = [
        event for event in runtime.history if event.type == EventType.SESSION_SWITCHED
    ][-1]
    new_thread_id = str(transition.payload["thread_id"])
    assert transition.payload["kind"] == "new"
    assert transition.payload["events"] == []
    assert new_thread_id != "thd_original"
    assert runtime.sessions.current_thread_id == new_thread_id
    assert store.info(new_thread_id).parent_thread_id is None
    assert not any(
        event.type == EventType.MESSAGE_USER for event in store.load(new_thread_id)
    )
    assert [(tab.thread_id, tab.status) for tab in runtime.sessions.tabs] == [
        ("thd_original", "paused"),
        (new_thread_id, "active"),
    ]


@pytest.mark.asyncio
async def test_runtime_owns_session_tab_activation_pin_and_close(tmp_path):
    store = SessionStore(tmp_path / "sessions")
    store.ensure("thd_one")
    store.ensure("thd_two")
    runtime = VulcanoRuntime(
        scope=ExecutionScope(thread_id="thd_one", namespace="vulcano"),
        session_store=store,
        stream_delay=0,
        extensions_enabled=False,
    )
    await runtime.start()

    await runtime.dispatch(ToggleSessionPin("thd_one"))
    await runtime.dispatch(ActivateSessionTab("thd_two"))
    await runtime.dispatch(ActivateSessionTab("thd_two"))

    assert [tab.thread_id for tab in runtime.sessions.tabs] == [
        "thd_one",
        "thd_two",
    ]
    assert [tab.status for tab in runtime.sessions.tabs] == ["paused", "active"]
    assert runtime.sessions.tabs[0].pinned

    await runtime.dispatch(CloseSessionTab("thd_two"))

    assert runtime.sessions.current_thread_id == "thd_one"
    assert [(tab.thread_id, tab.status) for tab in runtime.sessions.tabs] == [
        ("thd_one", "active")
    ]

    await runtime.dispatch(CloseSessionTab("thd_one"))
    assert runtime.sessions.tabs == ()
    tabs_event = [
        event
        for event in runtime.history
        if event.type == EventType.SESSION_TABS_UPDATED
    ][-1]
    assert tabs_event.payload["active_thread_id"] is None
    assert tabs_event.payload["closed"] == {
        "thread_id": "thd_one",
        "pinned": False,
        "status": "idle",
    }

    await runtime.dispatch(SubmitInput("/echo reopen"))
    assert [(tab.thread_id, tab.status) for tab in runtime.sessions.tabs] == [
        ("thd_one", "active")
    ]
