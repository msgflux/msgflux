import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Barrier

import msgspec
import pytest

from msgflux.data.stores import SQLiteCheckpointStore

from msgflux.runtime import (
    ApprovalConflictError,
    ExecutionEnvironment,
    ExecutionScope,
    InMemoryApprovalStore,
    InMemoryWorkspace,
    PermissionSet,
    PreparedFileChange,
    SQLiteApprovalStore,
    WorkspaceConflictError,
    WorkspaceEditor,
    execution_context,
)


def scope(fs, actions=("read", "write", "delete")):
    return ExecutionScope(
        namespace="editor",
        thread_id="thread",
        run_id="run",
        principal="user",
        environment=ExecutionEnvironment(fs),
        permissions=PermissionSet(
            resources=[
                fs.permission("/a", f"filesystem.{action}") for action in actions
            ]
        ),
    )


def request(editor, change, journal):
    binding = editor.approval_binding(
        change,
        tool_name="edit",
        tool_call_id="call",
        tool_revision="1",
        policy_version="1",
    )
    return journal.request(binding, request_id="request", expires_at=time.time() + 60)


@pytest.mark.parametrize("persistent", [False, True])
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_review_round_trip_and_apply(tmp_path, persistent, asynchronous):
    fs = InMemoryWorkspace("files", {"/a": b"old\n"})
    editor = WorkspaceEditor(fs)
    journal = (
        SQLiteApprovalStore(str(tmp_path / "approvals.db"))
        if persistent
        else InMemoryApprovalStore()
    )
    with execution_context(scope=scope(fs)):
        change = (
            await editor.aprepare_edit("/a", "old", "new")
            if asynchronous
            else editor.prepare_edit("/a", "old", "new")
        )
        assert fs.read_bytes("/a") == b"old\n"
        assert "-old\n+new\n" in change.diff
        restored = msgspec.json.decode(
            msgspec.json.encode(change), type=PreparedFileChange
        )
        assert restored == change
        record = request(editor, change, journal)
        with pytest.raises(ApprovalConflictError):
            editor.apply(restored, approval=record, approval_store=journal)
        journal.decide(
            "editor", record.request_id, approved=True, decided_by="reviewer"
        )
        if persistent:
            journal.close()
            journal = SQLiteApprovalStore(str(tmp_path / "approvals.db"))
        if asynchronous:
            await editor.aapply(restored, approval=record, approval_store=journal)
        else:
            editor.apply(restored, approval=record, approval_store=journal)
        assert fs.read_bytes("/a") == b"new\n"
        # Even restoring the exact original file cannot reuse the decision.
        fs.write_bytes("/a", b"old\n")
        with pytest.raises(ApprovalConflictError):
            editor.apply(restored, approval=record, approval_store=journal)
        assert fs.read_bytes("/a") == b"old\n"
    if persistent:
        journal.close()


def test_stale_changed_and_denied_reviews():
    fs = InMemoryWorkspace("files", {"/a": b"old"})
    editor, journal = WorkspaceEditor(fs), InMemoryApprovalStore()
    with execution_context(scope=scope(fs)):
        change = editor.prepare_write("/a", "new")
        record = request(editor, change, journal)
        journal.decide("editor", "request", approved=True, decided_by="reviewer")
        altered = msgspec.structs.replace(change, after="different")
        with pytest.raises(ApprovalConflictError):
            editor.apply(altered, approval=record, approval_store=journal)
        fs.write_text("/a", "concurrent")
        with pytest.raises(WorkspaceConflictError):
            editor.apply(change, approval=record, approval_store=journal)
        assert journal.get("editor", "request").status == "approved"
        assert fs.read_text("/a") == "concurrent"


def test_approval_required_and_live_authority_rechecked():
    fs = InMemoryWorkspace("files", {"/a": b"old"})
    editor, journal = WorkspaceEditor(fs), InMemoryApprovalStore()
    with execution_context(scope=scope(fs)):
        change = editor.prepare_write("/a", "new")
        record = request(editor, change, journal)
        journal.decide("editor", "request", approved=True, decided_by="reviewer")
        with pytest.raises(PermissionError, match="approval"):
            editor.apply(change)
    for current in (scope(fs, ("read",)), replace(scope(fs), principal="other")):
        with execution_context(scope=current):
            with pytest.raises((PermissionError, ApprovalConflictError)):
                editor.apply(change, approval=record, approval_store=journal)
    assert journal.get("editor", "request").status == "approved"


@pytest.mark.parametrize(
    "initial, final", [(None, ""), ("", None), (None, "new\n"), ("old\r\n", "new\r\n")]
)
def test_full_access_create_delete_and_newlines(initial, final):
    fs = InMemoryWorkspace("files", {} if initial is None else {"/a": initial.encode()})
    editor = WorkspaceEditor(fs, require_approval=False)
    with execution_context(scope=scope(fs)):
        change = (
            editor.prepare_delete("/a")
            if final is None
            else editor.prepare_write("/a", final)
        )
        assert change.operation == (
            "create" if initial is None else "delete" if final is None else "update"
        )
        editor.apply(change)
        if final is None:
            with pytest.raises(FileNotFoundError):
                fs.read_bytes("/a")
        else:
            assert fs.read_bytes("/a") == final.encode()


@pytest.mark.parametrize(
    "text, old", [("aaa", "aa"), ("a a", "a"), ("a", "missing"), ("a", "")]
)
def test_exact_edit_rejects_ambiguous_or_missing_matches(text, old):
    fs = InMemoryWorkspace("files", {"/a": text.encode()})
    with execution_context(scope=scope(fs)):
        with pytest.raises(ValueError):
            WorkspaceEditor(fs).prepare_edit("/a", old, "new")
        assert fs.read_text("/a") == text


def test_backend_without_atomic_support_fails_before_review():
    fs = InMemoryWorkspace("files", {"/a": b"old"})
    fs.supports_atomic_changes = False
    with execution_context(scope=scope(fs)):
        with pytest.raises(NotImplementedError):
            WorkspaceEditor(fs).prepare_write("/a", "new")


def test_two_writers_only_one_wins():
    fs = InMemoryWorkspace("files", {"/a": b"old"})
    editor = WorkspaceEditor(fs, require_approval=False)
    barrier = Barrier(2)

    def write(value):
        with execution_context(scope=scope(fs)):
            change = editor.prepare_write("/a", value)
            barrier.wait(timeout=3)
            try:
                editor.apply(change)
                return "applied"
            except WorkspaceConflictError:
                return "conflict"

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(write, ["one", "two"])) == ["applied", "conflict"]


def test_cas_catches_race_after_approval_consumption():
    fs = InMemoryWorkspace("files", {"/a": b"old"})
    editor, journal = WorkspaceEditor(fs), InMemoryApprovalStore()
    with execution_context(scope=scope(fs)):
        change = editor.prepare_write("/a", "new")
        record = request(editor, change, journal)
        journal.decide("editor", "request", approved=True, decided_by="reviewer")
        consume = journal.consume

        def race(*args, **kwargs):
            result = consume(*args, **kwargs)
            fs.write_text("/a", "concurrent")
            return result

        journal.consume = race
        with pytest.raises(WorkspaceConflictError):
            editor.apply(change, approval=record, approval_store=journal)
        assert fs.read_text("/a") == "concurrent"
        assert journal.get("editor", "request").status == "consumed"


def test_preview_missing_newline_and_strict_deserialization():
    change = PreparedFileChange(
        workspace_id="files", path="/a", before="old", after="new"
    )
    assert change.diff.count("\\ No newline at end of file") == 2
    for payload in (
        b'{"workspace_id":"files","path":"/../a","before":"old","after":"new"}',
        msgspec.json.encode(msgspec.to_builtins(change) | {"schema_version": 2}),
        msgspec.json.encode(msgspec.to_builtins(change) | {"authority": True}),
    ):
        with pytest.raises(msgspec.ValidationError):
            msgspec.json.decode(payload, type=PreparedFileChange)


@pytest.mark.parametrize("decision", ["denied", "expired"])
def test_terminal_decision_cannot_apply(decision):
    fs = InMemoryWorkspace("files", {"/a": b"old"})
    editor, journal = WorkspaceEditor(fs), InMemoryApprovalStore()
    with execution_context(scope=scope(fs)):
        change = editor.prepare_write("/a", "new")
        record = request(editor, change, journal)
        if decision == "denied":
            journal.decide("editor", "request", approved=False, decided_by="reviewer")
        else:
            journal._clock = lambda: time.time() + 120
        with pytest.raises(ApprovalConflictError):
            editor.apply(change, approval=record, approval_store=journal)
        assert fs.read_bytes("/a") == b"old"


def test_detached_preview_survives_checkpoint_restart(tmp_path):
    path = str(tmp_path / "checkpoint.db")
    fs = InMemoryWorkspace("files", {"/a": b"old"})
    with execution_context(scope=scope(fs)):
        change = WorkspaceEditor(fs).prepare_delete("/a")
    store = SQLiteCheckpointStore(path)
    store.save_state(
        "editor",
        "thread",
        "run",
        {"status": "paused", "prepared_change": msgspec.to_builtins(change)},
    )
    store.close()
    store = SQLiteCheckpointStore(path)
    restored = msgspec.convert(
        store.load_state("editor", "thread", "run")["prepared_change"],
        type=PreparedFileChange,
    )
    assert restored.diff == change.diff
    assert restored.digest == change.digest
    # The proposal carries data, never live permissions.
    with pytest.raises(PermissionError):
        WorkspaceEditor(fs, require_approval=False).apply(restored)
    store.close()


@pytest.mark.asyncio
async def test_async_create_delete_and_atomic_primitive():
    fs = InMemoryWorkspace("files")
    editor = WorkspaceEditor(fs, require_approval=False)
    with execution_context(scope=scope(fs)):
        change = await editor.aprepare_write("/a", "")
        assert "/dev/null" in change.diff
        await editor.aapply(change)
        with pytest.raises(WorkspaceConflictError):
            await fs.acompare_exchange("/a", expected=None, replacement=b"other")
        await editor.aapply(await editor.aprepare_delete("/a"))
        await fs.acompare_exchange("/a", expected=None, replacement=b"created")
        assert fs.read_bytes("/a") == b"created"
