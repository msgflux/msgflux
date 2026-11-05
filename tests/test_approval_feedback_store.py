import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict

import pytest

from msgflux.runtime import (
    ApprovalBinding,
    ApprovalConflictError,
    ApprovalExpiredError,
    ExecutionScope,
    PermissionSet,
    SQLiteApprovalStore,
    execution_context,
)
from msgflux.runtime.approvals.records import (
    MAX_APPROVAL_FEEDBACK_LENGTH,
    ApprovalEvent,
    ApprovalRecord,
    normalize_feedback,
)


def make_binding(**overrides):
    values = {
        "namespace": "tools",
        "thread_id": "thread",
        "run_id": "run",
        "principal": "user:42",
        "tool_call_id": "call:1",
        "tool_name": "write",
        "tool_revision": "impl:v1",
        "policy_version": "policy:v1",
        "arguments": {"path": "name"},
        "resources": {"root": "/workspace"},
        "required_permissions": ("filesystem.write",),
    }
    values.update(overrides)
    return ApprovalBinding.from_call(**values)


def scope_for(binding):
    return ExecutionScope(
        namespace=binding.namespace,
        thread_id=binding.thread_id,
        run_id=binding.run_id,
        principal=binding.principal,
        permissions=PermissionSet(binding.required_permissions),
    )


def request(store, binding):
    return store.request(binding, request_id="request", expires_at=200)


def test_normalize_feedback():
    assert normalize_feedback(None) is None
    assert normalize_feedback("  ok  ") == "ok"
    assert normalize_feedback("x" * MAX_APPROVAL_FEEDBACK_LENGTH) == (
        "x" * MAX_APPROVAL_FEEDBACK_LENGTH
    )
    for value in ("", "   ", "\t\n"):
        with pytest.raises(ValueError, match="blank"):
            normalize_feedback(value)
    for value in (42, b"text", ["text"], object()):
        with pytest.raises(TypeError):
            normalize_feedback(value)
    with pytest.raises(ValueError, match="maximum length"):
        normalize_feedback("x" * (MAX_APPROVAL_FEEDBACK_LENGTH + 1))
    # Record/event construction normalizes and rejects invalid feedback.
    record = ApprovalRecord(
        request_id="request",
        binding=make_binding(),
        expires_at=200,
        created_at=100,
        updated_at=100,
        feedback="  hi  ",
    )
    assert record.feedback == "hi"
    assert ApprovalEvent.from_record(record).feedback == "hi"
    with pytest.raises(ValueError, match="blank"):
        ApprovalRecord(
            request_id="request",
            binding=make_binding(),
            expires_at=200,
            created_at=100,
            updated_at=100,
            feedback=" ",
        )


@pytest.mark.parametrize("approved", [True, False])
def test_feedback_on_both_decisions_is_stored_atomically(approval_journal, approved):
    store, _ = approval_journal
    request(store, make_binding())
    decided = store.decide(
        "tools",
        "request",
        approved=approved,
        decided_by="reviewer:1",
        feedback="  looks good  ",
    )
    assert decided.feedback == "looks good"
    assert decided.decided_by == "reviewer:1"
    assert store.get("tools", "request").feedback == "looks good"
    events = store.events("tools", "request")
    assert events[-1].feedback == "looks good"
    assert events[-1].decided_by == "reviewer:1"
    assert events[0].feedback is None


@pytest.mark.parametrize(
    "feedback,error",
    [
        ("", ValueError),
        ("   ", ValueError),
        ("x" * (MAX_APPROVAL_FEEDBACK_LENGTH + 1), ValueError),
        (123, TypeError),
        (b"text", TypeError),
    ],
)
def test_invalid_feedback_leaves_no_state(approval_journal, feedback, error):
    store, _ = approval_journal
    request(store, make_binding())
    with pytest.raises(error):
        store.decide(
            "tools",
            "request",
            approved=True,
            decided_by="reviewer:1",
            feedback=feedback,
        )
    pending = store.get("tools", "request")
    assert pending.status == "pending"
    assert pending.feedback is None
    assert len(store.events("tools", "request")) == 1


@pytest.mark.parametrize("approved", [True, False])
def test_exact_retry_idempotent_but_changed_feedback_conflicts(
    approval_journal, approved
):
    store, _ = approval_journal
    pending = request(store, make_binding())
    decided = store.decide(
        "tools",
        "request",
        approved=approved,
        decided_by="reviewer:1",
        feedback=" first ",
        expected_revision=pending.revision,
    )
    # Same normalized feedback + same reviewer + stale CAS revision: idempotent.
    retry = store.decide(
        "tools",
        "request",
        approved=approved,
        decided_by="reviewer:1",
        feedback="first",
        expected_revision=pending.revision,
    )
    assert retry == decided
    assert len(store.events("tools", "request")) == 2
    # Changed feedback, different reviewer, or flipped decision all conflict.
    for kwargs in (
        {"feedback": "changed"},
        {"feedback": None},
        {"feedback": "first", "decided_by": "reviewer:2"},
        {"feedback": "first", "approved": not approved},
    ):
        with pytest.raises(ApprovalConflictError):
            store.decide(
                "tools",
                "request",
                approved=kwargs.pop("approved", approved),
                decided_by=kwargs.pop("decided_by", "reviewer:1"),
                feedback=kwargs.get("feedback"),
            )
    assert store.get("tools", "request") == decided
    assert len(store.events("tools", "request")) == 2


def test_accepted_feedback_survives_consume_and_expiry(approval_journal):
    store, clock = approval_journal
    binding = make_binding()
    request(store, binding)
    store.decide(
        "tools",
        "request",
        approved=True,
        decided_by="reviewer:1",
        feedback=" proceed carefully ",
    )
    with execution_context(scope=scope_for(binding)):
        consumed = store.consume("request", binding=binding)
    assert consumed.status == "consumed"
    assert consumed.feedback == "proceed carefully"
    assert consumed.decided_by == "reviewer:1"
    assert store.events("tools", "request")[-1].feedback == "proceed carefully"

    # A second, still-approvable request expires while retaining its feedback.
    store.request(
        make_binding(tool_call_id="call:2"), request_id="request2", expires_at=200
    )
    store.decide(
        "tools",
        "request2",
        approved=True,
        decided_by="reviewer:1",
        feedback=" meanwhile ",
    )
    clock[0] = 200
    with pytest.raises(ApprovalExpiredError):
        store.decide(
            "tools",
            "request2",
            approved=True,
            decided_by="reviewer:1",
            feedback=" meanwhile ",
        )
    expired = store.get("tools", "request2")
    assert expired.status == "expired"
    assert expired.feedback == "meanwhile"
    assert expired.decided_by == "reviewer:1"


def test_legacy_records_and_events_default_feedback_to_none():
    binding = make_binding()
    legacy_record = ApprovalRecord(
        request_id="request",
        binding=binding,
        expires_at=200,
        created_at=100,
        updated_at=100,
    )
    legacy_event = ApprovalEvent(
        namespace="tools",
        request_id="request",
        revision=1,
        status="pending",
        timestamp=100,
        decided_by=None,
    )
    assert legacy_record.feedback is None
    assert legacy_event.feedback is None
    # Legacy JSON without the new keys decodes the same way.
    payload = asdict(legacy_record)
    payload.pop("feedback")
    payload["binding"] = asdict(binding)
    assert ApprovalRecord(**{**payload, "binding": binding}).feedback is None
    event_payload = asdict(legacy_event)
    event_payload.pop("feedback")
    assert ApprovalEvent(**event_payload).feedback is None


def test_sqlite_legacy_payload_reopen_and_rollback(tmp_path):
    path = str(tmp_path / "approvals.db")
    binding = make_binding()
    store = SQLiteApprovalStore(path, clock=lambda: 100)
    request(store, binding)
    # Rewrite current payloads in legacy shape (no feedback key).
    for select, update in (
        (
            "SELECT payload FROM runtime_approvals",
            "UPDATE runtime_approvals SET payload=? WHERE payload=?",
        ),
        (
            "SELECT payload FROM runtime_approval_events",
            "UPDATE runtime_approval_events SET payload=? WHERE payload=?",
        ),
    ):
        rows = store._conn.execute(select).fetchall()
        for (payload,) in rows:
            values = json.loads(payload)
            values.pop("feedback", None)
            store._conn.execute(
                update,
                (json.dumps(values), payload),
            )
    store._conn.commit()
    assert store.get("tools", "request").feedback is None
    assert store.events("tools", "request")[0].feedback is None
    # Fault injection: audit failure rolls back record + feedback together.
    store._conn.execute(
        """CREATE TRIGGER fail_audit BEFORE INSERT ON runtime_approval_events
        BEGIN SELECT RAISE(ABORT, 'audit unavailable'); END"""
    )
    with pytest.raises(sqlite3.IntegrityError, match="audit unavailable"):
        store.decide(
            "tools",
            "request",
            approved=False,
            decided_by="reviewer:1",
            feedback=" blocked ",
        )
    pending = store.get("tools", "request")
    assert pending.status == "pending"
    assert pending.feedback is None
    assert len(store.events("tools", "request")) == 1
    store._conn.execute("DROP TRIGGER fail_audit")
    denied = store.decide(
        "tools",
        "request",
        approved=False,
        decided_by="reviewer:1",
        feedback=" blocked ",
    )
    assert denied.feedback == "blocked"
    store.close()
    # Reopen retains record and event feedback.
    store = SQLiteApprovalStore(path, clock=lambda: 100)
    assert store.get("tools", "request").feedback == "blocked"
    assert store.events("tools", "request")[-1].feedback == "blocked"
    store.close()


def test_concurrent_reviewers_only_one_feedback_wins(approval_journal):
    store, _ = approval_journal
    request(store, make_binding())

    def decide(index):
        try:
            return store.decide(
                "tools",
                "request",
                approved=True,
                decided_by=f"reviewer:{index}",
                feedback=f"note {index}",
            )
        except ApprovalConflictError as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(decide, range(2)))
    winners = [item for item in results if not isinstance(item, Exception)]
    assert len(winners) == 1
    assert sum(isinstance(item, ApprovalConflictError) for item in results) == 1
    final = store.get("tools", "request")
    assert final == winners[0]
    assert final.feedback == winners[0].feedback
    assert len(store.events("tools", "request")) == 2


@pytest.mark.asyncio
async def test_async_decide_with_feedback(approval_journal):
    store, _ = approval_journal
    binding = make_binding()
    await store.arequest(binding, request_id="request", expires_at=200)
    decided = await store.adecide(
        "tools",
        "request",
        approved=False,
        decided_by="reviewer:1",
        feedback="  no, see note ",
    )
    assert decided.feedback == "no, see note"
    retry = await store.adecide(
        "tools",
        "request",
        approved=False,
        decided_by="reviewer:1",
        feedback="no, see note",
        expected_revision=1,
    )
    assert retry == decided
    assert (await store.aevents("tools", "request"))[-1].feedback == "no, see note"
