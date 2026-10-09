"""Focused validation for read-only foreground run inspection projections."""

import pytest

from msgflux.runtime.service.inspection import approval_evidence


def _checkpoint(pending):
    return {"runtime": {"extensions": {"pending_approvals": pending}}}


@pytest.mark.parametrize(
    "pending",
    [
        {"schema_version": 1},
        {"schema_version": 1, "requests": None},
        {"schema_version": 1, "requests": ["private-request"]},
    ],
)
def test_malformed_approval_requests_are_uncertain_without_echoing_values(pending):
    phase, count, reasons = approval_evidence(_checkpoint(pending))

    assert phase == "uncertain"
    assert count == 0
    assert reasons == ("Pending approval requests are malformed.",)
    assert "private-request" not in str((phase, count, reasons))


@pytest.mark.parametrize("phase_value", ["", 0, False])
def test_malformed_approval_phase_is_uncertain(phase_value):
    phase, count, reasons = approval_evidence(
        _checkpoint(
            {"schema_version": 1, "requests": {"call": "request"}, "phase": phase_value}
        )
    )

    assert phase == "uncertain"
    assert count == 1
    assert reasons == ("Pending approval phase is malformed.",)
    if phase_value:
        assert str(phase_value) not in str(reasons)


def test_unknown_approval_phase_is_normalized_to_uncertain():
    phase, count, reasons = approval_evidence(
        _checkpoint(
            {
                "schema_version": 1,
                "requests": {"call": "request"},
                "phase": "private-phase-value",
            }
        )
    )

    assert phase == "uncertain"
    assert count == 1
    assert reasons == ("Pending approval phase is unsupported.",)
    assert "private-phase-value" not in str((phase, count, reasons))


def test_empty_approval_batch_does_not_claim_a_host_decision_is_pending():
    assert approval_evidence(_checkpoint({"schema_version": 1, "requests": {}})) == (
        None,
        0,
        (),
    )
    assert approval_evidence(
        _checkpoint(
            {
                "schema_version": 1,
                "requests": {},
                "phase": "awaiting_decision",
            }
        )
    ) == ("awaiting_decision", 0, ())


def test_nonempty_approval_batch_without_phase_awaits_host_decision():
    assert approval_evidence(
        _checkpoint({"schema_version": 1, "requests": {"call": "request"}})
    ) == (
        "awaiting_decision",
        1,
        ("Pending approvals await a host decision.",),
    )
