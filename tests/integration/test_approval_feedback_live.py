"""Opt-in real inference through HTTP approval review and SQLite recovery.

Set MSGFLUX_LIVE_APPROVAL_FEEDBACK=1 and the target credentials explicitly.
Run this file directly; no mocks replace the model or its HTTP transport.
"""

import asyncio
import json
import os
from contextlib import asynccontextmanager
from uuid import uuid4

import httpx2
import pytest

pytest.importorskip("litestar")

from msgflux import Model
from msgflux.data.stores import SQLiteCheckpointStore
from msgflux.nn import Agent
from msgflux.runtime import AgentApprovals, SQLiteApprovalStore
from msgflux.runtime.service import AgentService, AgentSession, SQLiteServiceStore
from msgflux.runtime.service.http import (
    AgentServiceClient,
    AgentSessionClient,
    create_service_app,
)

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(
        os.getenv("MSGFLUX_LIVE_APPROVAL_FEEDBACK") != "1",
        reason="Real provider calls require explicit opt-in",
    ),
]

TARGETS = [
    (
        "go-chat",
        "opencode-go/deepseek-v4.1-flash",
        {},
        "OPENCODE_GO_API_KEY",
    ),
    (
        "go-anthropic",
        "opencode-go/claude-haiku-5-5",
        {"api_mode": "anthropic_messages"},
        "OPENCODE_GO_API_KEY",
    ),
    ("gemini", "gemini/gemini-3.5-flash", {}, "GEMINI_API_KEY"),
    ("anthropic-direct", "anthropic/claude-haiku-5-5", {}, "ANTHROPIC_API_KEY"),
]


@asynccontextmanager
async def _server(path, model_path, options, effects):
    def record_probe(value: int) -> dict:
        """Record the requested integer once and return a structured observation."""
        effects.append(value)
        return {"observed": value, "ok": True}

    checkpoints = SQLiteCheckpointStore(path / "checkpoints.db")
    approvals = SQLiteApprovalStore(path / "approvals.db")
    store = SQLiteServiceStore(path / "service.db")
    model = Model.chat_completion(model_path, max_tokens=1024, **options)
    agent = Agent(
        name="live-feedback",
        model=model,
        tools=[record_probe],
        system_prompt=(
            "You are an integration test assistant. Call record_probe exactly once "
            "when initially asked. After its outcome and approval reviewer feedback "
            "arrive, follow the reviewer's next instruction. If asked to correct a "
            "denied call, propose a new call with the requested arguments. If asked "
            "for a final response, do not call any more tools."
        ),
        checkpoint_store=checkpoints,
        approvals=AgentApprovals(approvals, {"record_probe": "v1"}, "p1"),
    )
    service = AgentService(store=store)
    service.register(
        "main",
        lambda _thread: AgentSession(
            agent,
            checkpoint_store=checkpoints,
            approval_reviewer="live-reviewer",
            scope_factory=lambda scope: scope.with_overrides(principal="executor"),
        ),
    )
    transport = httpx2.ASGITransport(
        app=create_service_app(service, token="live-token")
    )
    try:
        async with httpx2.AsyncClient(transport=transport) as http:
            client = AgentServiceClient(
                "http://live-test", token="live-token", client=http
            )
            yield service, client, checkpoints
    finally:
        await service.aclose()
        await model.aclose()
        checkpoints.close()
        approvals.close()
        store.close()


@pytest.mark.parametrize("approved", [False, True])
@pytest.mark.parametrize(
    "target,model_path,options,key", TARGETS, ids=[t[0] for t in TARGETS]
)
async def test_real_feedback_decision_survives_restart(
    tmp_path, approved, target, model_path, options, key
):
    if not os.getenv(key):
        pytest.skip(f"{key} is required for {target}")
    effects = []
    marker = "REVIEW_" + uuid4().hex[:12]
    feedback = f"Do not call tools again. Reply with exactly {marker} and nothing else."
    async with _server(tmp_path, model_path, options, effects) as (_, client, _):
        session = await AgentSessionClient.open(
            client, agent_id="main", thread_id="live"
        )
        admission = await session.prompt(
            "Call record_probe with value=7 exactly once, then follow reviewer feedback.",
            request_id="probe",
        )
        paused = await asyncio.wait_for(session.wait("probe"), timeout=90)
        assert paused.status == "paused", (paused.status, paused.error)
        (review,) = await session.approval_reviews(admission.run_id)
        assert effects == []
        decided = await session.decide_approval(
            admission.run_id,
            review.request_id,
            approved=approved,
            expected_revision=review.revision,
            feedback=feedback,
        )
        assert decided.feedback == feedback
        assert effects == []

    # New Agent, real model client and all three SQLite stores; no live owner remains.
    async with _server(tmp_path, model_path, options, effects) as (service, client, cp):
        session = await AgentSessionClient.open(client, thread_id="live")
        (restored,) = await session.approval_reviews(admission.run_id)
        assert restored.feedback == feedback
        await service.resume_checkpoint("live", admission.run_id, worker_stopped=True)
        completed = await asyncio.wait_for(session.wait("probe"), timeout=90)
        assert completed.status == "completed", (completed.status, completed.error)
        result = await session.result(admission.run_id)
        assert marker in result.output, result.output
        assert effects == ([7] if approved else [])
        state = cp.load_state("live-feedback", "live", admission.run_id)
        items = state["messages"]["items"]
        (comment,) = [
            item for item in items if "approval_feedback" in item.get("metadata", {})
        ]
        (identity,) = comment["metadata"]["approval_feedback"]["reviews"]
        assert identity["feedback"] == feedback
        assert identity["approved"] is approved
        assert identity["decided_by"] == "live-reviewer"
        (output,) = [
            item for item in items if item.get("type") == "function_call_output"
        ]
        assert items.index(output) < items.index(comment)
        if approved:
            assert json.loads(output["output"]) == {"observed": 7, "ok": True}
        else:
            assert output["output"] == "Tool approval is denied"
        assert marker not in output["output"]
        print(
            json.dumps(
                {
                    "target": target,
                    "approved": approved,
                    "restart": True,
                    "feedback_obeyed": True,
                    "effects": len(effects),
                }
            )
        )


@pytest.mark.parametrize(
    "target,model_path,options,key", TARGETS, ids=[t[0] for t in TARGETS]
)
async def test_real_denied_feedback_produces_a_new_reviewable_call(
    tmp_path, target, model_path, options, key
):
    if not os.getenv(key):
        pytest.skip(f"{key} is required for {target}")
    effects = []
    marker = "CORRECTED_" + uuid4().hex[:12]
    async with _server(tmp_path, model_path, options, effects) as (service, client, cp):
        session = await AgentSessionClient.open(
            client, agent_id="main", thread_id="live"
        )
        admission = await session.prompt(
            "Call record_probe with value=7.", request_id="probe"
        )
        paused = await asyncio.wait_for(session.wait("probe"), timeout=90)
        assert paused.status == "paused", (paused.status, paused.error)
        (first,) = await session.approval_reviews(admission.run_id)
        await session.decide_approval(
            admission.run_id,
            first.request_id,
            approved=False,
            expected_revision=first.revision,
            feedback="Wrong value. Call record_probe with value=11 instead. Do not answer yet.",
        )
        await service.resume_checkpoint("live", admission.run_id)
        paused_again = await asyncio.wait_for(session.wait("probe"), timeout=90)
        assert paused_again.status == "paused", (
            paused_again.status,
            paused_again.error,
        )
        reviews = await session.approval_reviews(admission.run_id)
        (second,) = [review for review in reviews if review.status == "pending"]
        assert second.request_id != first.request_id
        assert effects == []
        state = cp.load_state("live-feedback", "live", admission.run_id)
        pending = state["runtime"]["extensions"]["pending_approvals"]
        assert len(pending["intents"]) == 1
        assert pending["intents"][0]["arguments"] == {"value": 11}
        await session.decide_approval(
            admission.run_id,
            second.request_id,
            approved=True,
            expected_revision=second.revision,
            feedback=f"Run exactly these arguments. Then reply with exactly {marker}. Do not call tools again.",
        )
        await service.resume_checkpoint("live", admission.run_id)
        completed = await asyncio.wait_for(session.wait("probe"), timeout=90)
        assert completed.status == "completed", (completed.status, completed.error)
        assert marker in (await session.result(admission.run_id)).output
        assert effects == [11]
        state = cp.load_state("live-feedback", "live", admission.run_id)
        comments = [
            item
            for item in state["messages"]["items"]
            if "approval_feedback" in item.get("metadata", {})
        ]
        assert len(comments) == 2
        assert [
            item["metadata"]["approval_feedback"]["reviews"][0]["request_id"]
            for item in comments
        ] == [first.request_id, second.request_id]
        print(
            json.dumps(
                {
                    "target": target,
                    "replanned": True,
                    "old_effects": 0,
                    "corrected_effects": effects,
                    "feedback_obeyed": True,
                }
            )
        )
