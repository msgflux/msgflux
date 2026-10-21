"""Opt-in live checks for Codex native manual compaction.

Run with ``MSGFLUX_LIVE_CODEX_COMPACTION=1`` and a configured Codex account.
The SDK test uses a temporary SQLite checkpoint store; the service test exercises
the HTTP facade and approval recovery against an in-process ASGI app.
"""

from __future__ import annotations

import os
from uuid import uuid4

import httpx2
import pytest

import msgflux as mf
from msgflux.chat_messages import ChatMessages
from msgflux.data.stores import SQLiteCheckpointStore
from msgflux.models.providers.openai_codex import CodexChatTransport
from msgflux.nn import Agent, CompactionExtension
from msgflux.runtime import ExecutionScope
from msgflux.runtime.service.http.client import AgentServiceClient
from msgflux.runtime.service.http.factory import create_service_app
from msgflux.runtime.service.http.session import AgentSessionClient
from tests.test_agent_service_approvals import _close_service
from tests.test_agent_service_approvals import _fixture


_OPT_IN = "MSGFLUX_LIVE_CODEX_COMPACTION"

pytestmark = pytest.mark.skipif(
    os.getenv(_OPT_IN) != "1",
    reason=f"Set {_OPT_IN}=1 to run billable live Codex compaction checks",
)


def _live_model():
    from msgflux.accounts import Accounts

    account_name = os.getenv("MSGFLUX_LIVE_SERVICE_ACCOUNT", "work")
    model_id = os.getenv("MSGFLUX_LIVE_CODEX_MODEL", "gpt-6-luna")
    return mf.Model.chat_completion(
        f"openai-codex/{model_id}",
        api_mode="responses",
        reasoning_effort="medium",
        credential_resolver=Accounts().resolver(
            "openai-codex", account_name=account_name
        ),
        chat_transport=CodexChatTransport(timeout=120, max_retries=0),
        retry=False,
    )


def _paused_checkpoint(scope: ExecutionScope) -> dict:
    messages = ChatMessages(thread_id=scope.thread_id, namespace=scope.namespace)
    messages.begin_turn(turn_id="prior_completed_turn")
    messages.add_user("Remember the fictional archive label for shelf R-17.")
    messages.add_assistant(
        "The fictional archive label for shelf R-17 is VELVET-ORCHID."
    )
    messages.end_turn()
    messages.begin_turn(turn_id=scope.run_id)
    messages.add_user(
        "What was the fictional archive label for shelf R-17? Give the exact label only."
    )
    return {
        "schema_version": 1,
        "status": "paused",
        "scope": scope.to_dict(),
        "messages": messages._to_state(),
        "model_preference": None,
        "runtime": {
            "schema_version": 1,
            "namespace": scope.namespace,
            "thread_id": scope.thread_id,
            "run_id": scope.run_id,
            "branch_id": "root",
            "budgets": {},
            "extensions": {},
        },
        "metadata": {"purpose": "live native compaction continuity check"},
    }


@pytest.mark.asyncio
async def test_sdk_compaction_survives_sqlite_reopen_and_resumes_from_opaque_view(
    tmp_path,
):
    model = _live_model()
    database = tmp_path / "codex-compaction-checkpoint.sqlite3"
    namespace = "codex_manual_compaction_live"
    scope = ExecutionScope(
        namespace=namespace,
        thread_id=f"codex-compact-{uuid4().hex}",
        run_id=f"paused-{uuid4().hex}",
        account_name=os.getenv("MSGFLUX_LIVE_SERVICE_ACCOUNT", "work"),
    )
    first_store = SQLiteCheckpointStore(database)
    agent = Agent(
        name=namespace,
        model=model,
        checkpoint_store=first_store,
        system_prompt=(
            "Preserve all unique facts, identifiers, and codes when compacting. "
            "Do not use tools. Answer factual questions briefly and exactly."
        ),
        extensions=[CompactionExtension()],
    )
    before = _paused_checkpoint(scope)
    first_store.save_state(namespace, scope.thread_id, scope.run_id, before)
    before = first_store.load_state(namespace, scope.thread_id, scope.run_id)
    try:
        result = await agent.acompact(scope=scope)
        assert result.status == "completed"
        assert result.reason == "manual"
        after = first_store.load_state(namespace, scope.thread_id, scope.run_id)
        assert after["status"] == "paused"
        assert after["scope"] == before["scope"]
        assert after["messages"]["items"][:-1] == before["messages"]["items"]
        operation = after["messages"]["items"][-1]
        assert operation["type"] == "compaction"
        assert operation["views"][0]["format"] == "provider"
        assert operation["views"][0]["provider"] == "openai-codex"
        assert operation["views"][0]["api_mode"] == "responses"

        # The provider view contains the original user prompt but omits the
        # assistant-only fact, so the continuation must use the opaque summary.
        compacted = operation["views"][0]["items"]
        assert not any("VELVET-ORCHID" in str(item) for item in compacted), (
            "the private source fact should be carried only by opaque provider state"
        )
        assert any(
            item.get("role") == "user"
            and "fictional archive label for shelf R-17" in str(item.get("content"))
            for item in compacted
        )
        opaque_items = [item for item in compacted if item.get("type") == "compaction"]
        assert len(opaque_items) == 1
        assert isinstance(opaque_items[0].get("encrypted_content"), str)
        assert opaque_items[0]["encrypted_content"]

        first_store.close()
        reopened_store = SQLiteCheckpointStore(database)
        resumed_agent = Agent(
            name=namespace,
            model=model,
            checkpoint_store=reopened_store,
            system_prompt=(
                "Preserve all unique facts, identifiers, and codes when compacting. "
                "Do not use tools. Answer factual questions briefly and exactly."
            ),
            extensions=[CompactionExtension()],
        )
        try:
            reopened_state = reopened_store.load_state(
                namespace, scope.thread_id, scope.run_id
            )
            reopened_history = ChatMessages()
            reopened_history._hydrate_state(reopened_state["messages"])
            projected_input = model.api_mode_capabilities.context_adapter._input(
                model, reopened_history
            )
            projected_compactions = [
                item for item in projected_input if item.get("type") == "compaction"
            ]
            assert [item.get("type") for item in projected_input] == [
                "message",
                "compaction",
                "message",
            ]
            assert len(projected_compactions) == 1
            assert (
                projected_compactions[0]["encrypted_content"]
                == opaque_items[0]["encrypted_content"]
            )
            assert not any("VELVET-ORCHID" in str(item) for item in projected_input)
            answer = await resumed_agent.acall(scope=scope)
            assert "VELVET-ORCHID" in answer
            resumed = reopened_store.load_state(
                namespace, scope.thread_id, scope.run_id
            )
            assert resumed["status"] == "completed"
        finally:
            await resumed_agent.aclose()
            reopened_store.close()
    finally:
        await agent.aclose()
        first_store.close()
        await model.aclose()


@pytest.mark.asyncio
async def test_service_http_compaction_preserves_pending_approval_and_resume_once():
    account_name = os.getenv("MSGFLUX_LIVE_SERVICE_ACCOUNT", "work")
    service, agent, _approvals, calls, _scope_options = _fixture()
    # Bind the fixture's paused admission to the same account as the model.
    service.default_account = account_name
    thread = await service.open_thread("main", thread_id="codex-approval-review")
    receipt = await service.prompt(
        thread.thread_id,
        "lookup",
        request_id="codex-approval-review-request",
        account_name=account_name,
    )
    paused = await service.wait(thread.thread_id, receipt.request_id)
    assert paused.status == "paused"
    review = (await service.approval_reviews(thread.thread_id, receipt.run_id))[0]
    agent.register_extension("compaction", CompactionExtension())
    state = agent.checkpoint_store.load_state(
        agent.get_module_name(), thread.thread_id, receipt.run_id
    )
    previous = ChatMessages(
        thread_id=thread.thread_id, namespace=agent.get_module_name()
    )
    previous.begin_turn(turn_id="previous")
    previous.add_user("Prior request")
    previous.add_assistant("Prior decision")
    previous.end_turn()
    state["messages"]["items"] = previous.to_items() + state["messages"]["items"]
    agent.checkpoint_store.commit_state(
        agent.get_module_name(),
        thread.thread_id,
        receipt.run_id,
        state,
        expected_revision=state["_checkpoint"]["revision"],
    )
    model = _live_model()
    for name in (
        "count_context_tokens",
        "acount_context_tokens",
        "compact_context",
        "acompact_context",
    ):
        setattr(agent.model, name, getattr(model, name))
    agent.model.context_capacity = 100_000
    agent.model.provider = model.provider
    agent.model.api_mode = model.api_mode
    agent.model.reasoning_codec = model.reasoning_codec
    app = create_service_app(service, token="test-token")
    before = agent.checkpoint_store.load_state(
        agent.get_module_name(), thread.thread_id, receipt.run_id
    )
    try:
        async with httpx2.AsyncClient(
            transport=httpx2.ASGITransport(app=app)
        ) as transport:
            client = AgentServiceClient(
                "http://test", token="test-token", client=transport
            )
            session = await AgentSessionClient.open(client, thread_id=thread.thread_id)
            result = await session.compact(run_id=receipt.run_id)
            assert result.status == "completed"
            assert result.run_id == receipt.run_id
            after = agent.checkpoint_store.load_state(
                agent.get_module_name(), thread.thread_id, receipt.run_id
            )
            assert after["status"] == "paused"
            assert after["messages"]["items"][:-1] == before["messages"]["items"]
            operation = after["messages"]["items"][-1]
            assert operation["views"][0]["format"] == "provider"
            assert operation["views"][0]["provider"] == "openai-codex"
            assert (await session.approval_reviews(receipt.run_id))[0] == review
            assert calls == []

            await session.decide_approval(
                receipt.run_id,
                review.request_id,
                approved=True,
                expected_revision=review.revision,
            )
            await session.resume(receipt.run_id)
            finished = await session.wait(receipt.request_id)
            assert finished.status == "completed"
            assert calls == ["sensitive query"]
    finally:
        await _close_service(service)
        await model.aclose()
