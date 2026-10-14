"""Manifest-built coding agents retain accounts and approvals through restart."""

import asyncio
import json
import os

import httpx2
import pytest

pytest.importorskip("litestar")
pytestmark = pytest.mark.skipif(os.name != "posix", reason="Local coding workspace")

from msgflux.accounts import Accounts
from msgflux.coding import factory as coding_factory
from msgflux.models import Model
from msgflux.models.chat_transport import HTTPChatTransport
from msgflux.runtime.service import ServiceRecoveryRequiredError
from msgflux.runtime.service.http import (
    AgentServiceClient,
    AgentSessionClient,
    create_service_app,
)
from msgflux.runtime.service.local.runner import _make_manifest_service
from msgflux.utils.msgspec import msgspec_dumps


def _stream_response(delta, finish_reason):
    chunks = [
        {"choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}]},
    ]
    content = "".join(f"data: {msgspec_dumps(chunk)}\n\n" for chunk in chunks)
    return httpx2.Response(
        200,
        headers={"content-type": "text/event-stream"},
        content=(content + "data: [DONE]\n\n").encode(),
    )


@pytest.mark.asyncio
async def test_coding_factory_refuses_replaced_workspace_root(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    config = tmp_path / "config.toml"
    config.write_text(
        '[defaults]\nagent="coding"\nmodel="openai/gpt-4.1"\n'
        'accounts_dir="accounts"\n[agents.coding]\nharness="coding"\n'
        '[agents.coding.tools]\nactive=["workspace"]\n'
    )
    service = _make_manifest_service(config, runtime, root)[0]
    try:
        thread = await service.open_thread(
            "coding", thread_id="custom:thread", cwd=root
        )
        await service.workspace_policy(thread.thread_id)
    finally:
        await service.aclose()
        service.store.close()
    root.rename(tmp_path / "original-project")
    root.mkdir()
    service = _make_manifest_service(config, runtime, root)[0]
    try:
        with pytest.raises(PermissionError, match="resource/configuration mismatch"):
            await service.workspace_policy(thread.thread_id)
        assert list(root.iterdir()) == []
    finally:
        await service.aclose()
        service.store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("approved", [True, False])
async def test_manifest_coding_approval_and_account_survive_restart(
    tmp_path, monkeypatch, approved
):
    workspace = tmp_path / "project"
    workspace.mkdir()
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    accounts = Accounts(tmp_path / "accounts")
    accounts.set_api_key("openai", "fixture-work-key", account_name="work")
    accounts.set_api_key("openai", "fixture-default-key")
    config = tmp_path / "config.toml"
    config.write_text(
        '[defaults]\nagent="coding"\naccount="work"\n'
        'accounts_dir="accounts"\nmodel="openai/gpt-4.1"\n'
        '[agents.coding]\nharness="coding"\n'
        '[agents.coding.tools]\nactive=["workspace"]\n'
    )
    requests = []
    transports = []

    async def provider(request):
        body = json.loads(request.content)
        requests.append((request.headers["authorization"], body))
        assert body["stream"] is True
        if len(requests) == 1:
            return _stream_response(
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "patch-call",
                            "type": "function",
                            "function": {
                                "name": "apply_patch",
                                "arguments": msgspec_dumps(
                                    {
                                        "operation": "create",
                                        "path": "hello.txt",
                                        "diff": "+Olá Vulcano\n",
                                    }
                                ),
                            },
                        }
                    ]
                },
                "tool_calls",
            )
        return _stream_response({"content": "finished"}, "stop")

    def create_model(options, *, accounts):
        client = httpx2.AsyncClient(transport=httpx2.MockTransport(provider))
        transports.append(client)
        return Model.chat_completion(
            options.model,
            api_mode="chat_completions",
            retry=False,
            credential_resolver=accounts.resolver("openai"),
            chat_transport=HTTPChatTransport(async_client=client, max_retries=0),
        )

    monkeypatch.setattr(coding_factory, "_create_model", create_model)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    def make_service():
        return _make_manifest_service(config, runtime, workspace)[0]

    service = make_service()
    try:
        async with httpx2.AsyncClient(
            transport=httpx2.ASGITransport(
                app=create_service_app(service, token="fixture-service-token")
            )
        ) as http:
            client = AgentServiceClient(
                "http://service", token="fixture-service-token", client=http
            )
            session = await AgentSessionClient.open(
                client, agent_id="coding", cwd=workspace
            )
            policy = await session.workspace_policy()
            await session.update_workspace_policy(
                approval_policy="on-request", expected_revision=policy.revision
            )
            receipt = await session.prompt(
                "Create hello.txt", request_id="patch-request"
            )
            paused = await asyncio.wait_for(session.wait("patch-request"), 5)
            assert paused.status == "paused", paused.error
            assert not (workspace / "hello.txt").exists()
            (review,) = await session.approval_reviews(receipt.run_id)
            assert "Olá Vulcano" in review.diff
        await service.aclose()
        service.store.close()

        config.write_text(
            config.read_text().replace('account="work"', 'account="default"')
        )
        service = make_service()
        async with httpx2.AsyncClient(
            transport=httpx2.ASGITransport(
                app=create_service_app(service, token="fixture-service-token")
            )
        ) as http:
            client = AgentServiceClient(
                "http://service", token="fixture-service-token", client=http
            )
            session = await AgentSessionClient.open(client, thread_id=receipt.thread_id)
            (restored,) = await session.approval_reviews(receipt.run_id)
            assert restored == review
            await session.decide_approval(
                receipt.run_id,
                review.request_id,
                approved=approved,
                expected_revision=review.revision,
            )
            with pytest.raises(ServiceRecoveryRequiredError, match="quiescence"):
                await session.resume(receipt.run_id)
            # The old service was explicitly closed above. Only the trusted
            # host can assert quiescence; an HTTP client cannot supply it.
            await service.resume_checkpoint(
                receipt.thread_id, receipt.run_id, worker_stopped=True
            )
            completed = await asyncio.wait_for(session.wait("patch-request"), 5)
            assert completed.status == "completed"
            assert (
                await client.run_result(receipt.thread_id, receipt.run_id)
            ).output == "finished"
            if approved:
                assert (workspace / "hello.txt").read_text() == "Olá Vulcano"
            else:
                assert not (workspace / "hello.txt").exists()
            assert (
                service.store.get(receipt.thread_id, "patch-request").account_name
                == "work"
            )
            assert len(requests) == 2
            assert all(auth == "Bearer fixture-work-key" for auth, _body in requests)
            feedback = [
                message
                for message in requests[-1][1]["messages"]
                if message["role"] == "tool"
            ]
            assert len(feedback) == 1
            if not approved:
                assert "denied" in feedback[0]["content"].lower()
    finally:
        await service.aclose()
        service.store.close()
        for transport in transports:
            await transport.aclose()
