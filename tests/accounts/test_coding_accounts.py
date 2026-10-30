"""HTTP and manifest wiring for managed coding account bundles."""

from unittest.mock import AsyncMock, Mock

import httpx2
import pytest

pytest.importorskip("litestar")

from msgflux.accounts import Accounts
from msgflux.core.dotdict import dotdict
from msgflux.models.response import ModelResponse
from msgflux.nn import Agent
from msgflux.runtime.context import (
    ExecutionScope,
    execution_context,
    get_execution_scope,
)
from msgflux.runtime.service import AgentService, AgentSession, SQLiteServiceStore
from msgflux.runtime.service.http import AgentServiceClient, create_service_app
from msgflux.runtime.service.local.manifest import load_service_manifest


def _response():
    response = ModelResponse()
    response.set_response_type("text_generation")
    response.add("account selected")
    response.reasoning = None
    return response


@pytest.mark.asyncio
async def test_http_prompt_forwards_explicit_and_default_account_names():
    seen = []
    service = AgentService(store=SQLiteServiceStore(), default_account="default")

    def make_session(_thread_id):
        model = Mock(model_type="chat_completion")
        agent = Agent(name="http-accounts", model=model)

        async def answer(**_kwargs):
            seen.append(get_execution_scope().account_name)
            return _response()

        agent.generator.aforward = AsyncMock(side_effect=answer)
        return AgentSession(agent)

    service.register("agent", make_session)
    app = create_service_app(service, token="service-token")
    transport = httpx2.ASGITransport(app=app)
    try:
        async with httpx2.AsyncClient(
            transport=transport, base_url="http://service"
        ) as http:
            client = AgentServiceClient(
                "http://service", token="service-token", client=http
            )
            thread = await client.open_thread("agent", thread_id="http-account")
            explicit = await client.prompt(
                thread.thread_id,
                "explicit",
                request_id="explicit",
                account_name="work",
            )
            assert (
                await service.wait(thread.thread_id, "explicit")
            ).status == "completed"
            omitted = await client.prompt(
                thread.thread_id, "omitted", request_id="omitted"
            )
            assert (
                await service.wait(thread.thread_id, "omitted")
            ).status == "completed"

        assert explicit.run_id != omitted.run_id
        assert seen == ["work", "default"]
        assert service.store.get(thread.thread_id, "explicit").account_name == "work"
        assert service.store.get(thread.thread_id, "omitted").account_name == "default"
    finally:
        await service.aclose()
        service.store.close()


@pytest.mark.asyncio
async def test_manifest_runner_injects_custom_accounts_root_into_parent_and_child(
    tmp_path, monkeypatch
):
    manifest_path = tmp_path / "config.toml"
    manifest_path.write_text(
        "[defaults]\n"
        'agent = "coding"\n'
        'account = "work"\n'
        'accounts_dir = "credential-bundles"\n'
        'agent_dir = "agent-state"\n'
        'model = "openai/gpt-4.1"\n'
        "\n[agents.coding]\n"
        'preset = "coding"\n'
        'subagents = ["groq-worker"]\n'
        "\n[agents.groq-worker]\n"
        'preset = "coding"\n'
        'model = "groq/openai/gpt-oss-20b"\n'
    )
    manifest = load_service_manifest(manifest_path)
    accounts_root = tmp_path / "credential-bundles"
    accounts = Accounts(accounts_root)
    accounts.set_api_key("openai", "work-openai-fake", "work")
    accounts.set_api_key("openai", "default-openai-fake", "default")
    accounts.set_api_key("groq", "work-groq-fake", "work")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    from msgflux.runtime.service.local.runner import _make_manifest_service

    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    service, resolved_path, _digest = _make_manifest_service(
        manifest_path, runtime_dir, tmp_path
    )
    try:
        assert service.default_account == "work"
        assert resolved_path == str(manifest_path.resolve())
        thread = await service.open_thread(
            "coding", thread_id="manifest-coding", cwd=tmp_path
        )
        async with service._use_session(thread.thread_id) as session:
            root_model = session.agent.model
            child = session.agent.tool_library.get_tool_definition(
                "groq-worker"
            ).executor.impl
            child_model = child.model
            root_resolver = root_model.credential_resolver
            child_resolver = child_model.credential_resolver
            assert root_resolver.provider == "openai"
            assert child_resolver.provider == "groq"
            assert root_resolver.accounts.root == accounts_root.resolve()
            assert child_resolver.accounts.root == accounts_root.resolve()
            assert manifest.defaults.accounts_dir == str(accounts_root.resolve())

            with execution_context(
                scope=ExecutionScope(
                    thread_id=thread.thread_id, run_id="run-work", account_name="work"
                )
            ):
                assert root_resolver.resolve(root_model).headers["Authorization"] == (
                    "Bearer work-openai-fake"
                )
                assert child_resolver.resolve(child_model).headers["Authorization"] == (
                    "Bearer work-groq-fake"
                )
            with execution_context(
                scope=ExecutionScope(
                    thread_id=thread.thread_id,
                    run_id="run-default",
                    account_name="default",
                )
            ):
                assert root_resolver.resolve(root_model).headers["Authorization"] == (
                    "Bearer default-openai-fake"
                )
    finally:
        await service.aclose()
        service.store.close()
