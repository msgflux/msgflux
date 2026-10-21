"""End-to-end service account selection through one shared model transport."""

import asyncio
import json

import httpx2
import pytest
import msgflux as mf

from msgflux.accounts.resolver import AccountCredentialResolver
from msgflux.accounts.store import Accounts
from msgflux.models import Model
from msgflux.models.chat_transport import HTTPChatTransport
from msgflux.nn import Agent
from msgflux.runtime.service import AgentService, AgentSession, SQLiteServiceStore
from msgflux.runtime.service.records import ServiceThread
from msgflux.tools.builtin import AgentTool


def test_dotted_account_name_is_accepted_by_store_and_service(tmp_path):
    accounts = Accounts(tmp_path / "accounts")
    accounts.set_api_key("openai", "fake-key", "team.work")
    assert accounts.get("openai", "team.work").value == "fake-key"

    store = SQLiteServiceStore()
    try:
        service = AgentService(store=store, default_account="team.work")
        assert service.default_account == "team.work"
    finally:
        store.close()


def _model(accounts, client):
    return Model.chat_completion(
        "openai/gpt-4.1",
        api_mode="chat_completions",
        base_url="https://api.example.test/v1",
        retry=False,
        credential_resolver=AccountCredentialResolver(accounts, "openai"),
        chat_transport=HTTPChatTransport(async_client=client, max_retries=0),
    )


def _chat_response():
    return {
        "choices": [
            {
                "message": {"role": "assistant", "content": "done"},
                "finish_reason": "stop",
            }
        ]
    }


@pytest.mark.asyncio
async def test_two_threads_use_selected_accounts_with_shared_model_and_http_client(
    tmp_path,
):
    accounts = Accounts(tmp_path / "accounts")
    accounts.set_api_key("openai", "fake-default-key", "default")
    accounts.set_api_key("openai", "fake-work-key", "work")
    captured = []
    rendezvous = asyncio.Event()
    waiting = 0

    async def handler(request):
        nonlocal waiting
        captured.append(request.headers["authorization"])
        waiting += 1
        if waiting == 2:
            rendezvous.set()
        await asyncio.wait_for(rendezvous.wait(), timeout=2)
        await asyncio.sleep(0)
        return httpx2.Response(200, json=_chat_response())

    client = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    shared_model = _model(accounts, client)
    store = SQLiteServiceStore()
    service = AgentService(store=store, default_account="default")

    def make_session(thread_id):
        return AgentSession(Agent(name=f"worker-{thread_id}", model=shared_model))

    service.register("agent", make_session)
    work_thread = await service.open_thread("agent", thread_id="work-thread")
    default_thread = await service.open_thread("agent", thread_id="default-thread")
    try:
        await asyncio.gather(
            service.prompt(
                work_thread.thread_id,
                "work request",
                request_id="work-request",
                account_name="work",
            ),
            service.prompt(
                default_thread.thread_id,
                "default request",
                request_id="default-request",
            ),
        )
        work_receipt, default_receipt = await asyncio.gather(
            service.wait(work_thread.thread_id, "work-request"),
            service.wait(default_thread.thread_id, "default-request"),
        )
        assert work_receipt.status == default_receipt.status == "completed"
        assert sorted(captured) == [
            "Bearer fake-default-key",
            "Bearer fake-work-key",
        ]
        assert shared_model.chat_transport.http._async_client is client
    finally:
        await service.aclose()
        await client.aclose()
        store.close()


@pytest.mark.asyncio
async def test_legacy_admission_with_no_account_does_not_fall_back_to_default(
    tmp_path,
):
    accounts = Accounts(tmp_path / "accounts")
    accounts.set_api_key("openai", "fake-default-key", "default")
    captured = []

    async def handler(request):
        captured.append(request)
        return httpx2.Response(200, json=_chat_response())

    client = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    shared_model = _model(accounts, client)
    store = SQLiteServiceStore()
    store.bind_thread(ServiceThread("legacy-thread", "agent"))
    store.admit("legacy-thread", "legacy-request", "resume me", "legacy-agent", None)
    service = AgentService(store=store, default_account="default")
    service.register(
        "agent",
        lambda _thread_id: AgentSession(Agent(name="legacy-agent", model=shared_model)),
    )
    try:
        await service.resume("legacy-thread", "legacy-request", worker_stopped=True)
        result = await service.wait("legacy-thread", "legacy-request")
        assert result.status == "failed"
        assert "no recorded account selection" in result.error
        assert captured == []
    finally:
        await service.aclose()
        await client.aclose()
        store.close()


@pytest.mark.asyncio
async def test_restarted_service_resumes_checkpoint_with_persisted_account(tmp_path):
    accounts = Accounts(tmp_path / "accounts")
    accounts.set_api_key("openai", "fake-default-key", "default")
    accounts.set_api_key("openai", "fake-work-key", "work")
    captured = []

    async def handler(request):
        captured.append(request.headers["authorization"])
        return httpx2.Response(200, json=_chat_response())

    client = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    service_path = tmp_path / "service.sqlite3"
    agent_dir = tmp_path / "agent"
    first_store = SQLiteServiceStore(service_path)
    first_service = AgentService(store=first_store, default_account="default")
    first_model = _model(accounts, client)
    first_agent = Agent(name="restarted", model=first_model, agent_dir=agent_dir)
    first_service.register("agent", lambda _thread: AgentSession(first_agent))
    thread = await first_service.open_thread("agent", thread_id="restart-account")
    try:
        await first_service.prompt(
            thread.thread_id,
            "initial prompt",
            request_id="restart-request",
            account_name="work",
        )
        first_result = await first_service.wait(thread.thread_id, "restart-request")
        assert first_result.status == "completed"
        checkpoint_store = first_agent._owned_threads[
            thread.thread_id
        ].resources.checkpoint_store
        checkpoint = checkpoint_store.load_state(
            first_agent.get_module_name(), thread.thread_id, first_result.run_id
        )
        assert checkpoint["scope"]["account_name"] == "work"
        checkpoint = dict(checkpoint)
        checkpoint["status"] = "paused"
        messages = dict(checkpoint["messages"])
        messages["items"] = messages["items"][:-1]
        checkpoint["messages"] = messages
        checkpoint_store.save_state(
            first_agent.get_module_name(),
            thread.thread_id,
            first_result.run_id,
            checkpoint,
        )
        first_store._connection.execute(
            "UPDATE service_admissions SET status='paused' WHERE thread_id=? AND request_id=?",
            (thread.thread_id, "restart-request"),
        )
        first_store._connection.commit()
    finally:
        await first_service.aclose()
        first_store.close()

    restarted_store = SQLiteServiceStore(service_path)
    restarted_service = AgentService(store=restarted_store, default_account="default")
    restarted_model = _model(accounts, client)
    restarted_agent = Agent(
        name="restarted", model=restarted_model, agent_dir=agent_dir
    )
    restarted_service.register("agent", lambda _thread: AgentSession(restarted_agent))
    try:
        record = restarted_store.get_for_run(thread.thread_id, first_result.run_id)
        assert record is not None
        await restarted_service.resume(
            thread.thread_id, record.receipt.request_id, worker_stopped=True
        )
        result = await restarted_service.wait(thread.thread_id, "restart-request")
        assert result.status == "completed", result.error
        assert captured == ["Bearer fake-work-key", "Bearer fake-work-key"]
        assert (
            restarted_agent._owned_threads[
                thread.thread_id
            ].resources.checkpoint_store.load_state(
                restarted_agent.get_module_name(), thread.thread_id, result.run_id
            )["scope"]["account_name"]
            == "work"
        )
    finally:
        await restarted_service.aclose()
        restarted_store.close()
        await client.aclose()


@pytest.mark.asyncio
async def test_restored_legacy_checkpoint_does_not_fall_back_to_default(tmp_path):
    accounts = Accounts(tmp_path / "accounts")
    accounts.set_api_key("openai", "fake-default-key", "default")
    accounts.set_api_key("openai", "fake-work-key", "work")
    captured = []

    async def handler(request):
        captured.append(request.headers["authorization"])
        return httpx2.Response(200, json=_chat_response())

    client = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    service_path = tmp_path / "service.sqlite3"
    agent_dir = tmp_path / "agent"
    first_store = SQLiteServiceStore(service_path)
    first_service = AgentService(store=first_store, default_account="default")
    first_model = _model(accounts, client)
    first_agent = Agent(name="legacy-restore", model=first_model, agent_dir=agent_dir)
    first_service.register("agent", lambda _thread: AgentSession(first_agent))
    thread = await first_service.open_thread("agent", thread_id="legacy-restore")
    try:
        await first_service.prompt(
            thread.thread_id,
            "initial prompt",
            request_id="legacy-request",
            account_name="work",
        )
        first_result = await first_service.wait(thread.thread_id, "legacy-request")
        assert first_result.status == "completed"
        checkpoint_store = first_agent._owned_threads[
            thread.thread_id
        ].resources.checkpoint_store
        namespace = first_agent.get_module_name()
        checkpoint = dict(
            checkpoint_store.load_state(
                namespace, thread.thread_id, first_result.run_id
            )
        )
        checkpoint["status"] = "paused"
        checkpoint["scope"] = dict(checkpoint["scope"])
        checkpoint["scope"].pop("account_name")
        messages = dict(checkpoint["messages"])
        messages["items"] = messages["items"][:-1]
        checkpoint["messages"] = messages
        checkpoint_store.save_state(
            namespace, thread.thread_id, first_result.run_id, checkpoint
        )
        first_store._connection.execute(
            "UPDATE service_admissions SET status='paused', account_name=NULL WHERE thread_id=? AND request_id=?",
            (thread.thread_id, "legacy-request"),
        )
        first_store._connection.commit()
    finally:
        await first_service.aclose()
        first_store.close()

    restarted_store = SQLiteServiceStore(service_path)
    restarted_service = AgentService(store=restarted_store, default_account="default")
    restarted_model = _model(accounts, client)
    restarted_agent = Agent(
        name="legacy-restore", model=restarted_model, agent_dir=agent_dir
    )
    restarted_service.register("agent", lambda _thread: AgentSession(restarted_agent))
    try:
        await restarted_service.resume(
            thread.thread_id, "legacy-request", worker_stopped=True
        )
        result = await restarted_service.wait(thread.thread_id, "legacy-request")
        assert result.status == "failed"
        assert "no recorded account selection" in result.error
        assert captured == ["Bearer fake-work-key"]
    finally:
        await restarted_service.aclose()
        restarted_store.close()
        await client.aclose()


@pytest.mark.asyncio
async def test_background_subagent_inherits_admitted_account(tmp_path):
    accounts = Accounts(tmp_path / "accounts")
    accounts.set_api_key("openai", "fake-work-key", "work")
    captured = []
    root_tool_returned = False

    async def handler(request):
        nonlocal root_tool_returned
        captured.append(request.headers["authorization"])
        body = json.loads(request.content)
        has_agent_tool = any(
            item.get("function", {}).get("name") == "agent"
            for item in body.get("tools", [])
        )
        if has_agent_tool and not root_tool_returned:
            root_tool_returned = True
            return httpx2.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "delegate-call",
                                        "type": "function",
                                        "function": {
                                            "name": "agent",
                                            "arguments": json.dumps(
                                                {
                                                    "name": "child",
                                                    "message": "run child",
                                                    "run_in_background": True,
                                                }
                                            ),
                                        },
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ]
                },
            )
        return httpx2.Response(200, json=_chat_response())

    client = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    shared_model = _model(accounts, client)
    root = Agent(name="root", model=shared_model, agent_dir=tmp_path / "root-agent")
    child = Agent(name="child", model=shared_model)
    root.tool_library.add(mf.tool_config(allow_background=True)(AgentTool()))
    root.tool_library.add(child)
    store = SQLiteServiceStore()
    service = AgentService(store=store)
    service.register("agent", lambda _thread: AgentSession(root))
    thread = await service.open_thread("agent", thread_id="background-account")
    try:
        receipt = await service.prompt(
            thread.thread_id,
            "delegate",
            request_id="background-account-run",
            account_name="work",
        )
        settled = await asyncio.wait_for(
            service.wait(thread.thread_id, "background-account-run"), timeout=5
        )
        assert settled.status == "completed"
        for _ in range(100):
            owned = root._owned_threads.get(thread.thread_id)
            tasks = owned.resources.task_store.list() if owned is not None else ()
            if tasks and all(task.status in {"completed", "failed"} for task in tasks):
                break
            await asyncio.sleep(0.02)
        assert tasks
        resources = root._owned_threads[thread.thread_id].resources
        tasks = resources.task_store.list()
        assert tasks
        assert tasks[0].metadata["account_name"] == "work"
        await asyncio.sleep(0.05)
        assert captured
        assert set(captured) == {"Bearer fake-work-key"}
    finally:
        await service.aclose()
        await client.aclose()
        store.close()
