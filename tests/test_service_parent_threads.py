"""Direct parent/child thread bindings stored as service metadata."""

import sqlite3
from unittest.mock import AsyncMock, Mock

import pytest

from msgflux.models.response import ModelResponse
from msgflux.nn import Agent
from msgflux.runtime.service import AgentService, AgentSession, SQLiteServiceStore
from msgflux.runtime.service.records import ServiceConflictError, ServiceThread

_BINDING_ERRORS = (ServiceConflictError, ValueError, KeyError)


def _host(tmp_path):
    """Open a store whose factory fails if any test hydrates a session."""
    store = SQLiteServiceStore(tmp_path / "service.sqlite3")
    service = AgentService(store=store, default_agent="main")

    def factory(thread):
        raise AssertionError(f"factory must not run for {thread.thread_id!r}")

    service.register("main", factory)
    return service, store


def _children(owner, parent_id):
    return tuple(child.thread for child in owner.child_threads(parent_id).children)


def _ids(threads):
    return sorted(thread.thread_id for thread in threads)


@pytest.mark.asyncio
async def test_open_thread_preserves_parent_binding_without_hydrating(tmp_path):
    service, store = _host(tmp_path)
    try:
        parent = await service.open_thread("main", thread_id="parent")
        child = await service.open_thread(
            "main", thread_id="child", parent_thread_id="parent"
        )
        assert (parent.parent_thread_id, child.parent_thread_id) == (None, "parent")
        assert store.thread("child").parent_thread_id == "parent"
        assert (await service.open_thread("main", thread_id="child")) == child
        assert set(service.threads()) == {parent, child}
    finally:
        store.close()


@pytest.mark.asyncio
async def test_conflicting_parent_binding_is_rejected_and_immutable(tmp_path):
    service, store = _host(tmp_path)
    try:
        for name, parent in (("parent", None), ("other", None), ("child", "parent")):
            await service.open_thread("main", thread_id=name, parent_thread_id=parent)
        with pytest.raises(AttributeError):
            store.thread("child").parent_thread_id = "other"
        with pytest.raises(_BINDING_ERRORS):
            await service.open_thread(
                "main", thread_id="child", parent_thread_id="other"
            )
        with pytest.raises(_BINDING_ERRORS):
            store.bind_thread(ServiceThread("child", "main", parent_thread_id="other"))
        assert store.thread("child").parent_thread_id == "parent"
    finally:
        store.close()


@pytest.mark.asyncio
async def test_parent_must_exist_and_cannot_be_the_thread_itself(tmp_path):
    service, store = _host(tmp_path)
    try:
        for thread_id, parent in (("orphan", "missing"), ("loop", "loop")):
            with pytest.raises(_BINDING_ERRORS):
                await service.open_thread(
                    "main", thread_id=thread_id, parent_thread_id=parent
                )
        assert store.threads() == ()
    finally:
        store.close()


@pytest.mark.asyncio
async def test_listing_returns_only_direct_children(tmp_path):
    service, store = _host(tmp_path)
    try:
        bindings = (
            ("root", None),
            ("child-b", "root"),
            ("child-a", "root"),
            ("grandchild", "child-a"),
        )
        for thread_id, parent in bindings:
            await service.open_thread(
                "main", thread_id=thread_id, parent_thread_id=parent
            )
        children = _children(service, "root")
        assert [type(item) for item in children] == [ServiceThread, ServiceThread]
        assert _ids(children) == ["child-a", "child-b"]
        assert _ids(_children(service, "child-a")) == ["grandchild"]
        assert _children(service, "grandchild") == ()
    finally:
        store.close()


@pytest.mark.asyncio
async def test_parent_links_survive_reopening_from_another_store(tmp_path):
    service, store = _host(tmp_path)
    await service.open_thread("main", thread_id="root")
    await service.open_thread("main", thread_id="child", parent_thread_id="root")
    store.close()
    service, reopened = _host(tmp_path)
    try:
        assert reopened.thread("child").parent_thread_id == "root"
        assert _ids(_children(service, "root")) == ["child"]
        binding = await service.open_thread("main", thread_id="child")
        assert binding.parent_thread_id == "root"
    finally:
        reopened.close()


@pytest.mark.asyncio
async def test_legacy_sqlite_thread_without_parent_column_still_opens(tmp_path):
    connection = sqlite3.connect(tmp_path / "service.sqlite3")
    connection.execute(
        "CREATE TABLE service_threads (thread_id TEXT PRIMARY KEY, "
        "agent_id TEXT NOT NULL, cwd TEXT, interactive INTEGER NOT NULL DEFAULT 1)"
    )
    connection.execute(
        "INSERT INTO service_threads (thread_id, agent_id, interactive) "
        "VALUES ('legacy', 'main', 1)"
    )
    connection.commit()
    connection.close()
    service, store = _host(tmp_path)
    try:
        stored = store.thread("legacy")
        assert (stored.agent_id, stored.cwd, stored.interactive) == ("main", None, True)
        assert (stored.parent_thread_id, _children(service, "legacy")) == (None, ())
        assert (await service.open_thread("main", thread_id="legacy")) == stored
    finally:
        store.close()


@pytest.mark.asyncio
async def test_child_binding_reaches_factory_without_a_task_store(tmp_path):
    store = SQLiteServiceStore(tmp_path / "service.sqlite3")
    seen = []

    def factory(thread):
        seen.append(thread.parent_thread_id)
        model = Mock()
        model.model_type = "chat_completion"
        agent = Agent(name="main", model=model)
        response = ModelResponse()
        response.set_response_type("text_generation")
        response.add("done")
        agent.generator.aforward = AsyncMock(return_value=response)
        return AgentSession(agent)

    service = AgentService(store=store, default_agent="main")
    service.register("main", factory)
    try:
        await service.open_thread("main", thread_id="root")
        await service.open_thread("main", thread_id="child", parent_thread_id="root")
        await service.prompt("child", "hello", request_id="first")
        receipt = await service.wait("child", "first")
        assert receipt.status == "completed"
        assert seen == ["root"]
        assert _ids(_children(service, "root")) == ["child"]
    finally:
        await service.aclose()
        store.close()
