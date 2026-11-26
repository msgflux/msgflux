"""A service reconstructs stopped compositions instead of filtering live hooks."""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from msgflux.data.stores import InMemoryCheckpointStore
from msgflux.models.response import ModelResponse
from msgflux.nn import Agent
from msgflux.nn.extensions import AgentExtension
from msgflux.nn.extensions.discovery import AgentExtensionCatalog, ExtensionConfig
from msgflux.nn.hooks import Hook
from msgflux.runtime.service import (
    AgentService,
    AgentSession,
    SQLiteServiceStore,
    ServiceBusyError,
    ServiceConflictError,
)
from msgflux.runtime.service.records import ExtensionSelection
from msgflux.runtime import ExecutionScope


class Marker(AgentExtension):
    def __init__(self):
        super().__init__("marker")

    def hooks(self):
        from dataclasses import replace

        return (
            Hook(
                event="transform_system_prompt",
                handler=lambda ctx: replace(
                    ctx, system_prompt=ctx.system_prompt + " EXTENSION_ENABLED"
                ),
            ),
        )

    def tools(self):
        def extension_probe() -> str:
            """Return the marker provided by the host extension."""
            return "marker"

        return (extension_probe,)


class Branch(AgentExtension):
    def __init__(self):
        super().__init__("branch_extension")

    def tools(self):
        return (Agent(name="branch", model=Mock(model_type="chat_completion")),)


class Broken(AgentExtension):
    def __init__(self):
        raise RuntimeError("deliberate constructor failure")


def _text(value="done"):
    response = ModelResponse()
    response.set_response_type("text_generation")
    response.add(value)
    return response


def _catalog():
    return AgentExtensionCatalog(
        {
            "marker": ExtensionConfig(
                factory="tests.test_service_extension_controls:Marker"
            ),
            "branch_extension": ExtensionConfig(
                factory="tests.test_service_extension_controls:Branch"
            ),
            "broken": ExtensionConfig(
                factory="tests.test_service_extension_controls:Broken"
            ),
        }
    )


def _setup(path=":memory:", *, store=None, answer=None, selected=False):
    store = store or SQLiteServiceStore(path)
    built = []
    checkpoints = InMemoryCheckpointStore()

    def factory(_thread):
        child = Agent(name="builder", model=Mock(model_type="chat_completion"))
        child.generator.aforward = AsyncMock(return_value=_text())
        agent = Agent(
            name="main",
            model=Mock(model_type="chat_completion"),
            tools=[child],
            checkpoint_store=checkpoints,
        )
        agent.generator.aforward = (
            AsyncMock(side_effect=answer) if answer else AsyncMock(return_value=_text())
        )
        if selected:
            agent.register_extension("marker", Marker())
        built.append(agent)

        async def close():
            await agent.aclose()
            await child.aclose()

        return AgentSession(
            agent,
            checkpoint_store=checkpoints,
            extension_catalog=_catalog(),
            on_close=close,
        )

    service = AgentService(store=store)
    service.register("main", factory)
    return service, built


@pytest.mark.asyncio
async def test_tree_then_targeted_updates_reconstruct_preserve_history_and_restore(
    tmp_path,
):
    service, built = _setup(tmp_path / "service.db")
    try:
        await service.open_thread("main", thread_id="t")
        initial = await service.thread_extensions("t")
        assert initial.revision == 0
        assert [(a.agent_id, a.name, a.parent_id) for a in initial.agents] == [
            ("root", "main", None),
            ("root/builder", "builder", "root"),
        ]
        assert {"broken", "marker"} <= set(initial.available)
        assert all("marker" not in a.extensions for a in initial.agents)
        original = built[-1]
        enabled = await service.update_extensions(
            "t", scope="tree", enable=["marker"], expected_revision=0
        )
        assert enabled.revision == 1
        assert built[-1] is not original
        assert all("marker" in a.extensions for a in enabled.agents)
        assert "extension_probe" in built[-1].tool_library.get_tool_names()
        receipt = await service.prompt("t", "hello", request_id="one")
        assert (await service.wait("t", "one")).status == "completed"
        assert (
            "EXTENSION_ENABLED"
            in built[-1].generator.aforward.call_args.kwargs["system_prompt"]
        )
        changed = await service.update_extensions(
            "t", agent_ids=["root"], disable=["marker"], expected_revision=1
        )
        assert changed.revision == 2
        assert ["marker" in a.extensions for a in changed.agents] == [False, True]
        assert "extension_probe" not in built[-1].tool_library.get_tool_names()
        assert (await service.run_result("t", receipt.run_id)).output == "done"
        assert (
            await service.update_extensions(
                "t", agent_ids=["root"], disable=["marker"], expected_revision=2
            )
        ).revision == 2
        await service.release_session("t")
        restored = await service.thread_extensions("t")
        assert restored == changed
    finally:
        await service.aclose()
        service.store.close()


@pytest.mark.asyncio
async def test_revision_invalid_requests_and_failed_factory_leave_selection_intact():
    service, _built = _setup()
    try:
        await service.open_thread("main", thread_id="t")
        await service.update_extensions("t", scope="tree", enable=["marker"])
        before = await service.thread_extensions("t")
        with pytest.raises(ServiceConflictError):
            await service.update_extensions(
                "t", scope="tree", disable=["marker"], expected_revision=0
            )
        invalid = [
            {"scope": "tree", "agent_ids": ["root"], "enable": ["marker"]},
            {"agent_ids": ["missing"], "enable": ["marker"]},
            {"scope": "tree", "enable": ["unknown"]},
            {"scope": "tree", "enable": ["marker"], "disable": ["marker"]},
            {"scope": "tree", "enable": []},
            {"scope": "tree", "enable": ["marker", "marker"]},
        ]
        for kwargs in invalid:
            with pytest.raises(ValueError):
                await service.update_extensions("t", **kwargs)
            assert await service.thread_extensions("t") == before
        with pytest.raises(ValueError, match="deliberate constructor failure"):
            await service.update_extensions("t", scope="tree", enable=["broken"])
        assert service.store.extension_selection("t").revision == 1
        assert await service.thread_extensions("t") == before
    finally:
        await service.aclose()
        service.store.close()


@pytest.mark.asyncio
async def test_running_generation_and_retained_lease_block_reconstruction():
    entered, release = asyncio.Event(), asyncio.Event()

    async def answer(**_kwargs):
        entered.set()
        await release.wait()
        return _text()

    service, built = _setup(answer=answer)
    try:
        await service.open_thread("main", thread_id="t")
        await service.prompt("t", "hello", request_id="one")
        await asyncio.wait_for(entered.wait(), 2)
        with pytest.raises(ServiceBusyError):
            await service.update_extensions("t", scope="tree", enable=["marker"])
        assert len(built) == 1
        release.set()
        await service.wait("t", "one")
        lease = await service.acquire_session("t")
        try:
            with pytest.raises(ServiceBusyError, match="lease"):
                await service.update_extensions("t", scope="tree", enable=["marker"])
        finally:
            await lease.aclose()
        assert (
            await service.update_extensions("t", scope="tree", enable=["marker"])
        ).revision == 1
    finally:
        release.set()
        await service.aclose()
        service.store.close()


def test_store_cas_guards_active_admission_and_stale_factory_generation(tmp_path):
    path = tmp_path / "service.db"
    store = SQLiteServiceStore(path)
    other = SQLiteServiceStore(path)
    try:
        from msgflux.runtime.service import ServiceThread

        store.bind_thread(ServiceThread("t", "main"))
        saved = store.update_extension_selection(
            ExtensionSelection("t", agents={"root": ("marker",)}), expected_revision=0
        )
        assert other.extension_selection("t") == saved
        with pytest.raises(ServiceConflictError):
            other.admit("t", "stale", "hello", "main", extension_revision=0)
        admission = other.admit("t", "new", "hello", "main", extension_revision=1)
        with pytest.raises(ServiceBusyError):
            store.update_extension_selection(
                ExtensionSelection("t", agents={"root": ()}), expected_revision=1
            )
        assert store.extension_selection("t") == saved
        assert admission.receipt.status == "accepted"
    finally:
        other.close()
        store.close()


@pytest.mark.asyncio
async def test_reconfiguration_fences_prompt_and_cancelled_waiter_does_not_abort_update():
    service, _built = _setup()
    entered, release = asyncio.Event(), asyncio.Event()
    try:
        await service.open_thread("main", thread_id="t")
        await service.thread_extensions("t")
        factory = service._factories["main"]

        async def delayed(thread):
            entered.set()
            await release.wait()
            return factory(thread)

        service._factories["main"] = delayed
        update = asyncio.create_task(
            service.update_extensions("t", scope="tree", enable=["marker"])
        )
        await asyncio.wait_for(entered.wait(), 2)
        with pytest.raises(ServiceBusyError, match="updated"):
            await service.prompt("t", "do not admit", request_id="blocked")
        assert service.store.get("t", "blocked") is None
        update.cancel()
        with pytest.raises(asyncio.CancelledError):
            await update
        owned = service._extension_updates["t"]
        release.set()
        result = await asyncio.wait_for(asyncio.shield(owned), 2)
        assert result.revision == 1
        assert (await service.thread_extensions("t")) == result
    finally:
        release.set()
        await service.aclose()
        service.store.close()


@pytest.mark.asyncio
async def test_subagent_call_blocks_update_even_when_parent_has_no_foreground_run():
    service, built = _setup()
    entered, release = asyncio.Event(), asyncio.Event()
    try:
        await service.open_thread("main", thread_id="t")
        await service.thread_extensions("t")
        child = built[-1].tool_library.library["builder"].impl

        async def answer(**_kwargs):
            entered.set()
            await release.wait()
            return _text()

        child.generator.aforward = AsyncMock(side_effect=answer)
        child_task = asyncio.create_task(
            child.acall("child", scope=ExecutionScope(thread_id="child-thread"))
        )
        await asyncio.wait_for(entered.wait(), 2)
        with pytest.raises(ServiceBusyError, match="subagent"):
            await service.update_extensions("t", scope="tree", enable=["marker"])
        assert service.store.extension_selection("t").revision == 0
        release.set()
        await child_task
        assert (
            await service.update_extensions("t", scope="tree", enable=["marker"])
        ).revision == 1
    finally:
        release.set()
        await service.aclose()
        service.store.close()


@pytest.mark.asyncio
async def test_another_service_reloads_its_stale_session_after_update(tmp_path):
    first, _ = _setup(tmp_path / "service.db")
    second, built = _setup(tmp_path / "service.db")
    try:
        await first.open_thread("main", thread_id="t")
        assert (await second.thread_extensions("t")).revision == 0
        old = built[-1]
        changed = await first.update_extensions("t", scope="tree", enable=["marker"])
        assert await second.thread_extensions("t") == changed
        assert built[-1] is not old
        await second.prompt("t", "after update", request_id="one")
        assert (await second.wait("t", "one")).status == "completed"
        assert (
            "EXTENSION_ENABLED"
            in built[-1].generator.aforward.call_args.kwargs["system_prompt"]
        )
    finally:
        await first.aclose()
        await second.aclose()
        first.store.close()
        second.store.close()


@pytest.mark.asyncio
async def test_new_service_process_state_loads_saved_selection(tmp_path):
    path = tmp_path / "service.db"
    first, _ = _setup(path)
    await first.open_thread("main", thread_id="t")
    expected = await first.update_extensions(
        "t", agent_ids=["root/builder"], enable=["marker"]
    )
    await first.aclose()
    first.store.close()
    second, _ = _setup(path)
    try:
        assert await second.thread_extensions("t") == expected
    finally:
        await second.aclose()
        second.store.close()


@pytest.mark.asyncio
async def test_failed_reconfiguration_cleanup_quarantines_instead_of_overlapping_sessions():
    from msgflux.runtime.service import ServiceRecoveryRequiredError

    service, _ = _setup()
    try:
        await service.open_thread("main", thread_id="t")
        await service.thread_extensions("t")
        factory = service._factories["main"]

        def failing(thread):
            session = factory(thread)

            async def close():
                raise RuntimeError("cleanup failed")

            session.on_close = close
            return session

        service._factories["main"] = failing
        original = service.store.update_extension_selection

        def lose_cas(*_args, **_kwargs):
            raise ServiceConflictError("another updater won")

        service.store.update_extension_selection = lose_cas
        with pytest.raises(ServiceRecoveryRequiredError, match="quarantined"):
            await service.update_extensions("t", scope="tree", enable=["marker"])
        service.store.update_extension_selection = original
        assert service.store.extension_selection("t").revision == 0
        with pytest.raises(ServiceRecoveryRequiredError, match="quarantined"):
            await service.thread_extensions("t")
    finally:
        with pytest.raises(ExceptionGroup):
            await service.aclose()
        service.store.close()


@pytest.mark.asyncio
async def test_stale_prompt_lease_cannot_schedule_after_other_owner_updated(tmp_path):
    first, _ = _setup(tmp_path / "service.db")
    second, _ = _setup(tmp_path / "service.db")
    try:
        await first.open_thread("main", thread_id="t")
        lease = await second.acquire_session("t")
        try:
            await first.update_extensions("t", scope="tree", enable=["marker"])
            async with second._lock:
                with pytest.raises(ServiceConflictError, match="extension"):
                    second._admit_prompt_locked(lease.session, "t", "stale", "stale")
            assert second.store.get("t", "stale") is None
        finally:
            await lease.aclose()
        assert (await second.thread_extensions("t")).revision == 1
    finally:
        await first.aclose()
        await second.aclose()
        first.store.close()
        second.store.close()


@pytest.mark.asyncio
async def test_factory_declared_extension_can_be_reenabled_without_a_catalog():
    service, _ = _setup(selected=True)
    factory = service._factories["main"]

    def without_catalog(thread):
        session = factory(thread)
        session.extension_catalog = None
        return session

    service._factories["main"] = without_catalog
    try:
        await service.open_thread("main", thread_id="t")
        await service.update_extensions("t", agent_ids=["root"], disable=["marker"])
        disabled = await service.thread_extensions("t")
        assert "marker" not in disabled.agents[0].extensions
        assert "marker" in disabled.available
        enabled = await service.update_extensions(
            "t", agent_ids=["root"], enable=["marker"]
        )
        assert "marker" in enabled.agents[0].extensions
    finally:
        await service.aclose()
        service.store.close()


def test_independent_store_reviewers_only_one_revision_wins(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from msgflux.runtime.service import ServiceThread

    path = tmp_path / "store.db"
    stores = [SQLiteServiceStore(path), SQLiteServiceStore(path)]
    stores[0].bind_thread(ServiceThread("t", "main"))

    def update(index):
        try:
            return stores[index].update_extension_selection(
                ExtensionSelection("t", agents={"root": (f"extension{index}",)}),
                expected_revision=0,
            )
        except ServiceConflictError:
            return None

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(update, [0, 1]))
        assert sum(result is not None for result in results) == 1
        winner = next(result for result in results if result is not None)
        assert (
            stores[0].extension_selection("t")
            == stores[1].extension_selection("t")
            == winner
        )
    finally:
        for store in stores:
            store.close()


@pytest.mark.asyncio
async def test_coding_factory_reconstructs_nested_agents_and_retains_registered_catalog(
    tmp_path, monkeypatch
):
    from msgflux.coding.config import CodingAgentOptions, resolve_coding_agent_options
    from msgflux.coding.factory import CodingAgentCatalog, create_coding_session_factory
    from tests.test_coding_factory import _FakeModel

    monkeypatch.setattr(
        "msgflux.coding.factory._create_model", lambda _options: _FakeModel()
    )
    parent = resolve_coding_agent_options(
        CodingAgentOptions(
            agent_id="main",
            model="openai/example",
            reasoning_effort="low",
            tools_active=("read",),
            subagents=("builder",),
        )
    )
    child = resolve_coding_agent_options(
        CodingAgentOptions(
            agent_id="builder",
            model="openai/example",
            reasoning_effort="low",
            tools_active=("read",),
        )
    )
    factory = create_coding_session_factory(
        options=parent,
        catalog=CodingAgentCatalog(agents={"main": parent, "builder": child}),
        config_path=tmp_path / "config.toml",
        runtime_dir=tmp_path / "runtime",
        agent_dir=tmp_path / "state",
        extension_catalog=_catalog(),
    )
    service = AgentService(store=SQLiteServiceStore())
    service.register("main", factory)
    try:
        await service.open_thread("main", thread_id="t", cwd=tmp_path)
        initial = await service.thread_extensions("t")
        assert [a.agent_id for a in initial.agents] == ["root", "root/builder"]
        result = await service.update_extensions("t", scope="tree", enable=["marker"])
        assert all("marker" in a.extensions for a in result.agents)
        await service.release_session("t")
        assert await service.thread_extensions("t") == result
    finally:
        await service.aclose()
        service.store.close()


@pytest.mark.asyncio
async def test_abort_request_does_not_allow_update_until_producer_really_stops():
    entered, release = asyncio.Event(), asyncio.Event()

    async def answer(**_kwargs):
        entered.set()
        await release.wait()
        return _text()

    service, _ = _setup(answer=answer)
    try:
        await service.open_thread("main", thread_id="t")
        admission = await service.prompt("t", "wait", request_id="one")
        await asyncio.wait_for(entered.wait(), 2)
        assert await service.interrupt("t", admission.run_id)
        with pytest.raises(ServiceBusyError):
            await service.update_extensions("t", scope="tree", enable=["marker"])
        release.set()
        await service.wait("t", "one")
        assert (
            await service.update_extensions("t", scope="tree", enable=["marker"])
        ).revision == 1
    finally:
        release.set()
        await service.aclose()
        service.store.close()


@pytest.mark.asyncio
async def test_paused_nested_background_task_blocks_update_across_child_thread_ids():
    from msgflux.tasks import InMemoryTaskStore

    service, _ = _setup()
    tasks = InMemoryTaskStore()
    factory = service._factories["main"]

    def with_tasks(thread):
        session = factory(thread)
        session.task_store = tasks
        return session

    service._factories["main"] = with_tasks
    try:
        await service.open_thread("main", thread_id="t")
        run = await service.prompt("t", "done", request_id="one")
        await service.wait("t", "one")
        task = tasks.create(
            "grandchild",
            metadata={"thread_id": "nested-thread", "root_run_id": run.run_id},
        )
        tasks.pause(task.task_id)
        with pytest.raises(ServiceBusyError, match="descendant"):
            await service.update_extensions("t", scope="tree", enable=["marker"])
        assert service.store.extension_selection("t").revision == 0
        tasks.interrupt(task.task_id)
        assert (
            await service.update_extensions("t", scope="tree", enable=["marker"])
        ).revision == 1
    finally:
        await service.aclose()
        service.store.close()


@pytest.mark.asyncio
async def test_real_background_agent_dispatch_blocks_idle_parent_reconstruction():
    from threading import Event
    import json
    from msgflux import tool_config
    from msgflux.models.tool_call_agg import ToolCallAggregator
    from msgflux.tools.builtin import AgentTool
    from msgflux.tasks import InMemoryTaskStore

    entered, release = Event(), Event()
    tasks = InMemoryTaskStore()
    service, _ = _setup()
    factory = service._factories["main"]

    async def answer(**_kwargs):
        entered.set()
        await asyncio.to_thread(release.wait, 5)
        return _text()

    def background_factory(thread):
        session = factory(thread)
        root = session.agent
        child = root.tool_library.get_tool_definition("builder").executor.impl
        child.generator.aforward = AsyncMock(side_effect=answer)
        root.tool_library.add(tool_config(allow_background=True)(AgentTool()))
        calls = ToolCallAggregator()
        calls.process(
            0,
            "launch",
            "agent",
            json.dumps(
                {"name": "builder", "message": "work", "run_in_background": True}
            ),
        )
        response = ModelResponse()
        response.set_response_type("tool_call")
        response.add(calls)
        root.generator.aforward = AsyncMock(side_effect=[response, _text()])
        session.task_store = tasks
        return session

    service._factories["main"] = background_factory
    try:
        await service.open_thread("main", thread_id="t")
        await service.prompt("t", "delegate", request_id="one")
        assert await asyncio.to_thread(entered.wait, 3)
        assert (
            await asyncio.wait_for(service.wait("t", "one"), 3)
        ).status == "completed"
        with pytest.raises(ServiceBusyError):
            await service.update_extensions("t", scope="tree", enable=["marker"])
        release.set()
        async with asyncio.timeout(5):
            while tasks.list(status="running") or tasks.list(status="queued"):
                await asyncio.sleep(0.01)
        # Future finalizers also have to release their execution context.
        async with asyncio.timeout(5):
            while True:
                try:
                    result = await service.update_extensions(
                        "t", scope="tree", enable=["marker"]
                    )
                    break
                except ServiceBusyError:
                    await asyncio.sleep(0.01)
        assert result.revision == 1
    finally:
        release.set()
        await service.aclose()
        service.store.close()


@pytest.mark.asyncio
async def test_same_display_name_in_distinct_branches_has_distinct_update_ids():
    service, _ = _setup()
    original = service._factories["main"]

    def factory(thread):
        session = original(thread)
        for name in ("left", "right"):
            child = Agent(name="builder", model=Mock(model_type="chat_completion"))
            parent = Agent(
                name=name, model=Mock(model_type="chat_completion"), tools=[child]
            )
            session.agent.tool_library.add(parent)
        return session

    service._factories["main"] = factory
    try:
        await service.open_thread("main", thread_id="t")
        initial = await service.thread_extensions("t")
        builders = [a for a in initial.agents if a.name == "builder"]
        assert {a.agent_id for a in builders} == {
            "root/builder",
            "root/left/builder",
            "root/right/builder",
        }
        changed = await service.update_extensions(
            "t", agent_ids=["root/left/builder"], enable=["marker"]
        )
        assert [a.agent_id for a in changed.agents if "marker" in a.extensions] == [
            "root/left/builder"
        ]
        await service.release_session("t")
        assert await service.thread_extensions("t") == changed
    finally:
        await service.aclose()
        service.store.close()


def test_extension_selection_commit_survives_abrupt_process_exit(tmp_path):
    import subprocess
    import sys
    from msgflux.runtime.service import ServiceThread

    path = tmp_path / "service.db"
    store = SQLiteServiceStore(path)
    store.bind_thread(ServiceThread("t", "main"))
    store.close()
    code = """
import os, sys
from msgflux.runtime.service import SQLiteServiceStore
from msgflux.runtime.service.records import ExtensionSelection
store = SQLiteServiceStore(sys.argv[1])
store.update_extension_selection(ExtensionSelection('t', agents={'root':('marker',)}), expected_revision=0)
os._exit(73)
"""
    completed = subprocess.run(  # noqa: S603 - fixed interpreter and local test code
        [sys.executable, "-c", code, str(path)],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert completed.returncode == 73, completed.stderr
    reopened = SQLiteServiceStore(path)
    try:
        assert reopened.extension_selection("t") == ExtensionSelection(
            "t", 1, {"root": ("marker",)}
        )
    finally:
        reopened.close()


@pytest.mark.asyncio
async def test_cached_acquisition_checks_revision_without_decoding_selections(
    monkeypatch,
):
    service, _ = _setup()
    try:
        await service.open_thread("main", thread_id="t")
        await service.update_extensions("t", scope="tree", enable=["marker"])

        def should_not_decode(_thread):
            raise AssertionError("Cached acquisition must not decode the Agent tree")

        monkeypatch.setattr(service.store, "extension_selection", should_not_decode)
        lease = await service.acquire_session("t")
        await lease.aclose()
    finally:
        await service.aclose()
        service.store.close()


@pytest.mark.asyncio
async def test_failed_revision_lookup_releases_session_lease(monkeypatch):
    service, _ = _setup()
    try:
        await service.open_thread("main", thread_id="t")
        await service.thread_extensions("t")
        original = service.store.extension_revision

        def fail(_thread):
            raise OSError("journal unavailable")

        monkeypatch.setattr(service.store, "extension_revision", fail)
        with pytest.raises(OSError, match="journal unavailable"):
            await service.acquire_session("t")
        monkeypatch.setattr(service.store, "extension_revision", original)
        assert await service.release_session("t")
    finally:
        await service.aclose()
        service.store.close()


@pytest.mark.asyncio
async def test_reconstruction_cannot_silently_change_history_namespace():
    service, _ = _setup()
    original = service._factories["main"]
    try:
        await service.open_thread("main", thread_id="t")
        before = await service.thread_extensions("t")

        def wrong_namespace(_thread):
            agent = Agent(name="different", model=Mock(model_type="chat_completion"))
            return AgentSession(
                agent, extension_catalog=_catalog(), on_close=agent.aclose
            )

        service._factories["main"] = wrong_namespace
        with pytest.raises(ServiceConflictError, match="namespace"):
            await service.update_extensions("t", agent_ids=["root"], enable=["marker"])
        service._factories["main"] = original
        assert await service.thread_extensions("t") == before
    finally:
        await service.aclose()
        service.store.close()


@pytest.mark.asyncio
async def test_tree_changes_can_add_and_remove_extension_contributed_subagents():
    service, _ = _setup()
    try:
        await service.open_thread("main", thread_id="t")
        enabled = await service.update_extensions(
            "t", scope="tree", enable=["branch_extension"]
        )
        assert {a.agent_id for a in enabled.agents} == {
            "root",
            "root/builder",
            "root/branch",
            "root/builder/branch",
        }
        await service.release_session("t")
        assert await service.thread_extensions("t") == enabled
        disabled = await service.update_extensions(
            "t", scope="tree", disable=["branch_extension"]
        )
        assert [a.agent_id for a in disabled.agents] == ["root", "root/builder"]
        await service.release_session("t")
        assert await service.thread_extensions("t") == disabled
    finally:
        await service.aclose()
        service.store.close()
