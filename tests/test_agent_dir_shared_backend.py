"""Automatic process-wide backend selection for Agent(agent_dir=...)."""

from unittest.mock import Mock

import pytest

from msgflux.nn import Agent
from msgflux.runtime.agent_backend import AgentDirectoryResources
from msgflux.runtime.agent_resources import AgentResources
from msgflux.runtime.resources import RuntimeResources


def _agent(name, agent_dir):
    model = Mock()
    model.model_type = "chat_completion"
    return Agent(name=name, model=model, agent_dir=agent_dir)


def test_agents_with_same_agent_dir_share_one_pool_across_threads(tmp_path):
    root = tmp_path / "agents"
    first = _agent("first", root)
    second = _agent("second", root)

    bindings = [first._bind_resources(f"first-{index}") for index in range(12)] + [
        second._bind_resources(f"second-{index}") for index in range(12)
    ]
    backend = first._resources.backend

    assert isinstance(first._resources, AgentDirectoryResources)
    assert backend is second._resources.backend
    assert backend.connection_count == 4
    assert backend.active_bindings == 24
    assert len({id(binding.checkpoint_store) for binding in bindings}) == 1

    for binding in bindings:
        binding.close()
    assert backend.connection_count == 4
    assert backend.active_bindings == 0

    import asyncio

    asyncio.run(first.aclose())
    assert backend.connection_count == 4
    asyncio.run(second.aclose())
    assert backend.connection_count == 0


@pytest.mark.asyncio
async def test_shared_pool_closes_only_after_last_agent_releases(tmp_path):
    root = tmp_path / "agents"
    first = _agent("first", root)
    second = _agent("second", root)
    first_binding = first._bind_resources("thread-first")
    second_binding = second._bind_resources("thread-second")
    backend = first._resources.backend

    first_binding.close()
    second_binding.close()
    await first.aclose()
    assert backend.connection_count == 4
    assert second._resources.backend is backend

    await second.aclose()
    assert backend.connection_count == 0


@pytest.mark.asyncio
async def test_existing_legacy_agent_dir_keeps_per_thread_stores(tmp_path):
    root = tmp_path / "legacy"
    legacy = AgentResources(root).bind("old-thread", namespace="legacy")
    legacy.close()
    agent = _agent("assistant", root)

    binding = agent._bind_resources("new-thread")
    assert agent._resources._mode == "legacy"
    assert agent._resources.backend is None
    assert binding.checkpoint_store.path.endswith(
        "threads/new-thread/checkpoints.sqlite3"
    )
    assert not (root / "runtime" / "backend" / "layout.json").exists()

    await agent.aclose()


def test_runtime_checkpoint_helper_can_open_shared_backend_database(tmp_path):
    root = tmp_path / "shared"
    agent = _agent("assistant", root)
    binding = agent._bind_resources("thread-a")
    backend = agent._resources.backend

    store = RuntimeResources(root).checkpoint_store("thread-a")
    assert store.routing_id == binding.checkpoint_store.routing_id
    store.close()
    assert backend.connection_count == 4
    assert binding.checkpoint_store.list_runs("assistant", "thread-a") == []

    binding.close()
    import asyncio

    asyncio.run(agent.aclose())
