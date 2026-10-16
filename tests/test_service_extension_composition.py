"""Configured extension composition follows custom session factory ownership."""

from unittest.mock import Mock

import pytest

from msgflux.nn import Agent, AgentExtension
from msgflux.runtime.service import AgentSession
from msgflux.runtime.service.local.runner import _make_manifest_service

_instances = []
_closed = []


class CustomExtension(AgentExtension):
    def __init__(self, label="default"):
        super().__init__("implementation_name")
        self.label = label
        _instances.append(self)

    async def aon_remove(self, agent):
        _closed.append(("extension", self.label))


class FailingExtension(CustomExtension):
    def on_register(self, agent):
        raise ValueError("registration refused")


def create_session(thread):
    agent = Agent(name="custom", model=Mock(model_type="chat_completion"))
    return AgentSession(
        agent, on_close=lambda: _closed.append(("session", thread.thread_id))
    )


async def create_session_async(thread):
    return create_session(thread)


def _config(tmp_path, *, factory="create_session", failing=False):
    config = tmp_path / "config.toml"
    config.write_text(
        "[defaults]\nagent='custom'\n"
        f"[agents.custom]\nfactory='tests.test_service_extension_composition:{factory}'\n"
        "extensions=['first', 'second']\n"
        "[extensions.first]\nfactory='tests.test_service_extension_composition:CustomExtension'\n"
        "[extensions.first.options]\nlabel='first'\n"
        "[extensions.second]\n"
        f"factory='tests.test_service_extension_composition:{'FailingExtension' if failing else 'CustomExtension'}'\n"
        "[extensions.second.options]\nlabel='second'\n"
    )
    return config


@pytest.fixture(autouse=True)
def clear_observations():
    _instances.clear()
    _closed.clear()


@pytest.mark.asyncio
@pytest.mark.parametrize("factory", ["create_session", "create_session_async"])
async def test_custom_factory_extensions_lazy_isolated_and_closed(tmp_path, factory):
    service, _path, _digest = _make_manifest_service(
        _config(tmp_path, factory=factory), tmp_path, tmp_path
    )
    try:
        assert not _instances
        assert service.inspect_agent("custom").configuration.extensions == (
            "first",
            "second",
        )
        await service.open_thread("custom", thread_id="one")
        await service.open_thread("custom", thread_id="two")
        first = await service.acquire_session("one")
        second = await service.acquire_session("two")
        try:
            assert (
                first.session.agent.extensions["first"]
                is not second.session.agent.extensions["first"]
            )
            assert [extension.name for extension in _instances] == [
                "first",
                "second",
                "first",
                "second",
            ]
        finally:
            await first.aclose()
            await second.aclose()
        assert await service.release_session("one")
        assert _closed == [
            ("extension", "second"),
            ("extension", "first"),
            ("session", "one"),
        ]
        reloaded = await service.acquire_session("one")
        try:
            assert len(_instances) == 6
            assert reloaded.session.agent.extensions["first"] is _instances[4]
        finally:
            await reloaded.aclose()
    finally:
        await service.aclose()
        service.store.close()
    assert len([item for item in _closed if item[0] == "extension"]) == 6


@pytest.mark.asyncio
async def test_custom_factory_registration_failure_closes_extensions_and_session(
    tmp_path,
):
    service, _path, _digest = _make_manifest_service(
        _config(tmp_path, failing=True), tmp_path, tmp_path
    )
    try:
        await service.open_thread("custom", thread_id="one")
        with pytest.raises(ValueError, match="registration refused"):
            await service.acquire_session("one")
        assert _closed == [
            ("extension", "second"),
            ("extension", "first"),
            ("session", "one"),
        ]
        assert not service._sessions
    finally:
        await service.aclose()
        service.store.close()


def create_managed_session(thread):
    agent = Agent(
        name="managed-custom",
        model=Mock(model_type="chat_completion"),
        agent_dir=thread.cwd,
    )
    original_close = agent._close_thread_resources

    async def settle(thread_id, *, before_close=None):
        _closed.append(("settle", thread_id))
        await original_close(thread_id, before_close=before_close)

    agent._close_thread_resources = settle
    return AgentSession(
        agent, on_close=lambda: _closed.append(("session", thread.thread_id))
    )


@pytest.mark.asyncio
async def test_managed_custom_session_settles_children_before_removing_extensions(
    tmp_path,
):
    service, _, _ = _make_manifest_service(
        _config(tmp_path, factory="create_managed_session"), tmp_path, tmp_path
    )
    try:
        await service.open_thread("custom", thread_id="one", cwd=tmp_path)
        lease = await service.acquire_session("one")
        await lease.aclose()
        assert await service.release_session("one")
        assert _closed == [
            ("settle", "one"),
            ("extension", "second"),
            ("extension", "first"),
            ("session", "one"),
        ]
    finally:
        await service.aclose()
        service.store.close()
