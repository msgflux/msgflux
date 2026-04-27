import textwrap

import pytest

from msgflux.channels import ChannelRegistry
from msgflux.channels.exceptions import AgentNotFoundError
from msgflux.channels.registry import load_registry_target


class NamedAgent:
    name = "named_agent"


def test_channel_registry_registers_agent_by_name_attr():
    registry = ChannelRegistry()
    agent = NamedAgent()

    registry.agent(agent)

    assert registry.get_agent("named_agent") is agent
    assert "named_agent" in registry


def test_channel_registry_registers_agent_with_explicit_name():
    registry = ChannelRegistry()
    agent = object()

    registry.agent(agent, name="support")

    assert registry.get_agent("support") is agent


def test_channel_registry_missing_agent_raises_channel_error():
    registry = ChannelRegistry()

    with pytest.raises(AgentNotFoundError, match="Agent `missing` is not registered"):
        registry.get_agent("missing")


def test_load_registry_target_from_python_file(tmp_path):
    module_path = tmp_path / "app.py"
    module_path.write_text(
        textwrap.dedent(
            """
            from msgflux.channels import ChannelRegistry

            registry = ChannelRegistry()
            """
        )
    )

    registry = load_registry_target(f"{module_path}:registry")

    assert isinstance(registry, ChannelRegistry)
