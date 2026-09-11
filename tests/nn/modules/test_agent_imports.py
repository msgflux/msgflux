import msgflux.nn.modules.agent as agent_module
from msgflux.nn.modules import Agent as ModulesAgent
from msgflux.nn.modules.agent import Agent
from msgflux.nn.modules.agent.context import _RESERVED_KWARGS
from msgflux.nn.modules.agent.core import Agent as FragmentedAgent


def test_fragmented_agent_package_preserves_public_class_identity():
    assert Agent is FragmentedAgent
    assert ModulesAgent is Agent


def test_fragmented_agent_package_preserves_reserved_kwargs_import():
    assert agent_module._RESERVED_KWARGS is _RESERVED_KWARGS
