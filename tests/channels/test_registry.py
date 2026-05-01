import textwrap
from types import SimpleNamespace

import pytest

from msgflux.channels import ChannelRegistry
from msgflux.channels.exceptions import AgentNotFoundError, RateLimitExceededError
from msgflux.channels.registry import load_registry_target


class NamedAgent:
    name = "named_agent"


class ClassAgent:
    name = "class_agent"


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


def test_channel_registry_instantiates_agent_class():
    registry = ChannelRegistry()

    registry.agent(ClassAgent)

    agent = registry.get_agent("class_agent")
    assert isinstance(agent, ClassAgent)
    assert agent is not ClassAgent


def test_channel_registry_instantiates_agent_class_with_explicit_name():
    registry = ChannelRegistry()

    registry.agent(ClassAgent, name="support")

    agent = registry.get_agent("support")
    assert isinstance(agent, ClassAgent)


def test_channel_registry_decorator_registers_agent_class_and_returns_class():
    registry = ChannelRegistry()

    @registry.agent(name="support")
    class SupportAgent:
        pass

    agent = registry.get_agent("support")
    assert isinstance(agent, SupportAgent)
    assert SupportAgent.__name__ == "SupportAgent"


def test_channel_registry_decorator_uses_class_name_when_no_agent_name():
    registry = ChannelRegistry()

    @registry.agent
    class SupportAgent:
        pass

    assert isinstance(registry.get_agent("SupportAgent"), SupportAgent)


def test_channel_registry_rejects_class_that_requires_constructor_args():
    registry = ChannelRegistry()

    class RequiredArgsAgent:
        def __init__(self, model):
            self.model = model

    with pytest.raises(TypeError, match="instantiable without arguments"):
        registry.agent(RequiredArgsAgent)


def test_channel_registry_missing_agent_raises_channel_error():
    registry = ChannelRegistry()

    with pytest.raises(AgentNotFoundError, match="Agent `missing` is not registered"):
        registry.get_agent("missing")


def test_channel_registry_settings_are_global_and_validated():
    registry = ChannelRegistry()

    settings = registry.settings(
        max_request_bytes=1024,
        request_timeout_s=3,
        enable_docs=False,
        cors=True,
        allowed_origins=["https://app.example.com"],
    )

    assert settings is registry.settings()
    assert settings.max_request_bytes == 1024
    assert settings.request_timeout_s == 3
    assert settings.enable_docs is False
    assert settings.cors is True
    assert settings.allowed_origins == ["https://app.example.com"]

    with pytest.raises(TypeError, match="Unknown channel setting"):
        registry.settings(unknown=True)


def test_channel_registry_defaults_are_global_and_per_agent():
    registry = ChannelRegistry()

    global_defaults = registry.defaults(
        vars={"tenant": "default"},
        model_preference="fast",
        tool_filter={"block": "*"},
        reasoning_policy={"effort": "low"},
    )
    support_defaults = registry.defaults(
        "support",
        vars={"tenant": "support"},
        tool_filter={"allow": ["search"]},
    )

    merged = registry.run_defaults("support")

    assert registry.defaults() is global_defaults
    assert registry.defaults("support") is support_defaults
    assert merged.vars == {"tenant": "support"}
    assert merged.model_preference == "fast"
    assert merged.tool_filter == {"allow": ["search"]}
    assert merged.reasoning_policy == {"effort": "low"}

    with pytest.raises(TypeError, match="Unknown agent default"):
        registry.defaults(unknown=True)


def test_channel_registry_rate_limit_validates_policy():
    registry = ChannelRegistry()

    with pytest.raises(ValueError, match="requests"):
        registry.rate_limit(requests=0)

    with pytest.raises(ValueError, match="window_s"):
        registry.rate_limit(requests=1, window_s=0)

    with pytest.raises(ValueError, match="api_key"):
        registry.rate_limit(requests=1, by="unknown")


@pytest.mark.asyncio
async def test_channel_registry_rate_limit_by_tenant():
    registry = ChannelRegistry()
    registry.rate_limit(requests=1, window_s=60, by="tenant")
    request = SimpleNamespace(run_config={"vars": {"tenant": "acme"}})
    context = SimpleNamespace(state={})

    await registry.check_rate_limits(request, context)

    with pytest.raises(RateLimitExceededError):
        await registry.check_rate_limits(request, context)


def test_channel_registry_registers_auth_authorizer_and_hooks():
    registry = ChannelRegistry()
    calls = []

    @registry.auth
    def auth():
        calls.append("auth")
        return {"tenant": "acme"}

    @registry.authorize(agent="support")
    def authorize():
        calls.append("authorize")

    @registry.error_handler(ValueError)
    def handle_error():
        calls.append("error")

    @registry.startup
    def startup():
        calls.append("startup")

    @registry.shutdown
    def shutdown():
        calls.append("shutdown")

    @registry.on_request_start
    def request_start():
        calls.append("request_start")

    assert registry.auth_handler() is auth
    assert registry.authorizers("support") == [authorize]
    assert registry.error_handlers(ValueError("bad")) == [(ValueError, handle_error)]
    assert registry.has_lifespan_hooks() is True


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
