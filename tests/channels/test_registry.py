import textwrap
from types import SimpleNamespace

import pytest

from msgflux.channels import ChannelRegistry, RateLimitDecision
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

    @registry.agent(
        name="support",
        tags=["support", "orders"],
        capabilities={"streaming": True, "tools": True},
    )
    class SupportAgent:
        """Support agent for order questions."""

        pass

    agent = registry.get_agent("support")
    metadata = registry.agent_metadata("support")
    assert isinstance(agent, SupportAgent)
    assert SupportAgent.__name__ == "SupportAgent"
    assert metadata.name == "support"
    assert metadata.description == "Support agent for order questions."
    assert metadata.tags == ["support", "orders"]
    assert metadata.capabilities == {"streaming": True, "tools": True}


def test_channel_registry_agent_metadata_uses_agent_description_attr():
    registry = ChannelRegistry()

    class SupportAgent:
        name = "support"
        description = "Description from AutoParams-style attribute."

    registry.agent(SupportAgent())

    assert registry.agent_metadata("support").description == (
        "Description from AutoParams-style attribute."
    )
    assert registry.agents_metadata()["support"].name == "support"

    with pytest.raises(AgentNotFoundError):
        registry.agent_metadata("missing")


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
        title="Support Agents",
        description="Support HTTP boundary.",
        max_request_bytes=1024,
        request_timeout_s=3,
        enable_docs=False,
        cors=True,
        allowed_origins=["https://app.example.com"],
    )

    assert settings is registry.settings()
    assert settings.title == "Support Agents"
    assert settings.description == "Support HTTP boundary."
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

    registry.rate_limit(name="api-key-minute", requests=1)

    with pytest.raises(ValueError, match="already registered"):
        registry.rate_limit(name="api-key-minute", requests=1)

    with pytest.raises(ValueError, match="must not be empty"):
        registry.rate_limit(name="", requests=1)

    with pytest.raises(TypeError, match="rate_limit_store"):
        registry.rate_limit_store(object())


@pytest.mark.asyncio
async def test_channel_registry_rate_limit_by_tenant():
    registry = ChannelRegistry()
    registry.rate_limit(requests=1, window_s=60, by="tenant")
    request = SimpleNamespace(run_config={"vars": {"tenant": "acme"}})
    context = SimpleNamespace(agent_name="support", state={})

    await registry.check_rate_limits(request, context)

    with pytest.raises(RateLimitExceededError):
        await registry.check_rate_limits(request, context)


@pytest.mark.asyncio
async def test_channel_registry_rate_limit_by_service():
    registry = ChannelRegistry()
    registry.rate_limit(requests=1, window_s=60, by="service")
    request = SimpleNamespace(run_config={})
    context = SimpleNamespace(agent_name="support", state={})

    await registry.check_rate_limits(request, context)

    with pytest.raises(RateLimitExceededError):
        await registry.check_rate_limits(request, context)


@pytest.mark.asyncio
async def test_channel_registry_rate_limit_by_client_uses_key_then_ip():
    registry = ChannelRegistry()
    registry.rate_limit(requests=1, window_s=60, by="client")
    request = SimpleNamespace(run_config={})
    context = SimpleNamespace(agent_name="support", state={})
    first_key = SimpleNamespace(
        headers={"authorization": "Bearer key-1"},
        client=SimpleNamespace(host="10.0.0.1"),
    )
    other_key = SimpleNamespace(
        headers={"authorization": "Bearer key-2"},
        client=SimpleNamespace(host="10.0.0.1"),
    )

    await registry.check_rate_limits(request, context, first_key)

    with pytest.raises(RateLimitExceededError):
        await registry.check_rate_limits(request, context, first_key)

    await registry.check_rate_limits(request, context, other_key)


@pytest.mark.asyncio
async def test_channel_registry_rate_limit_can_target_agent():
    registry = ChannelRegistry()
    registry.rate_limit(agent="support", requests=1, window_s=60, by="service")
    request = SimpleNamespace(run_config={})
    support_context = SimpleNamespace(agent_name="support", state={})
    billing_context = SimpleNamespace(agent_name="billing", state={})

    await registry.check_rate_limits(request, support_context)

    with pytest.raises(RateLimitExceededError):
        await registry.check_rate_limits(request, support_context)

    await registry.check_rate_limits(request, billing_context)


@pytest.mark.asyncio
async def test_channel_registry_uses_pluggable_rate_limit_store():
    class RecordingRateLimitStore:
        def __init__(self):
            self.calls = []

        async def hit(self, bucket, request, context, http_request):
            self.calls.append((bucket, request, context, http_request))
            return RateLimitDecision(
                allowed=len(self.calls) == 1,
                retry_after_s=7,
            )

    registry = ChannelRegistry()
    store = RecordingRateLimitStore()
    registry.rate_limit_store(store)
    policy = registry.rate_limit(
        name="support-api-key",
        requests=10,
        window_s=60,
        by="api_key",
    )
    request = SimpleNamespace(run_config={})
    context = SimpleNamespace(agent_name="support", state={})
    http_request = SimpleNamespace(
        headers={"authorization": "Bearer key-1"},
        client=SimpleNamespace(host="10.0.0.1"),
    )

    assert registry.rate_limit_store() is store
    await registry.check_rate_limits(request, context, http_request)

    with pytest.raises(RateLimitExceededError) as exc_info:
        await registry.check_rate_limits(request, context, http_request)

    first_bucket = store.calls[0][0]
    assert first_bucket.name == "support-api-key"
    assert first_bucket.identity == "key-1"
    assert first_bucket.key == "msgflux:rate_limit:support-api-key:key-1"
    assert first_bucket.policy is policy
    assert store.calls[0][1:] == (request, context, http_request)
    assert exc_info.value.headers["Retry-After"] == "7"


@pytest.mark.asyncio
async def test_channel_registry_rate_limit_store_accepts_callable_mapping_result():
    def deny(bucket):
        assert bucket.name == "policy:0"
        return {"allowed": False, "retry_after_s": 3}

    registry = ChannelRegistry()
    registry.rate_limit_store(deny)
    registry.rate_limit(requests=1, window_s=60, by="service")
    request = SimpleNamespace(run_config={})
    context = SimpleNamespace(agent_name="support", state={})

    with pytest.raises(RateLimitExceededError) as exc_info:
        await registry.check_rate_limits(request, context)

    assert exc_info.value.headers["Retry-After"] == "3"


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
