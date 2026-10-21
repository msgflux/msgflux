import pytest

from msgflux.accounts.providers import (
    API_KEY_PROVIDERS,
    OAUTH_PROVIDERS,
    ProviderAuthSpec,
    get_api_key_env,
    get_provider_auth,
)
import msgspec


@pytest.mark.parametrize(
    ("provider", "env_name"),
    [("openai", "OPENAI_API_KEY"), ("groq", "GROQ_API_KEY")],
)
def test_api_key_provider_specs_follow_provider_metadata(provider, env_name):
    assert API_KEY_PROVIDERS[provider].env_name == env_name
    assert get_api_key_env(provider) == env_name


def test_codex_is_explicit_oauth_provider():
    assert OAUTH_PROVIDERS["openai-codex"].method == "oauth"
    assert get_provider_auth("OPENAI-CODEX").provider == "openai-codex"


def test_unsupported_oauth_provider_is_not_invented():
    with pytest.raises(ValueError, match="Unsupported account provider"):
        get_provider_auth("anthropic-claude-code")


def test_oauth_provider_cannot_be_used_as_api_key():
    with pytest.raises(ValueError, match="does not use an API key"):
        get_api_key_env("openai-codex")


def test_provider_auth_spec_is_msgspec_struct():
    assert isinstance(get_provider_auth("openai"), ProviderAuthSpec)
    assert issubclass(ProviderAuthSpec, msgspec.Struct)


def test_provider_registered_after_first_lookup_is_discovered(monkeypatch):
    from msgflux.models.provider_env import ProviderEnvBase
    from msgflux.models.registry import model_registry, register_model

    get_provider_auth("openai")

    class CustomProvider(ProviderEnvBase):
        model_type = "account_discovery_test"
        provider = "custom-account-provider"
        api_key_env = "CUSTOM_ACCOUNT_API_KEY"

    monkeypatch.setitem(model_registry, CustomProvider.model_type, {})
    register_model(CustomProvider)

    spec = get_provider_auth(CustomProvider.provider)
    assert spec.env_name == CustomProvider.api_key_env
    assert API_KEY_PROVIDERS[CustomProvider.provider] == spec
