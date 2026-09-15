"""Tests for per-instance API key environment overrides."""

import pytest


@pytest.fixture
def key_envs(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "groq-default-key")
    monkeypatch.setenv("GROQ_ACME_KEY", "groq-acme-key")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-default-key")
    monkeypatch.setenv("OPENAI_ACME_KEY", "openai-acme-key")


def test_default_env_unchanged(key_envs):
    from msgflux.models.providers.groq import GroqChatCompletion

    model = GroqChatCompletion(model_id="openai/gpt-oss-20b")

    assert model.api_key_env == "GROQ_API_KEY"
    assert model._get_api_key() == "groq-default-key"


def test_api_key_env_override(key_envs):
    from msgflux.models.providers.groq import GroqChatCompletion

    model = GroqChatCompletion(
        model_id="openai/gpt-oss-20b", api_key_env="GROQ_ACME_KEY"
    )

    assert model.api_key_env == "GROQ_ACME_KEY"
    assert model._get_api_key() == "groq-acme-key"


def test_two_instances_two_keys(key_envs):
    import msgflux as mf

    default = mf.Model.chat_completion("groq/openai/gpt-oss-20b")
    acme = mf.Model.chat_completion(
        "groq/openai/gpt-oss-20b", api_key_env="GROQ_ACME_KEY"
    )

    assert default._get_api_key() == "groq-default-key"
    assert acme._get_api_key() == "groq-acme-key"


def test_missing_custom_env_cites_its_name(key_envs, monkeypatch):
    from msgflux.models.providers.groq import GroqChatCompletion

    monkeypatch.delenv("GROQ_ACME_KEY")

    with pytest.raises(ValueError, match="GROQ_ACME_KEY"):
        GroqChatCompletion(model_id="openai/gpt-oss-20b", api_key_env="GROQ_ACME_KEY")


def test_api_key_env_rejects_non_string(key_envs):
    from msgflux.models.providers.groq import GroqChatCompletion

    with pytest.raises(TypeError, match="`api_key_env`"):
        GroqChatCompletion(model_id="openai/gpt-oss-20b", api_key_env=123)


def test_api_key_env_rejects_empty(key_envs):
    from msgflux.models.providers.groq import GroqChatCompletion

    with pytest.raises(ValueError, match="`api_key_env`"):
        GroqChatCompletion(model_id="openai/gpt-oss-20b", api_key_env="  ")


def test_secret_value_never_serialized(key_envs):
    import msgflux as mf

    model = mf.Model.chat_completion(
        "openai/gpt-4.1-mini", api_key_env="OPENAI_ACME_KEY"
    )

    assert model._get_api_key() == "openai-acme-key"
    assert "openai-acme-key" not in str(model.serialize())
