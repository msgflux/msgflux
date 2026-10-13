from msgflux.models.model_credentials import (
    ModelCredentialResolver,
    ResolvedModelCredentials,
)


class _InjectedResolver(ModelCredentialResolver):
    def resolve(self, owner):
        return ResolvedModelCredentials(headers={"Authorization": "Bearer injected"})


def test_custom_resolver_skips_initial_api_key_environment_check(monkeypatch):
    from msgflux.models.providers.groq import GroqChatCompletion

    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    resolver = _InjectedResolver()
    model = GroqChatCompletion(
        model_id="openai/gpt-oss-20b", credential_resolver=resolver
    )

    assert model.credential_resolver is resolver


def test_codex_preserves_injected_resolver(monkeypatch, tmp_path):
    from msgflux.models.providers.openai_codex import (
        OpenAICodexChatCompletion,
        CodexCredentialResolver,
    )

    resolver = _InjectedResolver()
    model = OpenAICodexChatCompletion(
        "gpt-5.6",
        auth_file=tmp_path / "missing-auth.json",
        credential_resolver=resolver,
    )

    assert model.credential_resolver is resolver
    default_model = OpenAICodexChatCompletion(
        "gpt-5.6", auth_file=tmp_path / "missing-auth.json"
    )
    assert isinstance(default_model.credential_resolver, CodexCredentialResolver)


def test_response_cache_is_partitioned_by_execution_account(monkeypatch):
    from msgflux.models.providers.groq import GroqChatCompletion
    from msgflux.runtime.context import ExecutionScope, execution_context

    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    model = GroqChatCompletion(model_id="openai/gpt-oss-20b", enable_cache=True)
    kwargs = {"messages": [{"role": "user", "content": "same prompt"}]}
    with execution_context(scope=ExecutionScope(account_name="work")):
        model._store_cache("work response", **kwargs)
    with execution_context(scope=ExecutionScope(account_name="personal")):
        assert model._check_cache(**kwargs) is None
    with execution_context(scope=ExecutionScope(account_name="work")):
        assert model._check_cache(**kwargs) == "work response"
