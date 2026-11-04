"""Response-cache partitioning for deployment-pinned account overrides."""

import pytest

from msgflux.accounts import Accounts
from msgflux.models.model_credentials import (
    BearerTokenCredentialResolver,
    ModelCredentialResolver,
)
from msgflux.models.providers.groq import GroqChatCompletion
from msgflux.runtime.context import ExecutionScope, execution_context

KWARGS = {"messages": [{"role": "user", "content": "same prompt"}]}


def _model(monkeypatch, resolver):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    return GroqChatCompletion(
        model_id="openai/gpt-oss-20b",
        enable_cache=True,
        credential_resolver=resolver,
    )


def _scope(account_name):
    return ExecutionScope(account_name=account_name)


def test_default_resolver_cache_identity_is_none():
    assert ModelCredentialResolver().cache_identity(object()) is None
    assert BearerTokenCredentialResolver().cache_identity(object()) is None


def test_account_resolver_identity_is_safe_and_reads_no_credentials(tmp_path):
    accounts = Accounts(tmp_path / "accounts")
    accounts.set_api_key("groq", "personal-fake-key", "personal")
    resolver = accounts.resolver(
        "groq", account_name="personal", allow_account_override=True
    )
    identity = resolver.cache_identity(type("Owner", (), {"provider": "groq"})())
    assert identity == {
        "provider": "groq",
        "account_name": "personal",
        "accounts_root": str(accounts.root),
    }
    assert "personal-fake-key" not in str(identity)
    # Identity for a not-yet-configured account must not read credentials.
    missing = accounts.resolver(
        "groq", account_name="missing", allow_account_override=True
    )
    assert (
        missing.cache_identity(type("Owner", (), {"provider": "groq"})())[
            "account_name"
        ]
        == "missing"
    )


def test_pinned_accounts_miss_on_shared_cache_same_request(monkeypatch, tmp_path):
    accounts = Accounts(tmp_path / "accounts")
    accounts.set_api_key("groq", "work-fake-key", "work")
    accounts.set_api_key("groq", "personal-fake-key", "personal")
    model = _model(
        monkeypatch,
        accounts.resolver("groq", account_name="personal", allow_account_override=True),
    )
    with execution_context(scope=_scope("work")):
        model._store_cache("personal response", **KWARGS)
    # Same execution scope and request, but pinned to work: must miss.
    model.credential_resolver = accounts.resolver(
        "groq", account_name="work", allow_account_override=True
    )
    with execution_context(scope=_scope("work")):
        assert model._check_cache(**KWARGS) is None
    # Same effective pinned account hits.
    model.credential_resolver = accounts.resolver(
        "groq", account_name="personal", allow_account_override=True
    )
    with execution_context(scope=_scope("work")):
        assert model._check_cache(**KWARGS) == "personal response"


def test_same_account_name_different_roots_miss(monkeypatch, tmp_path):
    first = Accounts(tmp_path / "first")
    first.set_api_key("groq", "first-fake-key", "personal")
    second = Accounts(tmp_path / "second")
    second.set_api_key("groq", "second-fake-key", "personal")
    model = _model(
        monkeypatch,
        first.resolver("groq", account_name="personal", allow_account_override=True),
    )
    with execution_context(scope=_scope("work")):
        model._store_cache("first response", **KWARGS)
        model.credential_resolver = second.resolver(
            "groq", account_name="personal", allow_account_override=True
        )
        assert model._check_cache(**KWARGS) is None
        # A fresh Accounts object over the same root hits.
        model.credential_resolver = Accounts(tmp_path / "first").resolver(
            "groq", account_name="personal", allow_account_override=True
        )
        assert model._check_cache(**KWARGS) == "first response"


def test_inherited_account_partitioning_preserved(monkeypatch, tmp_path):
    accounts = Accounts(tmp_path / "accounts")
    accounts.set_api_key("groq", "work-fake-key", "work")
    accounts.set_api_key("groq", "personal-fake-key", "personal")
    model = _model(monkeypatch, accounts.resolver("groq"))
    with execution_context(scope=_scope("work")):
        model._store_cache("work response", **KWARGS)
    with execution_context(scope=_scope("personal")):
        assert model._check_cache(**KWARGS) is None
    with execution_context(scope=_scope("work")):
        assert model._check_cache(**KWARGS) == "work response"


def test_no_resolver_identity_falls_back_to_scope(monkeypatch):
    model = _model(monkeypatch, BearerTokenCredentialResolver())
    with execution_context(scope=_scope("work")):
        model._store_cache("work response", **KWARGS)
    with execution_context(scope=_scope("personal")):
        assert model._check_cache(**KWARGS) is None
    with execution_context(scope=_scope("work")):
        assert model._check_cache(**KWARGS) == "work response"


@pytest.mark.asyncio
async def test_sync_async_credential_policy_unchanged(tmp_path):
    accounts = Accounts(tmp_path / "accounts")
    accounts.set_api_key("groq", "personal-fake-key", "personal")
    resolver = accounts.resolver(
        "groq", account_name="personal", allow_account_override=True
    )
    owner = type("Owner", (), {"provider": "groq"})()
    with execution_context(scope=_scope("work")):
        assert resolver.resolve(owner).headers["Authorization"] == (
            "Bearer personal-fake-key"
        )
        assert (await resolver.aresolve(owner)).headers["Authorization"] == (
            "Bearer personal-fake-key"
        )
