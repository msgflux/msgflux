"""Deployment-specific account override for model credential resolution."""

import asyncio
from types import SimpleNamespace

import pytest

from msgflux.accounts import AccountError, AccountNotFoundError, Accounts
from msgflux.accounts.resolver import AccountCredentialResolver
from msgflux.runtime.context import ExecutionScope, execution_context

OWNER = SimpleNamespace(provider="openai")


def _accounts(tmp_path):
    accounts = Accounts(tmp_path / "accounts")
    accounts.set_api_key("openai", "work-fake-key", "work")
    accounts.set_api_key("openai", "personal-fake-key", "personal")
    accounts.set_api_key("openai", "default-fake-key", "default")
    return accounts


def _scope(account_name, run_id="run-1"):
    return ExecutionScope(
        thread_id="thread-1", run_id=run_id, account_name=account_name
    )


def test_default_mismatch_guard_retained(tmp_path):
    accounts = _accounts(tmp_path)
    resolver = accounts.resolver("openai", account_name="personal")
    with execution_context(scope=_scope("work")):
        with pytest.raises(AccountError, match="conflicts with the execution account"):
            resolver.resolve(OWNER)


@pytest.mark.asyncio
async def test_default_mismatch_guard_retained_async(tmp_path):
    accounts = _accounts(tmp_path)
    resolver = AccountCredentialResolver(accounts, "openai", account_name="personal")
    with execution_context(scope=_scope("work")):
        with pytest.raises(AccountError, match="conflicts"):
            await resolver.aresolve(OWNER)


def test_explicit_override_sync_uses_chosen_account(tmp_path):
    accounts = _accounts(tmp_path)
    resolver = accounts.resolver(
        "openai", account_name="personal", allow_account_override=True
    )
    scope = _scope("work")
    with execution_context(scope=scope):
        resolved = resolver.resolve(OWNER)
        assert resolved.headers["Authorization"] == "Bearer personal-fake-key"
        assert scope.account_name == "work"
    # Override also resolves when the execution has no recorded account.
    with execution_context(scope=_scope(None)):
        assert (
            resolver.resolve(OWNER).headers["Authorization"]
            == "Bearer personal-fake-key"
        )


@pytest.mark.asyncio
async def test_explicit_override_async_uses_chosen_account(tmp_path):
    accounts = _accounts(tmp_path)
    resolver = AccountCredentialResolver(
        accounts,
        "openai",
        account_name="personal",
        allow_account_override=True,
    )
    with execution_context(scope=_scope("work")):
        resolved = await resolver.aresolve(OWNER)
        assert resolved.headers["Authorization"] == "Bearer personal-fake-key"


def test_omitted_account_inherits_execution_account(tmp_path):
    accounts = _accounts(tmp_path)
    resolver = accounts.resolver("openai")
    with execution_context(scope=_scope("work")):
        assert (
            resolver.resolve(OWNER).headers["Authorization"] == "Bearer work-fake-key"
        )
    with execution_context(scope=_scope(None, run_id=None)):
        assert (
            resolver.resolve(OWNER).headers["Authorization"]
            == "Bearer default-fake-key"
        )


def test_unknown_execution_account_guard_retained_without_override(tmp_path):
    accounts = _accounts(tmp_path)
    resolver = accounts.resolver("openai")
    with execution_context(scope=_scope(None)):
        with pytest.raises(AccountError, match="no recorded account"):
            resolver.resolve(OWNER)


@pytest.mark.asyncio
async def test_concurrent_scopes_do_not_cross_contaminate(tmp_path):
    accounts = _accounts(tmp_path)
    shared = accounts.resolver("openai")
    personal = accounts.resolver(
        "openai", account_name="personal", allow_account_override=True
    )
    work = accounts.resolver("openai", account_name="work", allow_account_override=True)

    async def resolve_in(account_name, resolver):
        with execution_context(scope=_scope(account_name)):
            await asyncio.sleep(0)
            return resolver.resolve(OWNER).headers["Authorization"]

    results = await asyncio.gather(
        resolve_in("work", shared),
        resolve_in("default", shared),
        resolve_in("work", personal),
        resolve_in("default", work),
    )
    assert results == [
        "Bearer work-fake-key",
        "Bearer default-fake-key",
        "Bearer personal-fake-key",
        "Bearer work-fake-key",
    ]


def test_unknown_override_account_and_provider_fail_closed(tmp_path):
    accounts = _accounts(tmp_path)
    resolver = accounts.resolver(
        "openai", account_name="missing", allow_account_override=True
    )
    with execution_context(scope=_scope("work")):
        with pytest.raises(AccountNotFoundError):
            resolver.resolve(OWNER)
    with pytest.raises(ValueError):
        accounts.resolver(
            "no-such-provider", account_name="work", allow_account_override=True
        )


def test_no_scope_mutation_and_safe_repr(tmp_path):
    accounts = _accounts(tmp_path)
    resolver = accounts.resolver(
        "openai", account_name="personal", allow_account_override=True
    )
    scope = _scope("work")
    before = scope.to_dict()
    with execution_context(scope=scope):
        resolver.resolve(OWNER)
        assert scope.to_dict() == before
    text = repr(resolver)
    assert "personal-fake-key" not in text
    assert "allow_account_override=True" in text


@pytest.mark.parametrize("value", ["true", "True", "1", 1, 0, None])
def test_allow_account_override_must_be_bool(tmp_path, value):
    accounts = _accounts(tmp_path)
    with pytest.raises(TypeError):
        AccountCredentialResolver(
            accounts, "openai", account_name="work", allow_account_override=value
        )
    with pytest.raises(TypeError):
        accounts.resolver("openai", account_name="work", allow_account_override=value)


def test_override_requires_explicit_account_name(tmp_path):
    accounts = _accounts(tmp_path)
    with pytest.raises(AccountError, match="requires an explicit account_name"):
        AccountCredentialResolver(accounts, "openai", allow_account_override=True)
    with pytest.raises(AccountError, match="requires an explicit account_name"):
        accounts.resolver("openai", allow_account_override=True)
