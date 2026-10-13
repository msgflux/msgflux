import json
import multiprocessing
import os
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from msgflux.accounts import (
    AccountCredential,
    AccountError,
    AccountNotFoundError,
    Accounts,
    OAuthCredential,
)


def _hold_lock(root: str, ready, release) -> None:
    with Accounts(root).locked("openai", "default"):
        ready.set()
        release.wait(5)


def test_api_key_round_trip_is_private_and_redacted(tmp_path):
    accounts = Accounts(tmp_path / "accounts")
    result = accounts.set_api_key("openai", "very-secret")
    path = accounts.credential_path("openai")

    assert isinstance(result, AccountCredential)
    assert result.env_name == "OPENAI_API_KEY"
    assert "very-secret" not in repr(result)
    assert accounts.get("openai").value == "very-secret"
    assert path.stat().st_mode & 0o777 == 0o600
    assert path.parent.stat().st_mode & 0o777 == 0o700
    assert json.loads(path.read_text())["env_name"] == "OPENAI_API_KEY"


def test_accounts_keep_multiple_provider_credentials_separate(tmp_path):
    accounts = Accounts(tmp_path)
    accounts.set_api_key("openai", "one", "work")
    accounts.set_api_key("groq", "two", "work")

    assert accounts.get("openai", "work").value == "one"
    assert accounts.get("groq", "work").value == "two"
    assert accounts.list() == ["work"]
    assert accounts.info("work") == {
        "groq": {"kind": "api_key", "status": "configured"},
        "openai": {"kind": "api_key", "status": "configured"},
    }
    assert "one" not in repr(accounts.info("work"))


def test_oauth_payload_is_native_json_and_repr_redacted(tmp_path):
    accounts = Accounts(tmp_path)
    payload = {"tokens": {"access_token": "secret", "refresh_token": "refresh"}}
    record = accounts.set_oauth("openai-codex", payload)

    assert isinstance(record, OAuthCredential)
    assert "secret" not in repr(record)
    assert json.loads(accounts.credential_path("openai-codex").read_text()) == payload
    assert accounts.get_oauth("openai-codex").payload == payload


def test_missing_account_and_path_traversal_fail_without_creation(tmp_path):
    accounts = Accounts(tmp_path / "accounts")
    with pytest.raises(AccountNotFoundError):
        accounts.get("openai")
    assert not accounts.root.exists()
    assert accounts.info("default") == {}
    assert accounts.list() == []
    assert not accounts.root.exists()
    with pytest.raises(AccountError):
        accounts.set_api_key("openai", "x", "../outside")
    with pytest.raises(AccountError):
        accounts.set_api_key("openai", "x", "nested/name")


@pytest.mark.parametrize("value", ["", "   ", "\t\n"])
def test_api_key_rejects_empty_or_whitespace_value(tmp_path, value):
    with pytest.raises(AccountError, match="non-empty"):
        Accounts(tmp_path).set_api_key("openai", value)


def test_get_rejects_api_key_record_with_wrong_env_name(tmp_path):
    accounts = Accounts(tmp_path)
    path = accounts.credential_path("openai")
    path.write_text(
        json.dumps(
            {
                "provider": "openai",
                "kind": "api_key",
                "env_name": "OTHER_KEY",
                "value": "secret",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(AccountError, match="Invalid API key record"):
        accounts.get("openai")


def test_symlink_account_is_rejected(tmp_path):
    accounts = Accounts(tmp_path / "accounts")
    accounts.root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (accounts.root / "default").symlink_to(outside, target_is_directory=True)
    with pytest.raises(AccountError):
        accounts.set_api_key("openai", "x")


def test_lock_serializes_independent_processes(tmp_path):
    ctx = multiprocessing.get_context("spawn")
    ready, release = ctx.Event(), ctx.Event()
    process = ctx.Process(target=_hold_lock, args=(str(tmp_path), ready, release))
    process.start()
    assert ready.wait(5)
    started = time.monotonic()
    with Accounts(tmp_path).locked("openai"):
        elapsed = time.monotonic() - started
    release.set()
    process.join(5)
    assert not process.is_alive()
    assert elapsed >= 0.05


def test_atomic_replacement_leaves_only_complete_json(tmp_path):
    accounts = Accounts(tmp_path)
    accounts.set_api_key("openai", "first")
    accounts.set_api_key("openai", "second")
    assert accounts.get("openai").value == "second"
    assert not list(accounts.account_dir().glob(".credential-*"))


def test_concurrent_first_writes_create_account_and_provider_idempotently(tmp_path):
    accounts = Accounts(tmp_path / "accounts")
    workers = 12
    barrier = Barrier(workers)

    def write(index):
        barrier.wait(timeout=5)
        return accounts.set_api_key("openai", f"key-{index}", "shared").value

    with ThreadPoolExecutor(max_workers=workers) as executor:
        values = list(executor.map(write, range(workers)))

    assert accounts.get("openai", "shared").value in values
    assert accounts.account_dir("shared").stat().st_mode & 0o777 == 0o700
    assert (
        accounts.credential_path("openai", "shared").parent.stat().st_mode & 0o777
        == 0o700
    )


def test_info_does_not_echo_untrusted_record_metadata(tmp_path):
    import json

    accounts = Accounts(tmp_path)
    accounts.set_api_key("openai", "secret-value", account_name="work")
    path = accounts.credential_path("openai", account_name="work")
    payload = json.loads(path.read_text())
    payload["kind"] = "secret-value"
    path.write_text(json.dumps(payload))
    info = accounts.info("work")
    assert info["openai"]["kind"] == "api_key"
    assert "secret-value" not in str(info)
