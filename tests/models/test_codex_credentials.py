"""Tests for the read-only Codex OAuth credential adapter."""

import base64
import json

import pytest

from msgflux.models.codex_credentials import (
    CodexOAuthCredentials,
    read_codex_credentials,
    resolve_codex_auth_file,
)


def _jwt(claims):
    encode = lambda value: (
        base64.urlsafe_b64encode(json.dumps(value).encode()).decode().rstrip("=")
    )
    return f"{encode({'alg': 'none'})}.{encode(claims)}.signature"


def _write(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.mark.parametrize("api_key_value", [None, "inactive-api-key"])
def test_reads_codex_cli_credentials_and_jwt_expiry(tmp_path, api_key_value):
    token = _jwt({"exp": 2_000_000_000})
    path = _write(
        tmp_path / "auth.json",
        {
            "auth_mode": "chatgpt",
            "OPENAI_API_KEY": api_key_value,
            "tokens": {"access_token": token, "account_id": "acct-cli"},
            "last_refresh": "yesterday",
        },
    )

    result = read_codex_credentials(path, now=1_900_000_000)

    assert result == CodexOAuthCredentials(
        access_token=token, account_id="acct-cli", expires_at=2_000_000_000
    )
    assert token not in repr(result)
    assert "acct-cli" not in repr(result)


def test_reads_tau_credentials_and_millisecond_expiry(tmp_path):
    path = _write(
        tmp_path / "credentials.json",
        {
            "openai-codex": {
                "type": "oauth",
                "access": "tau-access-secret",
                "refresh": "tau-refresh-secret",
                "expires": 2_000_000_000_000,
                "account_id": "acct-tau",
            }
        },
    )

    result = read_codex_credentials(path, now=1_900_000_000)

    assert result.access_token == "tau-access-secret"
    assert result.account_id == "acct-tau"
    assert result.expires_at == 2_000_000_000
    assert "tau-access-secret" not in repr(result)
    assert "tau-refresh-secret" not in repr(result)


def test_explicit_auth_file_takes_precedence_over_environment(tmp_path):
    selected = tmp_path / "explicit.json"

    assert (
        resolve_codex_auth_file(
            selected,
            environ={"MSGFLUX_CODEX_AUTH_FILE": str(tmp_path / "env.json")},
            home=tmp_path,
        )
        == selected
    )


def test_environment_auth_file_takes_precedence_over_default(tmp_path):
    selected = tmp_path / "env.json"

    assert (
        resolve_codex_auth_file(
            environ={"MSGFLUX_CODEX_AUTH_FILE": str(selected)}, home=tmp_path
        )
        == selected
    )


def test_defaults_to_codex_cli_file(tmp_path):
    assert resolve_codex_auth_file(home=tmp_path) == tmp_path / ".codex" / "auth.json"


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            {
                "auth_mode": "apikey",
                "OPENAI_API_KEY": "secret",
                "tokens": {"access_token": "secret", "account_id": "acct"},
            },
            "not an API key",
        ),
        (
            {
                "auth_mode": "apikey",
                "OPENAI_API_KEY": None,
                "tokens": {"access_token": "secret", "account_id": "acct"},
            },
            "not an API key",
        ),
        (
            {
                "OPENAI_API_KEY": "secret",
                "tokens": {"access_token": "secret", "account_id": "acct"},
            },
            "not an API key",
        ),
        ({"unexpected": "secret"}, "unsupported format"),
        (
            {"auth_mode": "chatgpt", "tokens": {"account_id": "acct"}},
            "missing an access token",
        ),
        (
            {
                "openai-codex": {
                    "type": "oauth",
                    "access": "secret",
                    "account_id": "acct",
                    "expires": "secret-expiry",
                }
            },
            "expiry is invalid",
        ),
    ],
)
def test_rejects_invalid_formats_without_echoing_secrets(tmp_path, payload, message):
    path = _write(tmp_path / "auth.json", payload)

    with pytest.raises(ValueError, match=message) as exc_info:
        read_codex_credentials(path)

    assert "secret" not in str(exc_info.value)
    assert str(path) not in str(exc_info.value)


def test_rejects_expired_tau_token(tmp_path):
    path = _write(
        tmp_path / "credentials.json",
        {
            "openai-codex": {
                "type": "oauth",
                "access": "expired-secret",
                "account_id": "acct",
                "expires": 1_000,
            }
        },
    )

    with pytest.raises(ValueError, match="has expired") as exc_info:
        read_codex_credentials(path, now=2)

    assert "expired-secret" not in str(exc_info.value)


def test_missing_file_error_does_not_include_path(tmp_path):
    path = tmp_path / "private-path" / "auth.json"

    with pytest.raises(ValueError, match="was not found") as exc_info:
        read_codex_credentials(path)

    assert str(path) not in str(exc_info.value)


def test_invalid_utf8_error_does_not_include_path(tmp_path):
    path = tmp_path / "auth.json"
    path.write_bytes(b"\xff")

    with pytest.raises(ValueError, match="not valid UTF-8") as exc_info:
        read_codex_credentials(path)

    assert str(path) not in str(exc_info.value)
