"""Read existing Codex OAuth credentials without taking ownership of them.

The Codex CLI and Tau credential layouts are private implementation details, so
all format-specific parsing stays in this module. This reader never writes or
refreshes the source file.
"""

from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path
from typing import Mapping

import msgspec


class CodexOAuthCredentials(msgspec.Struct, frozen=True):
    """Normalized request credentials; secret fields are omitted from repr."""

    access_token: str
    account_id: str
    expires_at: float | None = None

    def __repr__(self) -> str:
        return f"{type(self).__name__}(expires_at={self.expires_at!r})"


def resolve_codex_auth_file(
    auth_file: str | Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    home: str | Path | None = None,
) -> Path:
    """Resolve explicit path, environment override, then Codex CLI default."""
    env = os.environ if environ is None else environ
    selected = (
        auth_file if auth_file is not None else env.get("MSGFLUX_CODEX_AUTH_FILE")
    )
    if selected is None:
        home_dir = Path.home() if home is None else Path(home)
        selected = home_dir / ".codex" / "auth.json"
    return Path(selected).expanduser()


def read_codex_credentials(
    auth_file: str | Path | None = None,
    *,
    now: float | None = None,
    environ: Mapping[str, str] | None = None,
    home: str | Path | None = None,
) -> CodexOAuthCredentials:
    """Read and normalize a Codex CLI or Tau OAuth credential file.

    Errors intentionally omit the file path and parsed values because paths may
    themselves contain sensitive information and OAuth material must never be
    included in diagnostics.
    """
    path = resolve_codex_auth_file(auth_file, environ=environ, home=home)
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise ValueError("Codex OAuth credential file was not found") from None
    except PermissionError:
        raise ValueError("Codex OAuth credential file cannot be read") from None
    except UnicodeDecodeError:
        raise ValueError("Codex OAuth credential file is not valid UTF-8") from None
    except OSError:
        raise ValueError("Codex OAuth credential file cannot be read") from None
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise ValueError("Codex OAuth credential file is not valid JSON") from None
    if not isinstance(data, dict):
        raise ValueError("Codex OAuth credential file has an unsupported format")

    access_token, account_id, expires_at = _parse_credentials(data)

    if not isinstance(access_token, str) or not access_token.strip():
        raise ValueError("Codex OAuth credential file is missing an access token")
    if not isinstance(account_id, str) or not account_id.strip():
        raise ValueError("Codex OAuth credential file is missing an account ID")
    current_time = time.time() if now is None else now
    if expires_at is not None and expires_at <= current_time:
        raise ValueError(
            "Codex OAuth access token has expired; renew login with Codex CLI or Tau"
        )
    return CodexOAuthCredentials(
        access_token=access_token,
        account_id=account_id,
        expires_at=expires_at,
    )


def _parse_credentials(data: dict) -> tuple[object, object, float | None]:
    auth_mode = data.get("auth_mode")
    if auth_mode in {"apikey", "api_key"}:
        raise ValueError("Codex provider requires OAuth credentials, not an API key")
    if auth_mode not in {"chatgpt", "chatgptAuthTokens"}:
        if data.get("OPENAI_API_KEY"):
            raise ValueError(
                "Codex provider requires OAuth credentials, not an API key"
            )
        if auth_mode is not None:
            raise ValueError("Codex OAuth credential file has an unsupported format")
    if _looks_like_cli(data):
        tokens = data["tokens"]
        access_token = tokens.get("access_token")
        account_id = tokens.get("account_id")
        return access_token, account_id, _jwt_expiration(access_token)

    entry = data.get("openai-codex")
    if not isinstance(entry, dict) or entry.get("type") != "oauth":
        raise ValueError("Codex OAuth credential file has an unsupported format")
    raw_expiry = entry.get("expires")
    expires_at = _tau_expiration(raw_expiry)
    if raw_expiry is not None and expires_at is None:
        raise ValueError("Codex OAuth credential expiry is invalid")
    return entry.get("access"), entry.get("account_id"), expires_at


def _looks_like_cli(data: dict) -> bool:
    return isinstance(data.get("tokens"), dict) and (
        data.get("auth_mode") in {"chatgpt", "chatgptAuthTokens"}
        or "last_refresh" in data
        or "access_token" in data["tokens"]
    )


def _tau_expiration(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        return None
    # Tau stores Unix milliseconds.
    return float(value) / 1000.0


def _jwt_expiration(token: object) -> float | None:
    if not isinstance(token, str):
        return None
    parts = token.split(".")
    if len(parts) != 3:
        return None
    try:
        payload = parts[1] + "=" * (-len(parts[1]) % 4)
        claims = json.loads(base64.urlsafe_b64decode(payload.encode("ascii")))
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    exp = claims.get("exp") if isinstance(claims, dict) else None
    if isinstance(exp, bool) or not isinstance(exp, (float, int)):
        return None
    return float(exp)


__all__ = [
    "CodexOAuthCredentials",
    "read_codex_credentials",
    "resolve_codex_auth_file",
]
