"""Managed OAuth login and token refresh for Codex CLI accounts.

Codex CLI owns its login protocol. We run it with a private, temporary
``CODEX_HOME`` and only publish its validated native credential document after
it exits successfully. Managed refreshes update that same native document
while holding the account store's interprocess lock.
"""

from __future__ import annotations

import asyncio
import base64
import importlib
import json
import os
import shutil
import sys
import tempfile
import time
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from msgflux.models.codex_credentials import (
    CodexOAuthCredentials,
    parse_codex_credentials,
)

if TYPE_CHECKING:
    from msgflux.accounts.store import OAuthCredential

OAUTH_ENDPOINT = "https://auth.openai.com/oauth/token"
# This public Codex CLI client ID is also used by the documented legacy Codex
# refresh flow. The CLI remains responsible for initial login and PKCE.
CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
REFRESH_SKEW_SECONDS = 300
_AUTH_OVERRIDES = (
    "OPENAI_API_KEY",
    "CODEX_API_KEY",
    "CODEX_ACCESS_TOKEN",
    "OPENAI_FEDERATION_RULE_ID",
    "OPENAI_IDENTITY_TOKEN_FILE",
)


async def login(
    accounts: Any, provider: str, *, account_name: str = "default"
) -> OAuthCredential:
    """Run Codex CLI login in an isolated home and publish its OAuth grant."""
    if provider != "openai-codex":
        raise ValueError(f"OAuth login is not supported for provider {provider!r}")
    return await login_codex(accounts, account_name)


async def login_codex(accounts: Any, account_name: str = "default") -> OAuthCredential:
    """Authenticate an account through Codex CLI without touching user Codex data."""
    executable = shutil.which("codex")
    if executable is None:
        raise RuntimeError(
            "Codex CLI was not found; install it and retry account login"
        )

    account_dir = accounts.account_dir(account_name)
    account_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    staging = Path(tempfile.mkdtemp(prefix=".codex-login-", dir=account_dir))
    try:
        env = os.environ.copy()
        env["CODEX_HOME"] = str(staging)
        # These Codex-documented variables take precedence over stored OAuth,
        # API-key, or WIF login. Never let inherited identities select a
        # different login in this isolated CLI invocation.
        for name in _AUTH_OVERRIDES:
            env.pop(name, None)
        # Never let inherited settings redirect the child to a keychain or the
        # user's normal Codex home. The config override is scoped to this child.
        try:
            process = await asyncio.create_subprocess_exec(
                executable,
                "-c",
                "cli_auth_credentials_store=file",
                "login",
                env=env,
                stdout=sys.stderr,
            )
        except OSError:
            raise RuntimeError(
                "Could not start Codex CLI; verify the installation"
            ) from None
        try:
            return_code = await process.wait()
        except asyncio.CancelledError:
            try:
                process.terminate()
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(process.wait(), timeout=5)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
            raise
        if return_code != 0:
            raise RuntimeError(
                f"Codex CLI login did not complete (exit status {return_code}); "
                "the existing account credentials were kept"
            )

        auth_file = staging / "auth.json"
        payload = _read_native_payload(auth_file)
        credential = _validate_refreshable_payload(payload)
        if credential.expires_at is not None and credential.expires_at <= time.time():
            raise ValueError(
                "Codex CLI produced an expired login; run account login again"
            )
        # Serialize publication with refresh so an older refresh cannot overwrite
        # this newly authenticated grant after it releases the account lock.
        commit = asyncio.create_task(
            asyncio.to_thread(
                _publish_login,
                accounts,
                payload,
                account_name,
            )
        )
        try:
            return await asyncio.shield(commit)
        except asyncio.CancelledError:
            await commit
            raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _publish_login(accounts, payload: dict, account_name: str):
    with accounts.locked("openai-codex", account_name=account_name):
        return accounts.set_oauth("openai-codex", payload, account_name=account_name)


def resolve_codex_credentials(
    accounts: Any,
    account_name: str = "default",
    *,
    now: float | None = None,
) -> CodexOAuthCredentials:
    """Resolve a managed Codex grant, refreshing and persisting rotation safely."""
    with accounts.locked("openai-codex", account_name=account_name):
        payload = _oauth_payload(accounts, account_name)
        current = time.time() if now is None else now
        credential = _validate_refreshable_payload(payload)
        if (
            credential.expires_at is None
            or credential.expires_at > current + REFRESH_SKEW_SECONDS
        ):
            return credential
        rotated = _refresh(payload)
        accounts.set_oauth("openai-codex", rotated, account_name=account_name)
        return _validate_refreshable_payload(rotated)


async def aresolve_codex_credentials(
    accounts: Any,
    account_name: str = "default",
    *,
    now: float | None = None,
) -> CodexOAuthCredentials:
    """Run file locking and HTTP refresh off-loop; finish commit if cancelled."""
    worker = asyncio.create_task(
        asyncio.to_thread(resolve_codex_credentials, accounts, account_name, now=now)
    )
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        # A refresh token may be single-use. Wait for its rotated replacement
        # to commit under lock before allowing cancellation to escape.
        with suppress(Exception):
            await worker
        # Cancellation remains caller-visible even if the worker also failed.
        raise


def _oauth_payload(accounts: Any, account_name: str) -> dict[str, Any]:
    record = accounts.get_oauth("openai-codex", account_name=account_name)
    payload = getattr(record, "payload", None)
    if not isinstance(payload, dict):
        raise ValueError(
            "Managed Codex OAuth credentials are missing or invalid; run account login"
        )
    return payload


def _read_native_payload(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise RuntimeError(
            "Codex CLI completed without a readable OAuth credential file; "
            "the existing account credentials were kept"
        ) from None
    if not isinstance(data, dict):
        raise RuntimeError(
            "Codex CLI produced an unsupported credential file; "
            "the existing account credentials were kept"
        )
    return data


def _validate_refreshable_payload(payload: dict[str, Any]) -> CodexOAuthCredentials:
    _tokens(payload)
    # now=0 permits the resolver to refresh expired access tokens using their
    # refresh token. Parsing stays in memory and shares the external reader's
    # format validation.
    return parse_codex_credentials(payload, now=0)


def _tokens(payload: dict[str, Any]) -> dict[str, Any]:
    tokens = payload.get("tokens")
    if payload.get("auth_mode") not in {
        "chatgpt",
        "chatgptAuthTokens",
    } or not isinstance(tokens, dict):
        raise ValueError(
            "Managed Codex credentials have an unsupported format; run account login"
        )
    if not isinstance(tokens.get("access_token"), str) or not tokens["access_token"]:
        raise ValueError(
            "Managed Codex credentials are missing an access token; run account login"
        )
    if not isinstance(tokens.get("refresh_token"), str) or not tokens["refresh_token"]:
        raise ValueError(
            "Managed Codex credentials are missing a refresh token; run account login"
        )
    return tokens


def _refresh(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        httpx2 = importlib.import_module("httpx2")
    except ImportError:
        raise RuntimeError(
            "Codex token refresh requires the httpx2 dependency"
        ) from None

    tokens = _tokens(payload)
    try:
        with httpx2.Client(timeout=20.0) as client:
            response = client.post(
                OAUTH_ENDPOINT,
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": tokens["refresh_token"],
                    "client_id": CLIENT_ID,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
    except Exception:
        raise RuntimeError(
            "Codex token refresh failed; check network access and retry"
        ) from None
    data = _read_refresh_response(response)
    return _merge_refreshed_tokens(payload, data)


def _read_refresh_response(response: Any) -> dict[str, Any]:
    if response.status_code in {400, 401}:
        raise RuntimeError(
            "Codex login expired or was revoked; run account login to reconnect"
        )
    if response.status_code < 200 or response.status_code >= 300:
        raise RuntimeError(
            f"Codex token refresh failed with HTTP {response.status_code}"
        )
    try:
        data = response.json()
    except Exception:
        raise RuntimeError("Codex token refresh returned an invalid response") from None
    if (
        not isinstance(data, dict)
        or not isinstance(data.get("access_token"), str)
        or not data["access_token"]
    ):
        raise RuntimeError("Codex token refresh returned no access token")
    expires_in = data.get("expires_in")
    if (
        isinstance(expires_in, bool)
        or not isinstance(expires_in, (float, int))
        or expires_in <= 0
    ):
        raise RuntimeError("Codex token refresh returned an invalid expiry")

    return data


def _merge_refreshed_tokens(
    payload: dict[str, Any], data: dict[str, Any]
) -> dict[str, Any]:
    tokens = _tokens(payload)
    updated = dict(payload)
    updated_tokens = dict(tokens)
    updated_tokens["access_token"] = data["access_token"]
    updated_account_id = _account_id_from_access_token(data["access_token"])
    if updated_account_id is not None:
        updated_tokens["account_id"] = updated_account_id
    if isinstance(data.get("refresh_token"), str) and data["refresh_token"]:
        updated_tokens["refresh_token"] = data["refresh_token"]
    if isinstance(data.get("id_token"), str) and data["id_token"]:
        updated_tokens["id_token"] = data["id_token"]
    # Keep the token endpoint expiry when the access token is opaque. JWT exp
    # remains authoritative when the token parser can read it.
    updated_tokens["expires_at"] = time.time() + float(data["expires_in"])
    updated["tokens"] = updated_tokens
    updated["last_refresh"] = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    return updated


def _account_id_from_access_token(access_token: str) -> str | None:
    parts = access_token.split(".")
    if len(parts) != 3:
        return None
    try:
        claims = json.loads(
            base64.urlsafe_b64decode(parts[1] + "=" * (-len(parts[1]) % 4))
        )
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    auth_claim = (
        claims.get("https://api.openai.com/auth") if isinstance(claims, dict) else None
    )
    account_id = (
        auth_claim.get("chatgpt_account_id") if isinstance(auth_claim, dict) else None
    )
    return account_id if isinstance(account_id, str) and account_id else None


__all__ = [
    "aresolve_codex_credentials",
    "login",
    "login_codex",
    "resolve_codex_credentials",
]
