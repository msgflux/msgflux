"""Private, atomic storage for named provider credential bundles."""

from __future__ import annotations

import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import msgspec

from msgflux.utils.validation import is_valid_account_name

if os.name == "nt":
    import msvcrt
else:
    import fcntl

from msgflux.accounts.providers import get_api_key_env, get_provider_auth


class AccountError(ValueError):
    """Base error for managed account operations."""


class AccountNotFoundError(AccountError):
    pass


class ProviderNotFoundError(AccountError):
    pass


class AccountCredential(msgspec.Struct, frozen=True):
    provider: str
    kind: str
    env_name: str
    value: str

    def __repr__(self) -> str:
        return (
            f"AccountCredential(provider={self.provider!r}, "
            "kind='api_key', value=<redacted>)"
        )


class OAuthCredential(msgspec.Struct, frozen=True):
    provider: str
    kind: str
    payload: dict[str, Any]

    def __repr__(self) -> str:
        return (
            f"OAuthCredential(provider={self.provider!r}, "
            "kind='oauth', payload=<redacted>)"
        )


class _StoredAPIKey(msgspec.Struct, frozen=True):
    provider: str
    kind: str
    env_name: str
    value: str


class Accounts:
    """Read and write account bundles under a private root directory."""

    def __init__(self, root: str | Path | None = None):
        self.root = (
            Path(root).expanduser()
            if root is not None
            else Path.home() / ".msgflux" / "accounts"
        )

    @staticmethod
    def _validate_name(value: str, what: str) -> str:
        if not is_valid_account_name(value):
            raise AccountError(f"Invalid {what} name")
        return value

    def _ensure_root(self) -> Path:
        root = self.root
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if root.is_symlink() or not root.is_dir():
            raise AccountError("Account root must be a real directory")
        os.chmod(root, 0o700)
        return root

    def account_dir(self, name: str = "default") -> Path:
        name = self._validate_name(name, "account")
        path = self._ensure_root() / name
        if path.exists() or path.is_symlink():
            if path.is_symlink() or not path.is_dir():
                raise AccountError("Account path must be a real directory")
        else:
            path.mkdir(mode=0o700, exist_ok=True)
        if path.is_symlink() or not path.is_dir():
            raise AccountError("Account path must be a real directory")
        os.chmod(path, 0o700)
        return path

    @staticmethod
    def _provider_name(provider: str) -> str:
        try:
            return get_provider_auth(provider).provider
        except ValueError as exc:
            raise ProviderNotFoundError(str(exc)) from exc

    def credential_path(self, provider: str, account_name: str = "default") -> Path:
        provider = self._provider_name(provider)
        directory = self.account_dir(account_name) / provider
        if directory.exists() or directory.is_symlink():
            if directory.is_symlink() or not directory.is_dir():
                raise AccountError("Provider path must be a real directory")
        else:
            directory.mkdir(mode=0o700, exist_ok=True)
        if directory.is_symlink() or not directory.is_dir():
            raise AccountError("Provider path must be a real directory")
        os.chmod(directory, 0o700)
        filename = (
            "auth.json"
            if get_provider_auth(provider).method == "oauth"
            else "credential.json"
        )
        return directory / filename

    def _existing_credential_path(self, provider: str, account_name: str) -> Path:
        provider = self._provider_name(provider)
        account_name = self._validate_name(account_name, "account")
        if self.root.is_symlink():
            raise AccountError("Account root must be a real directory")
        account_dir = self.root / account_name
        if not account_dir.exists():
            raise AccountNotFoundError(f"Account {account_name!r} does not exist")
        if account_dir.is_symlink() or not account_dir.is_dir():
            raise AccountError("Account path must be a real directory")
        provider_dir = account_dir / provider
        if provider_dir.is_symlink():
            raise AccountError("Provider path must be a real directory")
        filename = (
            "auth.json"
            if get_provider_auth(provider).method == "oauth"
            else "credential.json"
        )
        return provider_dir / filename

    @staticmethod
    def _atomic_write(path: Path, content: bytes) -> None:
        fd, temporary = tempfile.mkstemp(prefix=".credential-", dir=path.parent)
        primary_error = None
        try:
            with os.fdopen(fd, "wb") as stream:
                fd = None
                os.fchmod(stream.fileno(), 0o600)
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        except BaseException as error:
            primary_error = error
            raise
        finally:
            cleanup_errors = []
            if fd is not None:
                try:
                    os.close(fd)
                except OSError as error:
                    cleanup_errors.append(error)
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            except OSError as error:
                cleanup_errors.append(error)
            if primary_error is not None:
                for error in cleanup_errors:
                    primary_error.add_note(f"Credential cleanup failed: {error}")
            elif cleanup_errors:
                raise cleanup_errors[0]

    def set_api_key(
        self, provider: str, value: str, account_name: str = "default"
    ) -> AccountCredential:
        provider = self._provider_name(provider)
        env_name = get_api_key_env(provider)
        if not isinstance(value, str) or not value.strip():
            raise AccountError("API key must be a non-empty string")
        credential = AccountCredential(provider, "api_key", env_name, value)
        body = {
            "provider": provider,
            "kind": "api_key",
            "env_name": env_name,
            "value": value,
        }
        self._atomic_write(
            self.credential_path(provider, account_name), json.dumps(body).encode()
        )
        return credential

    def set_oauth(
        self, provider: str, payload: dict[str, Any], account_name: str = "default"
    ) -> OAuthCredential:
        provider = self._provider_name(provider)
        if get_provider_auth(provider).method != "oauth":
            raise AccountError(f"Provider {provider!r} does not use OAuth")
        try:
            payload = msgspec.convert(payload, type=dict[str, Any])
        except (msgspec.ValidationError, TypeError) as exc:
            raise AccountError("OAuth payload must be a JSON object") from exc
        credential = OAuthCredential(provider, "oauth", payload)
        self._atomic_write(
            self.credential_path(provider, account_name), json.dumps(payload).encode()
        )
        return credential

    def get(
        self, provider: str, account_name: str = "default"
    ) -> AccountCredential | OAuthCredential:
        provider = self._provider_name(provider)
        path = self._existing_credential_path(provider, account_name)
        if not path.exists():
            raise AccountNotFoundError(
                f"No credentials for {provider!r} in account {account_name!r}"
            )
        if path.is_symlink() or not path.is_file():
            raise AccountError("Credential path must be a regular file")
        try:
            raw_data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise AccountError(f"Invalid credential record for {provider!r}") from exc
        if get_provider_auth(provider).method == "api_key":
            try:
                data = msgspec.convert(raw_data, type=_StoredAPIKey)
            except (msgspec.ValidationError, TypeError) as exc:
                raise AccountError(f"Invalid API key record for {provider!r}") from exc
            if (
                data.provider != provider
                or data.kind != "api_key"
                or data.env_name != get_api_key_env(provider)
                or not data.value.strip()
            ):
                raise AccountError(f"Invalid API key record for {provider!r}")
            return AccountCredential(provider, "api_key", data.env_name, data.value)
        try:
            payload = msgspec.convert(raw_data, type=dict[str, Any])
        except (msgspec.ValidationError, TypeError) as exc:
            raise AccountError(f"Invalid OAuth record for {provider!r}") from exc
        return OAuthCredential(provider, "oauth", payload)

    def get_oauth(
        self, provider: str, account_name: str = "default"
    ) -> OAuthCredential:
        credential = self.get(provider, account_name)
        if not isinstance(credential, OAuthCredential):
            raise AccountError(f"Provider {provider!r} does not use OAuth")
        return credential

    @contextmanager
    def locked(self, provider: str, account_name: str = "default") -> Iterator[None]:
        path = self.credential_path(provider, account_name).with_suffix(".lock")
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            os.chmod(path, 0o600)
            if os.name == "nt":
                msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
            else:
                fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            if os.name == "nt":
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def list(self) -> list[str]:
        root = self.root
        if not root.exists():
            return []
        if root.is_symlink() or not root.is_dir():
            raise AccountError("Account root must be a real directory")
        return sorted(
            p.name for p in root.iterdir() if p.is_dir() and not p.is_symlink()
        )

    def info(self, account_name: str = "default") -> dict[str, dict[str, str]]:
        account_name = self._validate_name(account_name, "account")
        if self.root.is_symlink():
            raise AccountError("Account root must be a real directory")
        directory = self.root / account_name
        if not directory.exists():
            return {}
        if directory.is_symlink() or not directory.is_dir():
            raise AccountError("Account path must be a real directory")
        result: dict[str, dict[str, str]] = {}
        for provider_dir in sorted(directory.iterdir()):
            if provider_dir.is_symlink() or not provider_dir.is_dir():
                continue
            try:
                method = get_provider_auth(provider_dir.name).method
            except ValueError:
                continue
            path = provider_dir / (
                "auth.json" if method == "oauth" else "credential.json"
            )
            if path.is_symlink() or not path.is_file():
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not isinstance(data, dict):
                continue
            provider = provider_dir.name
            result[provider] = {
                "kind": method,
                "status": "configured",
            }
        return result

    def resolver(self, provider: str, *, account_name: str | None = None):
        """Build a request-time model resolver without reading credentials."""
        from msgflux.accounts.resolver import AccountCredentialResolver  # noqa: PLC0415

        return AccountCredentialResolver(self, provider, account_name=account_name)

    async def login(
        self, provider: str, account_name: str = "default"
    ) -> AccountCredential | OAuthCredential:
        from msgflux.accounts.oauth import login  # noqa: PLC0415

        return await login(self, provider, account_name=account_name)
