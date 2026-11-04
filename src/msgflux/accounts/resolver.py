"""Request-scoped account credentials for model transports."""

from __future__ import annotations

import asyncio

from msgflux.accounts.providers import get_provider_auth
from msgflux.accounts.store import AccountError, Accounts
from msgflux.models.model_credentials import (
    ModelCredentialResolver,
    ResolvedModelCredentials,
)
from msgflux.runtime.context import get_execution_scope


class AccountCredentialResolver(ModelCredentialResolver):
    """Resolve one provider from the immutable account selection of the run."""

    def __init__(
        self,
        accounts: Accounts,
        provider: str,
        *,
        account_name: str | None = None,
        allow_account_override: bool = False,
    ) -> None:
        if not isinstance(allow_account_override, bool):
            raise TypeError("`allow_account_override` must be a boolean")
        if allow_account_override and account_name is None:
            raise AccountError(
                "allow_account_override requires an explicit account_name"
            )
        self.accounts = accounts
        self.provider = get_provider_auth(provider).provider
        if account_name is not None:
            Accounts._validate_name(account_name, "account")
        self.account_name = account_name
        self.allow_account_override = allow_account_override

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(provider={self.provider!r}, "
            f"account_name={self.account_name!r}, "
            f"allow_account_override={self.allow_account_override!r})"
        )

    def _selected_account(self) -> str:
        if self.allow_account_override:
            if self.account_name is None:
                raise AccountError(
                    "allow_account_override requires an explicit account_name"
                )
            return self.account_name
        scope = get_execution_scope()
        selected = scope.account_name
        if selected is not None:
            if self.account_name is not None and self.account_name != selected:
                raise AccountError(
                    "The model account conflicts with the execution account"
                )
            return selected
        if self.account_name is not None:
            return self.account_name
        if scope.run_id is not None:
            raise AccountError(
                "This execution has no recorded account selection; start a new "
                "execution with an explicit account"
            )
        return "default"

    def cache_identity(self, owner) -> dict[str, str]:
        self._validate_owner(owner)
        return {
            "provider": self.provider,
            "account_name": self._selected_account(),
            "accounts_root": str(self.accounts.root),
        }

    def _validate_owner(self, owner) -> None:
        if getattr(owner, "provider", self.provider) != self.provider:
            raise AccountError("The account resolver provider does not match the model")

    def resolve(self, owner) -> ResolvedModelCredentials:
        self._validate_owner(owner)
        selected = self._selected_account()
        if self.provider == "openai-codex":
            from msgflux.accounts.oauth import (  # noqa: PLC0415
                resolve_codex_credentials,
            )

            credential = resolve_codex_credentials(self.accounts, account_name=selected)
            return ResolvedModelCredentials(
                headers={
                    "Authorization": f"Bearer {credential.access_token}",
                    "chatgpt-account-id": credential.account_id,
                }
            )
        credential = self.accounts.get(self.provider, account_name=selected)
        return ResolvedModelCredentials(
            headers={"Authorization": f"Bearer {credential.value}"}
        )

    async def aresolve(self, owner) -> ResolvedModelCredentials:
        self._validate_owner(owner)
        selected = self._selected_account()
        if self.provider == "openai-codex":
            from msgflux.accounts.oauth import (  # noqa: PLC0415
                aresolve_codex_credentials,
            )

            credential = await aresolve_codex_credentials(
                self.accounts, account_name=selected
            )
            return ResolvedModelCredentials(
                headers={
                    "Authorization": f"Bearer {credential.access_token}",
                    "chatgpt-account-id": credential.account_id,
                }
            )
        credential = await asyncio.to_thread(
            self.accounts.get, self.provider, account_name=selected
        )
        return ResolvedModelCredentials(
            headers={"Authorization": f"Bearer {credential.value}"}
        )
