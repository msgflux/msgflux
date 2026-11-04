"""Request-scoped account credentials for model transports."""

from __future__ import annotations

import asyncio

from msgflux.accounts.providers import get_provider_auth
from msgflux.accounts.store import AccountError, Accounts
from msgflux.models.model_credentials import (
    APIKeyCredentialResolver,
    BearerTokenCredentialResolver,
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
        provider_catalog=None,
    ) -> None:
        self.accounts = accounts
        catalog = (
            provider_catalog
            if provider_catalog is not None
            else getattr(accounts, "provider_catalog", None)
        )
        self.provider_catalog = catalog
        # Catalog-aware: dynamic aliases resolve through the Accounts catalog
        # (or the active catalog); builtins are unchanged.
        self.auth = get_provider_auth(provider, provider_catalog=catalog)
        self.provider = self.auth.provider
        if account_name is not None:
            Accounts._validate_name(account_name, "account")
        self.account_name = account_name

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(provider={self.provider!r}, "
            f"account_name={self.account_name!r})"
        )

    def _selected_account(self) -> str:
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

    def _validate_owner(self, owner) -> None:
        if getattr(owner, "provider", self.provider) != self.provider:
            raise AccountError("The account resolver provider does not match the model")

    def _api_key_headers(self, owner, value: str) -> dict[str, str]:
        """Emit the managed key in the model class's native auth header.

        The owner's class-level credential resolver is the base provider's
        native resolver (e.g. Anthropic's ``x-api-key``); the injected
        account resolver shadows it on the instance. Reuse that native
        resolver's ``resolve_api_key`` for the managed key, falling back to
        the legacy Bearer format for resolvers without it.
        """
        native = getattr(type(owner), "credential_resolver", None)
        if isinstance(native, APIKeyCredentialResolver):
            return native.resolve_api_key(value).headers
        return BearerTokenCredentialResolver().resolve_api_key(value).headers

    def _is_oauth(self) -> bool:
        return self.auth.method == "oauth"

    def _check_oauth_adapter(self) -> None:
        if self.auth.oauth_adapter != "codex":
            raise AccountError(
                f"OAuth account resolution is not implemented for provider "
                f"{self.provider!r}"
            )

    def resolve(self, owner) -> ResolvedModelCredentials:
        self._validate_owner(owner)
        selected = self._selected_account()
        if self._is_oauth():
            self._check_oauth_adapter()
            from msgflux.accounts.oauth import (  # noqa: PLC0415
                resolve_codex_credentials,
            )

            credential = resolve_codex_credentials(
                self.accounts, account_name=selected, provider=self.provider
            )
            return ResolvedModelCredentials(
                headers={
                    "Authorization": f"Bearer {credential.access_token}",
                    "chatgpt-account-id": credential.account_id,
                }
            )
        credential = self.accounts.get(self.provider, account_name=selected)
        return ResolvedModelCredentials(
            headers=self._api_key_headers(owner, credential.value)
        )

    async def aresolve(self, owner) -> ResolvedModelCredentials:
        self._validate_owner(owner)
        selected = self._selected_account()
        if self._is_oauth():
            self._check_oauth_adapter()
            from msgflux.accounts.oauth import (  # noqa: PLC0415
                aresolve_codex_credentials,
            )

            credential = await aresolve_codex_credentials(
                self.accounts, account_name=selected, provider=self.provider
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
            headers=self._api_key_headers(owner, credential.value)
        )
