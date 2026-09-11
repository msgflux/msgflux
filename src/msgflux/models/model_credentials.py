"""Request-time credentials shared by model transports."""

from __future__ import annotations

from typing import Any

import msgspec


class ResolvedModelCredentials(msgspec.Struct, frozen=True):
    """Secret request material resolved immediately before transport."""

    headers: dict[str, str] = msgspec.field(default_factory=dict)
    base_url: str | None = None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class ModelCredentialResolver:
    """Resolve model request authentication without exposing its storage."""

    def resolve(self, owner: Any) -> ResolvedModelCredentials:
        raise NotImplementedError

    async def aresolve(self, owner: Any) -> ResolvedModelCredentials:
        return self.resolve(owner)


class BearerTokenCredentialResolver(ModelCredentialResolver):
    """Resolve the provider API key as a Bearer token at request time."""

    def resolve(self, owner: Any) -> ResolvedModelCredentials:
        return ResolvedModelCredentials(
            headers={"Authorization": f"Bearer {owner._get_api_key()}"}
        )


__all__ = [
    "BearerTokenCredentialResolver",
    "ModelCredentialResolver",
    "ResolvedModelCredentials",
]
