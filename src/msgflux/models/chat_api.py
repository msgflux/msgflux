"""Protocol boundary for chat-model wire APIs."""

from __future__ import annotations

from typing import Any

import msgspec

from msgflux.core.dotdict import dotdict
from msgflux.models.model_credentials import (
    BearerTokenCredentialResolver as _BearerTokenCredentialResolver,
)
from msgflux.models.model_credentials import (
    ModelCredentialResolver,
    ResolvedModelCredentials,
)


class PreparedChatRequest(msgspec.Struct, frozen=True):
    """A protocol request ready to be sent by a chat transport.

    ``params`` retains provider extension containers. HTTP transports use
    :attr:`json` and :attr:`headers`, which expand those containers into their
    wire representation.
    """

    api: str
    endpoint: str
    params: dict[str, Any]
    method: str = "POST"

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(api={self.api!r}, "
            f"endpoint={self.endpoint!r}, method={self.method!r})"
        )

    @property
    def json(self) -> dict[str, Any]:
        body = dict(self.params)
        extra_body = body.pop("extra_body", None)
        body.pop("extra_headers", None)
        if extra_body is not None:
            body.update(extra_body)
        return body

    @property
    def headers(self) -> dict[str, str]:
        extra_headers = self.params.get("extra_headers")
        return dict(extra_headers) if extra_headers is not None else {}


# Compatibility names for the original chat-specific public API. New model
# transports use the provider-neutral names from ``model_credentials``.
BearerTokenCredentialResolver = _BearerTokenCredentialResolver
ResolvedChatCredentials = ResolvedModelCredentials
ChatCredentialResolver = ModelCredentialResolver


class ChatAPIAdapter:
    """Translate the shared chat runtime to one wire API.

    Adapters are stateless and may be shared by model instances. Providers
    select adapters by ``api_mode``; the model remains responsible for common
    lifecycle concerns such as caching, timing, aborts, and response metadata.
    """

    name: str
    endpoint: str
    canonical_history: bool = False

    def prepare_request(
        self, owner: Any, params: dict[str, Any]
    ) -> PreparedChatRequest:
        raise NotImplementedError

    def build_generation_params(self, owner: Any, *args: Any, **kwargs: Any):
        raise NotImplementedError

    def process_output(self, owner: Any, *args: Any, **kwargs: Any):
        raise NotImplementedError

    def decode_response(self, payload: Any) -> Any:
        """Decode one JSON response into the attribute-compatible internal view."""
        return dotdict(payload) if isinstance(payload, dict) else payload

    def decode_stream_event(self, payload: Any) -> Any:
        """Decode one streamed JSON object into the internal wire view."""
        return dotdict(payload) if isinstance(payload, dict) else payload

    def stream(self, owner: Any, **kwargs: Any):
        raise NotImplementedError

    async def astream(self, owner: Any, **kwargs: Any):
        raise NotImplementedError


class ChatTransport:
    """Send prepared requests without owning protocol conversion."""

    def create(
        self,
        owner: Any,
        request: PreparedChatRequest,
    ) -> Any:
        raise NotImplementedError

    async def acreate(
        self,
        owner: Any,
        request: PreparedChatRequest,
    ) -> Any:
        raise NotImplementedError

    def close(self, owner: Any) -> None:
        """Close transport resources owned by one model instance."""

    async def aclose(self, owner: Any) -> None:
        """Close async transport resources owned by one model instance."""
        self.close(owner)
