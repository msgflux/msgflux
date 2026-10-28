"""OpenCode gateway providers for Chat Completions, Responses and Anthropic.

``opencode-go`` and ``opencode-zen`` share the same gateway plumbing: the
OpenAI-compatible and native Anthropic protocols, Bearer authentication,
session headers and the Anthropic base URL derivation. Each public provider
class supplies only its credentials, endpoints and native backend selection.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import aclosing, closing
from typing import Any, ClassVar, Dict, Optional
from uuid import uuid4

from msgflux.chat_messages import ChatMessages
from msgflux.models.chat_capabilities import (
    ChatAPIModeCapabilities,
    ChatProviderCapabilities,
)
from msgflux.models.http_transport import merge_headers
from msgflux.models.model_credentials import BearerTokenCredentialResolver
from msgflux.models.openai_compatible import (
    OpenAIChatCompletionsAPI,
    OpenAICompatibleChatCompletion,
    OpenAIResponsesAPI,
)
from msgflux.models.provider_env import ProviderEnvBase
from msgflux.models.providers.anthropic import AnthropicChatCompletion
from msgflux.models.reasoning import (
    OpenAICompatibleReasoningCodec,
    OpenAIResponsesReasoningCodec,
)
from msgflux.models.registry import register_model
from msgflux.models.session import session_extra_headers
from msgflux.runtime.context import get_thread_id


class _BaseOpenCode(ProviderEnvBase):
    """Provider-neutral OpenCode gateway configuration.

    Concrete provider classes configure the credential and endpoint class
    variables and point ``_anthropic_messages_backend`` at their native
    Anthropic backend.
    """

    session_header: ClassVar[str] = "x-opencode-session"
    _openai_api_modes: ClassVar[tuple[str, ...]] = (
        "chat_completions",
        "responses",
    )
    _anthropic_messages_backend: ClassVar[type | None] = None

    @staticmethod
    def _anthropic_base_url(base_url: str) -> str:
        """Derive the native Anthropic host base from the OpenAI-compatible one.

        The base URL override keeps its original meaning: an
        OpenAI-compatible Chat Completions/Responses base that includes
        ``/v1``. Native Anthropic Messages mounts its own ``/v1/messages``
        below a host base, so ``/openai/v1`` becomes ``/anthropic`` and a
        generic trailing ``/v1`` is removed. Any other base is used verbatim
        as the Anthropic host base.
        """
        base = str(base_url).strip().rstrip("/")
        openai_suffix = "/openai/v1"
        if base.endswith(openai_suffix):
            return f"{base[: -len(openai_suffix)]}/anthropic"
        if base.endswith("/v1"):
            return base[: -len("/v1")].rstrip("/")
        return base


class _OpenCodeSessions:
    """Stable session and User-Agent headers shared by every API mode."""

    to_ignore = [
        *OpenAICompatibleChatCompletion.to_ignore,
        "_fallback_session_id",
    ]

    def _initialize(self):
        if not hasattr(self, "_fallback_session_id"):
            self._fallback_session_id = uuid4().hex
        super()._initialize()

    def _build_generation_params(self, messages, *args, **kwargs):
        params = super()._build_generation_params(messages, *args, **kwargs)
        if (
            isinstance(messages, ChatMessages)
            and messages.thread_id
            and not get_thread_id()
        ):
            params["extra_headers"] = merge_headers(
                {self.session_header: messages.thread_id},
                self.sampling_run_params.get("extra_headers") or {},
                params.get("extra_headers") or {},
            )
        return params

    def _with_session_headers(
        self,
        params: Dict[str, Any],
        *,
        messages: Any = None,
    ) -> Dict[str, Any]:
        """Add client identity and the highest-precedence session header.

        An explicit caller header always wins, matched case-insensitively.
        Otherwise prefer the active runtime thread, then
        ``ChatMessages.thread_id``, then the private per-instance fallback.
        """
        generated = session_extra_headers(self.session_header)
        if self.session_header not in generated:
            if isinstance(messages, ChatMessages) and messages.thread_id:
                generated[self.session_header] = messages.thread_id
            else:
                generated[self.session_header] = self._fallback_session_id
        params["extra_headers"] = merge_headers(
            generated,
            self.sampling_run_params.get("extra_headers") or {},
            params.get("extra_headers") or {},
        )
        return params

    def _adapt_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return self._with_session_headers(super()._adapt_params(params))

    def _adapt_responses_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return self._with_session_headers(super()._adapt_responses_params(params))


class _OpenCodeAnthropicMessages(
    _OpenCodeSessions,
    _BaseOpenCode,
    AnthropicChatCompletion,
):
    """Shared native Anthropic Messages backend for OpenCode gateways.

    This backend is selected by ``api_mode="anthropic_messages"`` and is not
    registered as a separate provider. It reuses the native Anthropic
    converters, signed thinking history, tool handling, caching and SSE
    expansion; only the gateway endpoint, Bearer authentication and session
    headers are specialized here.
    """

    credential_resolver = BearerTokenCredentialResolver()
    to_ignore = [
        *AnthropicChatCompletion.to_ignore,
        "_fallback_session_id",
    ]

    def __init__(
        self,
        model_id: str,
        *,
        base_url: Optional[str] = None,
        **kwargs: Any,
    ):
        resolved_base_url = (
            self._anthropic_base_url(base_url) if base_url is not None else None
        )
        super().__init__(model_id, base_url=resolved_base_url, **kwargs)

    def _get_base_url(self):
        return self._anthropic_base_url(super()._get_base_url())

    def _native_headers_for(
        self,
        params: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, str]:
        return merge_headers(
            super()._native_headers(),
            dict((params or {}).get("extra_headers") or {}),
        )

    def _native_headers(self) -> Dict[str, str]:
        return self._native_headers_for(self._with_session_headers({}))

    def _execute_model(self, **kwargs):
        self._raise_if_aborted()
        params = {**self.sampling_run_params, **kwargs}
        params = self._with_session_headers(
            params,
            messages=kwargs.get("messages"),
        )
        body = self._to_anthropic_body(params)
        headers = self._native_headers_for(params)
        if body.get("stream"):
            return self._stream_chunks(body, headers=headers)
        response = self.http_transport.request(
            self,
            self.endpoint,
            headers=headers,
            json=body,
        )
        self._raise_if_aborted()
        return self._anthropic_to_completion(response.json(), stream=False)

    async def _aexecute_model(self, **kwargs):
        self._raise_if_aborted()
        params = {**self.sampling_run_params, **kwargs}
        params = self._with_session_headers(
            params,
            messages=kwargs.get("messages"),
        )
        body = self._to_anthropic_body(params)
        headers = self._native_headers_for(params)
        if body.get("stream"):
            return self._astream_chunks(body, headers=headers)
        response = await self.http_transport.arequest(
            self,
            self.endpoint,
            headers=headers,
            json=body,
        )
        self._raise_if_aborted()
        return self._anthropic_to_completion(response.json(), stream=False)

    def _stream_chunks(
        self,
        body: Dict[str, Any],
        *,
        headers: Optional[Dict[str, str]] = None,
    ):
        stream = self.http_transport.stream(
            self,
            self.endpoint,
            headers=headers or self._native_headers(),
            json=body,
            iterate=lambda response: self._expand_stream_events(
                self._iter_sse_events(response.iter_lines())
            ),
        )
        with closing(stream):
            yield from stream

    async def _astream_chunks(
        self,
        body: Dict[str, Any],
        *,
        headers: Optional[Dict[str, str]] = None,
    ):
        stream = self.http_transport.astream(
            self,
            self.endpoint,
            headers=headers or self._native_headers(),
            json=body,
            iterate=lambda response: self._expand_astream_events(
                self._aiter_sse_events(response.aiter_lines())
            ),
        )
        async with aclosing(stream):
            async for item in stream:
                yield item


class _OpenCodeChatCompletion(
    _OpenCodeSessions,
    _BaseOpenCode,
    OpenAICompatibleChatCompletion,
):
    """Shared OpenCode facade over the three explicitly selected protocols.

    Concrete providers inherit this class and set their provider metadata,
    endpoints and ``_anthropic_messages_backend``. Only the concrete classes
    are registered, so one provider never answers for another provider's
    namespace.

    ``api_mode="chat_completions"`` (the default) and ``api_mode="responses"``
    use the OpenAI-compatible endpoints. The explicit
    ``api_mode="anthropic_messages"`` returns the provider's native Anthropic
    backend below the derived Anthropic host base.
    """

    capabilities = ChatProviderCapabilities(
        default_api_mode="chat_completions",
        api_modes=(
            ChatAPIModeCapabilities(
                name="chat_completions",
                adapter=OpenAIChatCompletionsAPI(),
                request_reasoning_effort=True,
            ),
            ChatAPIModeCapabilities(
                name="responses",
                adapter=OpenAIResponsesAPI(),
                reasoning_codec=OpenAIResponsesReasoningCodec(),
                request_reasoning_effort=True,
            ),
        ),
        default_reasoning_codec=OpenAICompatibleReasoningCodec(),
    )

    def __new__(cls, model_id: str, *args: Any, **kwargs: Any):
        # Anthropic Messages needs the native converter/streaming class. Keep
        # one registered provider per namespace while returning that concrete
        # backend for this explicit mode. The backend initializes itself in
        # ``__new__``; a non-facade instance skips the facade ``__init__``.
        backend = cls._anthropic_messages_backend
        if backend is not None:
            api_mode = kwargs.get("api_mode")
            if api_mode == "anthropic_messages":
                return backend(model_id, *args, **kwargs)
            if api_mode is not None and api_mode not in cls._openai_api_modes:
                supported = ", ".join((*cls._openai_api_modes, "anthropic_messages"))
                raise ValueError(
                    f"{cls.__name__} does not support `api_mode={api_mode!r}`; "
                    f"supported modes: {supported}."
                )
        return super().__new__(cls)

    @property
    def supported_api_modes(self) -> tuple[str, ...]:
        """Return every protocol the provider exposes."""
        return (*self._openai_api_modes, "anthropic_messages")

    @classmethod
    def from_serialized(cls, state: Mapping[str, Any]):
        """Restore the concrete backend selected by the serialized API mode."""
        backend = cls._anthropic_messages_backend
        if (
            backend is not None
            and isinstance(state, Mapping)
            and state.get("api_mode") == "anthropic_messages"
        ):
            return backend.from_serialized(state)
        return super().from_serialized(state)


class _OpenCodeGoConfig:
    """OpenCode Go gateway metadata shared by its facade and backend."""

    provider: str = "opencode-go"
    display_name: str = "OpenCode Go"
    api_key_env: str = "OPENCODE_GO_API_KEY"
    base_url_env: str = "OPENCODE_GO_BASE_URL"
    base_url: str = "https://opencode.ai/inference/go/openai/v1"


class _OpenCodeZenConfig:
    """OpenCode Zen gateway metadata shared by its facade and backend."""

    provider: str = "opencode-zen"
    display_name: str = "OpenCode Zen"
    api_key_env: str = "OPENCODE_ZEN_API_KEY"
    base_url_env: str = "OPENCODE_ZEN_BASE_URL"
    base_url: str = "https://opencode.ai/inference/openai/v1"


class _OpenCodeGoAnthropicMessages(
    _OpenCodeGoConfig,
    _OpenCodeAnthropicMessages,
):
    """OpenCode Go native Anthropic Messages backend."""


class _OpenCodeZenAnthropicMessages(
    _OpenCodeZenConfig,
    _OpenCodeAnthropicMessages,
):
    """OpenCode Zen native Anthropic Messages backend."""


@register_model
class OpenCodeGoChatCompletion(
    _OpenCodeGoConfig,
    _OpenCodeChatCompletion,
):
    """OpenCode Go chat completion through three explicitly selected protocols.

    ``api_mode="chat_completions"`` (the default) and
    ``api_mode="responses"`` use the OpenAI-compatible endpoints at
    ``https://opencode.ai/inference/go/openai/v1``. The explicit
    ``api_mode="anthropic_messages"`` selects the native Anthropic Messages
    endpoint below ``https://opencode.ai/inference/go/anthropic``.

    Every protocol authenticates with ``Authorization: Bearer`` using
    ``OPENCODE_GO_API_KEY``. Requests identify msgFlux with its own ``User-Agent``
    and carry a stable ``x-opencode-session`` for routing and prompt caching.
    An explicit session header wins regardless of casing, followed by the
    active runtime thread, ``ChatMessages.thread_id``, and a private
    per-instance fallback.

    OpenCode Go's Google/Interactions endpoint is not implemented.
    """

    _anthropic_messages_backend = _OpenCodeGoAnthropicMessages


@register_model
class OpenCodeZenChatCompletion(
    _OpenCodeZenConfig,
    _OpenCodeChatCompletion,
):
    """OpenCode Zen chat completion through three explicitly selected protocols.

    ``api_mode="chat_completions"`` (the default) and
    ``api_mode="responses"`` use the OpenAI-compatible endpoints at
    ``https://opencode.ai/inference/openai/v1``. The explicit
    ``api_mode="anthropic_messages"`` selects the native Anthropic Messages
    endpoint below ``https://opencode.ai/inference/anthropic``.

    Every protocol authenticates with ``Authorization: Bearer`` using
    ``OPENCODE_ZEN_API_KEY``. Requests identify msgFlux with its own
    ``User-Agent`` and carry a stable ``x-opencode-session`` for routing and
    prompt caching. An explicit session header wins regardless of casing,
    followed by the active runtime thread, ``ChatMessages.thread_id``, and a
    private per-instance fallback.

    OpenCode Zen's Google/Interactions endpoint is not implemented.
    """

    _anthropic_messages_backend = _OpenCodeZenAnthropicMessages


__all__ = ["OpenCodeGoChatCompletion", "OpenCodeZenChatCompletion"]
