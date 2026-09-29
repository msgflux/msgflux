"""Experimental ChatGPT subscription transport for the Codex Responses API."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from contextlib import aclosing, closing
from contextvars import ContextVar
from hashlib import sha256
from pathlib import Path
from typing import Any

from msgflux.chat_messages import ChatMessages
from msgflux.exceptions import ModelProviderHTTPError
from msgflux.models.chat_api import PreparedChatRequest
from msgflux.models.chat_capabilities import (
    ChatAPIModeCapabilities,
    ChatProviderCapabilities,
)
from msgflux.models.chat_context import OpenAIResponsesContextAdapter
from msgflux.models.chat_transport import HTTPChatTransport
from msgflux.models.codex_credentials import (
    read_codex_credentials,
    resolve_codex_auth_file,
)
from msgflux.models.http_transport import merge_headers
from msgflux.models.model_credentials import (
    ModelCredentialResolver,
    ResolvedModelCredentials,
)
from msgflux.models.openai_compatible import (
    OpenAICompatibleChatCompletion,
    OpenAIResponsesAPI,
)
from msgflux.models.reasoning import OpenAIResponsesReasoningCodec
from msgflux.models.registry import register_model
from msgflux.models.sse import aiter_sse_json, iter_sse_json
from msgflux.runtime.context import get_thread_id

_sent_token_fingerprint: ContextVar[bytes | None] = ContextVar(
    "codex_sent_token_fingerprint", default=None
)


class CodexCredentialResolver(ModelCredentialResolver):
    """Read the OAuth file for every HTTP attempt."""

    def resolve(self, owner: Any) -> ResolvedModelCredentials:
        credential = read_codex_credentials(owner.auth_file)
        _sent_token_fingerprint.set(sha256(credential.access_token.encode()).digest())
        return ResolvedModelCredentials(
            headers={
                "Authorization": f"Bearer {credential.access_token}",
                "chatgpt-account-id": credential.account_id,
            }
        )


class CodexResponsesAPI(OpenAIResponsesAPI):
    """Translate the shared Responses request to the Codex SSE endpoint."""

    endpoint = "/codex/responses"

    def prepare_request(
        self, owner: Any, params: dict[str, Any]
    ) -> PreparedChatRequest:
        body = owner._adapt_responses_params(params)
        # Expand provider-specific fields before validating the Codex surface
        # and deriving session headers from an explicit prompt_cache_key.
        body.update(body.pop("extra_body", None) or {})
        input_items = list(body.pop("input", []))
        instructions = body.pop("instructions", None)
        if input_items and input_items[0].get("role") in {"system", "developer"}:
            system_item = input_items.pop(0)
            instructions = system_item.get("content")
        if not isinstance(instructions, str) or not instructions:
            instructions = "You are a helpful assistant."
        if body.get("store") is True:
            raise ValueError("Codex Responses requires `store=False`")
        body["store"] = False
        body["stream"] = True
        body["input"] = input_items
        body["instructions"] = instructions
        include = list(body.get("include") or [])
        if "reasoning.encrypted_content" not in include:
            include.append("reasoning.encrypted_content")
        body["include"] = include
        body.setdefault("text", {"verbosity": "low"})
        body.setdefault("tool_choice", "auto")
        cache_key = body.get("prompt_cache_key")
        if cache_key is not None:
            if not isinstance(cache_key, str) or not cache_key:
                raise ValueError("`prompt_cache_key` must be a non-empty string")
            # Codex applies the same 64-character limit to the session headers.
            cache_key = cache_key[:64]
            body["prompt_cache_key"] = cache_key
            headers = dict(body.get("extra_headers") or {})
            headers["session-id"] = cache_key
            headers["x-client-request-id"] = cache_key
            body["extra_headers"] = headers
        # The subscription endpoint has a smaller request surface than the
        # public Responses API. Fail locally on fields whose behavior is unknown.
        supported = {
            "model",
            "store",
            "stream",
            "input",
            "instructions",
            "tools",
            "tool_choice",
            "parallel_tool_calls",
            "temperature",
            "reasoning",
            "text",
            "include",
            "prompt_cache_key",
            "extra_headers",
            "_msgflux_stream",
        }
        unsupported = set(body) - supported
        if unsupported:
            raise ValueError(
                f"Unsupported Codex request options: {', '.join(sorted(unsupported))}"
            )
        return PreparedChatRequest(api=self.name, endpoint=self.endpoint, params=body)


class CodexChatTransport(HTTPChatTransport):
    """Consume Codex SSE for both streamed and ordinary model calls."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("max_retries", 0)
        super().__init__(**kwargs)

    @staticmethod
    def _headers(request: PreparedChatRequest) -> dict[str, str]:
        return merge_headers(
            request.headers,
            {
                "accept": "text/event-stream",
                "content-type": "application/json",
                "OpenAI-Beta": "responses=experimental",
                "originator": "msgflux",
                "User-Agent": "msgflux",
            },
        )

    def create(self, owner: Any, request: PreparedChatRequest) -> Any:
        if request.params.get("_msgflux_stream"):
            return self._stream(owner, request)
        completed = None
        output_items: dict[int, Any] = {}
        for event in self._stream(owner, request):
            if event.type == "response.output_item.done":
                output_items[event.output_index] = event.item
            elif event.type == "response.completed":
                completed = event.response
            elif event.type in {"response.failed", "response.incomplete", "error"}:
                raise RuntimeError(f"Codex response ended with {event.type}")
        if completed is None:
            raise RuntimeError("Codex stream ended without a completed response")
        if output_items:
            completed = dict(completed)
            completed["output"] = [
                output_items[index] for index in sorted(output_items)
            ]
        return owner.api_adapter.decode_response(completed)

    async def acreate(self, owner: Any, request: PreparedChatRequest) -> Any:
        if request.params.get("_msgflux_stream"):
            return self._astream(owner, request)
        completed = None
        output_items: dict[int, Any] = {}
        async for event in self._astream(owner, request):
            if event.type == "response.output_item.done":
                output_items[event.output_index] = event.item
            elif event.type == "response.completed":
                completed = event.response
            elif event.type in {"response.failed", "response.incomplete", "error"}:
                raise RuntimeError(f"Codex response ended with {event.type}")
        if completed is None:
            raise RuntimeError("Codex stream ended without a completed response")
        if output_items:
            completed = dict(completed)
            completed["output"] = [
                output_items[index] for index in sorted(output_items)
            ]
        return owner.api_adapter.decode_response(completed)

    def _stream(self, owner: Any, request: PreparedChatRequest) -> Iterator[Any]:
        for attempt in range(2):
            emitted = False
            _sent_token_fingerprint.set(None)
            try:
                with closing(
                    self.http.stream(
                        owner,
                        request.endpoint,
                        headers=self._headers(request),
                        json=self._body(request),
                        iterate=lambda response: iter_sse_json(response.iter_lines()),
                    )
                ) as stream:
                    for payload in stream:
                        emitted = True
                        yield owner.api_adapter.decode_stream_event(payload)
                return
            except ModelProviderHTTPError as exc:
                if exc.status_code != 401:
                    raise RuntimeError(
                        f"Codex request failed with HTTP {exc.status_code}"
                    ) from None
                if emitted:
                    raise RuntimeError(
                        "Codex request failed after streaming began"
                    ) from None
                updated = read_codex_credentials(owner.auth_file).access_token
                if (
                    attempt
                    or sha256(updated.encode()).digest()
                    == _sent_token_fingerprint.get()
                ):
                    raise RuntimeError(
                        "Codex login expired or revoked; renew it in Codex CLI or Tau"
                    ) from None

    async def _astream(
        self, owner: Any, request: PreparedChatRequest
    ) -> AsyncIterator[Any]:
        for attempt in range(2):
            emitted = False
            _sent_token_fingerprint.set(None)
            try:
                async with aclosing(
                    self.http.astream(
                        owner,
                        request.endpoint,
                        headers=self._headers(request),
                        json=self._body(request),
                        iterate=lambda response: aiter_sse_json(response.aiter_lines()),
                    )
                ) as stream:
                    async for payload in stream:
                        emitted = True
                        yield owner.api_adapter.decode_stream_event(payload)
                return
            except ModelProviderHTTPError as exc:
                if exc.status_code != 401:
                    raise RuntimeError(
                        f"Codex request failed with HTTP {exc.status_code}"
                    ) from None
                if emitted:
                    raise RuntimeError(
                        "Codex request failed after streaming began"
                    ) from None
                updated = read_codex_credentials(owner.auth_file).access_token
                if (
                    attempt
                    or sha256(updated.encode()).digest()
                    == _sent_token_fingerprint.get()
                ):
                    raise RuntimeError(
                        "Codex login expired or revoked; renew it in Codex CLI or Tau"
                    ) from None

    @staticmethod
    def _body(request: PreparedChatRequest) -> dict[str, Any]:
        body = request.json
        body.pop("_msgflux_stream", None)
        return body


@register_model
class OpenAICodexChatCompletion(OpenAICompatibleChatCompletion):
    """Chat completion backed by a read-only Codex CLI or Tau OAuth file."""

    provider = "openai-codex"
    display_name = "OpenAI Codex"
    base_url = "https://chatgpt.com/backend-api"
    chat_transport = CodexChatTransport
    credential_resolver = CodexCredentialResolver()
    native_tools = False
    capabilities = ChatProviderCapabilities(
        default_api_mode="responses",
        api_modes=(
            ChatAPIModeCapabilities(
                name="responses",
                adapter=CodexResponsesAPI(),
                reasoning_codec=OpenAIResponsesReasoningCodec(),
                reasoning_summary=True,
                encrypted_reasoning=True,
                assistant_commentary=True,
                request_reasoning_effort=True,
                context_adapter=OpenAIResponsesContextAdapter(),
            ),
        ),
        default_reasoning_codec=OpenAIResponsesReasoningCodec(),
    )

    def __init__(
        self, model_id: str, *, auth_file: str | Path | None = None, **kwargs: Any
    ) -> None:
        if not model_id.strip():
            raise ValueError("An explicit Codex model ID is required")
        self.auth_file = str(resolve_codex_auth_file(auth_file))
        if kwargs.get("base_url") is None:
            kwargs["base_url"] = self.base_url
        kwargs.setdefault("native_tools", False)
        super().__init__(model_id, **kwargs)

    def _get_api_key(self) -> str:
        # The compatible base calls this during initialization. OAuth stays
        # unresolved until the request-time credential resolver runs.
        return ""

    def _initialize(self) -> None:
        if not isinstance(getattr(self, "chat_transport", None), CodexChatTransport):
            self.chat_transport = CodexChatTransport()
        self.credential_resolver = CodexCredentialResolver()
        self.api_mode_capabilities = self.capabilities.mode(self.api_mode)
        self.api_adapter = self.api_mode_capabilities.adapter
        self.reasoning_codec = self.api_mode_capabilities.reasoning_codec
        self._uses_canonical_history = True
        super()._initialize()

    def _adapt_responses_params(self, params: dict[str, Any]) -> dict[str, Any]:
        stream = params.get("stream", False)
        adapted = super()._adapt_responses_params(params)
        adapted["_msgflux_stream"] = bool(stream)
        return adapted

    def _build_generation_params(self, messages: Any, *args: Any, **kwargs: Any):
        params = super()._build_generation_params(messages, *args, **kwargs)
        extra_body = params.get("extra_body") or {}
        if "prompt_cache_key" not in params and "prompt_cache_key" not in extra_body:
            thread_id = get_thread_id()
            if thread_id is None and isinstance(messages, ChatMessages):
                thread_id = messages.thread_id
            if thread_id:
                params["prompt_cache_key"] = thread_id
        return params
