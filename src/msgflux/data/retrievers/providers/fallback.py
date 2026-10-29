"""Ordered fallback composition for web fetch retrievers."""

import inspect
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

from msgflux.data.retrievers.base import BaseRetriever, BaseWebFetch
from msgflux.data.retrievers.providers.firecrawl import FirecrawlWebFetchRetriever
from msgflux.data.retrievers.providers.http import normalize_url
from msgflux.data.retrievers.providers.markdown import (
    DEFAULT_MARKDOWN_ENDPOINT,
    MarkdownWebFetchRetriever,
)
from msgflux.data.retrievers.registry import register_retriever
from msgflux.data.retrievers.retriever import Retriever
from msgflux.data.retrievers.types import WebFetchRetriever


def _entry_content(entry: Any) -> Optional[str]:
    if not isinstance(entry, Mapping):
        return None
    payload = entry.get("data", entry)
    if not isinstance(payload, Mapping):
        return None
    content = payload.get("content")
    return content if isinstance(content, str) else None


def _response_entries(response: Any) -> List[Any]:
    if isinstance(response, (list, tuple)):
        return list(response)
    data = getattr(response, "data", None)
    if data is None and isinstance(response, Mapping):
        data = response.get("data")
    return list(data) if isinstance(data, (list, tuple)) else []


def _close_children(children: Sequence[BaseRetriever]) -> None:
    """Best-effort cleanup used while aborting partial construction."""
    for child in children:
        close = getattr(child, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # noqa: S110 - preserve the construction error
                pass


@register_retriever
class FallbackWebFetchRetriever(BaseWebFetch, BaseRetriever, WebFetchRetriever):
    """Try web fetch retrievers in order for each URL.

    The chain owns and closes every runtime child it receives, including
    injected instances. Serialized state contains only child configuration;
    runtime children are reused at construction and rebuilt only when restoring
    serialized state.
    """

    provider = "fallback"
    to_ignore = [*BaseRetriever.to_ignore, "children"]

    def __init__(
        self,
        retrievers: Optional[
            Union[
                BaseRetriever,
                str,
                Mapping[str, Any],
                Sequence[Union[BaseRetriever, str, Mapping[str, Any]]],
            ]
        ] = None,
        *,
        timeout: Optional[float] = None,
        web_parser: Optional[str] = None,
        default_headers: Optional[Mapping[str, str]] = None,
    ):
        """Initialize a fallback web fetch chain.

        Args:
            retrievers:
                Ordered retriever instances, provider names, or serialized
                retriever specs. Defaults to markdown.new followed by Firecrawl.
            timeout:
                Timeout passed to default child retrievers.
            web_parser:
                markdown.new-compatible endpoint for the default chain.
            default_headers:
                Headers passed to the default markdown retriever.
        """
        self.timeout = timeout
        children: List[BaseRetriever] = []
        specs: List[Dict[str, Any]] = []

        try:
            if retrievers is None:
                children = self._default_children(
                    timeout=timeout,
                    web_parser=web_parser,
                    default_headers=default_headers,
                )
                specs = [self._serialize_child(child) for child in children]
            else:
                if isinstance(retrievers, (BaseRetriever, str, Mapping)):
                    requested = [retrievers]
                else:
                    requested = list(retrievers)
                if not requested:
                    raise ValueError("`retrievers` must contain at least one retriever")
                children, specs = self._build_requested(requested)
        except BaseException:
            _close_children(children)
            raise

        self.retrievers = specs
        self.children = children

    @staticmethod
    def _default_children(
        *,
        timeout: Optional[float],
        web_parser: Optional[str],
        default_headers: Optional[Mapping[str, str]],
    ) -> List[BaseRetriever]:
        children: List[BaseRetriever] = [
            MarkdownWebFetchRetriever(
                endpoint=web_parser or DEFAULT_MARKDOWN_ENDPOINT,
                default_headers=default_headers,
                timeout=timeout,
            )
        ]
        try:
            children.append(FirecrawlWebFetchRetriever(timeout=timeout))
        except BaseException:
            _close_children(children)
            raise
        return children

    def _build_requested(
        self,
        requested: Sequence[Union[BaseRetriever, str, Mapping[str, Any]]],
    ) -> tuple[List[BaseRetriever], List[Dict[str, Any]]]:
        children: List[BaseRetriever] = []
        specs: List[Dict[str, Any]] = []
        try:
            for child in requested:
                if isinstance(child, BaseRetriever):
                    runtime = child
                elif isinstance(child, str):
                    # Provider names use the provider's normal constructor
                    # defaults, then record the resulting full state.
                    runtime = Retriever.web_fetch(child)
                elif isinstance(child, Mapping):
                    spec = self._normalize_spec(child)
                    runtime = self._build_child(spec)
                    children.append(runtime)
                    specs.append(spec)
                    continue
                else:
                    raise TypeError(
                        "Fallback retrievers must be retriever instances, "
                        "provider names, or serialized retriever specs"
                    )

                try:
                    spec = self._serialize_child(runtime)
                except BaseException:
                    _close_children([runtime])
                    raise
                children.append(runtime)
                specs.append(spec)
        except BaseException:
            _close_children(children)
            raise
        return children, specs

    def _normalize_spec(self, spec: Mapping[str, Any]) -> Dict[str, Any]:
        normalized = dict(spec)
        provider = normalized.get("provider")
        retriever_type = normalized.get("retriever_type")
        state = normalized.get("state")

        if not isinstance(provider, str) or not provider.strip():
            raise ValueError("Serialized fallback retriever requires `provider`")
        if retriever_type != "web_fetch":
            raise ValueError("Fallback retrievers require `retriever_type='web_fetch'`")
        if not isinstance(state, Mapping) or not state:
            raise ValueError(
                "Serialized fallback retriever requires a non-empty mapping `state`"
            )

        normalized["provider"] = provider.strip()
        normalized["retriever_type"] = "web_fetch"
        normalized["state"] = dict(state)
        return normalized

    def _serialize_child(self, child: BaseRetriever) -> Dict[str, Any]:
        return self._normalize_spec(child.serialize())

    def _build_child(self, spec: Mapping[str, Any]) -> BaseRetriever:
        normalized = self._normalize_spec(spec)
        return Retriever.from_serialized(
            normalized["provider"],
            normalized["retriever_type"],
            normalized["state"],
        )

    def _initialize(self) -> None:
        specs = getattr(self, "retrievers", None)
        if not specs:
            raise ValueError("Fallback retriever requires at least one child")
        children: List[BaseRetriever] = []
        try:
            for spec in specs:
                children.append(self._build_child(spec))
        except BaseException:
            _close_children(children)
            raise
        self.children = children

    def _provider_name(self, child: BaseRetriever) -> str:
        return getattr(child, "provider", type(child).__name__)

    def _fetch_one(self, url: str) -> Dict[str, Dict[str, str]]:
        target_url = normalize_url(url)
        errors = []
        for child in self.children:
            try:
                entries = _response_entries(child(target_url))
            except Exception as exc:
                errors.append(f"{self._provider_name(child)}: {exc}")
                continue

            if entries:
                content = _entry_content(entries[0])
                if content and content.strip():
                    return entries[0]
                errors.append(f"{self._provider_name(child)}: empty content")
            else:
                errors.append(f"{self._provider_name(child)}: empty result")
        raise RuntimeError(f"Failed to fetch {target_url}: {' | '.join(errors)}")

    async def _afetch_one(self, url: str) -> Dict[str, Dict[str, str]]:
        target_url = normalize_url(url)
        errors = []
        for child in self.children:
            try:
                entries = _response_entries(await child.acall(target_url))
            except Exception as exc:
                errors.append(f"{self._provider_name(child)}: {exc}")
                continue

            if entries:
                content = _entry_content(entries[0])
                if content and content.strip():
                    return entries[0]
                errors.append(f"{self._provider_name(child)}: empty content")
            else:
                errors.append(f"{self._provider_name(child)}: empty result")
        raise RuntimeError(f"Failed to fetch {target_url}: {' | '.join(errors)}")

    def _fetch(self, urls: Union[str, List[str]]) -> List[Dict[str, Dict[str, str]]]:
        return [self._fetch_one(url) for url in urls]

    async def _afetch(
        self, urls: Union[str, List[str]]
    ) -> List[Dict[str, Dict[str, str]]]:
        results = []
        for url in urls:
            results.append(await self._afetch_one(url))
        return results

    def close(self) -> None:
        errors = []
        for child in self.children:
            close = getattr(child, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as exc:
                    errors.append(exc)
        if errors:
            raise ExceptionGroup("Fallback retriever cleanup failed", errors)

    async def aclose(self) -> None:
        errors = []
        for child in self.children:
            aclose = getattr(child, "aclose", None)
            if callable(aclose):
                try:
                    result = aclose()
                    if inspect.isawaitable(result):
                        await result
                except Exception as exc:
                    errors.append(exc)
                continue
            close = getattr(child, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as exc:
                    errors.append(exc)
        if errors:
            raise ExceptionGroup("Fallback retriever cleanup failed", errors)
