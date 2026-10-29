"""Direct HTTP page fetching with the shared HTML-to-text converter."""

import re
from typing import Any, Dict, List, Mapping, Optional, Union
from urllib.parse import urlsplit, urlunsplit

import httpx2

from msgflux.data.retrievers.base import BaseRetriever, BaseWebFetch
from msgflux.data.retrievers.http import HTTPRetrieverMixin
from msgflux.data.retrievers.registry import register_retriever
from msgflux.data.retrievers.types import WebFetchRetriever
from msgflux.utils.html import html_to_text

_MARKDOWN_CONTENT_TYPES = ("text/markdown", "text/x-markdown", "application/markdown")
_MARKDOWN_SUFFIXES = (".md", ".markdown")


def _validate_bare_host_port(candidate: str, scheme: str) -> None:
    # A bare hostname followed by a numeric port is not a URI scheme.
    # Nested URLs in the path/query/fragment must not select the scheme.
    if scheme:
        port_and_tail = candidate.partition(":")[2]
        if not re.fullmatch(r"[0-9]+(?:[/?#].*)?", port_and_tail, re.DOTALL):
            raise ValueError(
                f"Unsupported URL scheme `{scheme}`; only http and https are supported"
            )


def normalize_url(url: str) -> str:
    """Return an absolute HTTP(S) URL.

    Inputs without a scheme get ``https://``. Other schemes are rejected so a
    fetch cannot turn into a local-file or protocol-handler request.
    """
    if not isinstance(url, str):
        raise TypeError("`url` must be a string")
    candidate = url.strip()
    if not candidate:
        raise ValueError("`url` must not be empty")

    parsed = urlsplit(candidate)
    scheme = parsed.scheme.lower()
    if scheme in {"http", "https"}:
        if not candidate.lower().startswith(("http://", "https://")):
            raise ValueError(f"Invalid URL `{url}`: missing host")
    else:
        _validate_bare_host_port(candidate, parsed.scheme)
        if candidate.startswith("//"):
            candidate = f"https:{candidate}"
        elif candidate.startswith(("?", "#")):
            raise ValueError(f"Invalid URL `{url}`: missing host")
        else:
            candidate = f"https://{candidate}"
        parsed = urlsplit(candidate)
        scheme = parsed.scheme.lower()

    if not parsed.netloc:
        raise ValueError(f"Invalid URL `{url}`: missing host")
    try:
        parsed.port  # noqa: B018 - force urlsplit port validation
    except ValueError as exc:
        raise ValueError(f"Invalid URL `{url}`: {exc}") from exc

    return urlunsplit(
        (scheme, parsed.netloc, parsed.path, parsed.query, parsed.fragment)
    )


def build_fetch_entry(url: str, content: str) -> Dict[str, Dict[str, str]]:
    """Build the standard BaseWebFetch result entry for one URL."""
    return {"data": {"url": url, "content": content}}


def _is_markdown_response(response: Any) -> bool:
    content_type = response.headers.get("content-type", "").split(";", 1)[0]
    if content_type.strip().lower() in _MARKDOWN_CONTENT_TYPES:
        return True
    path = urlsplit(str(response.url)).path.lower()
    return path.endswith(_MARKDOWN_SUFFIXES)


def _error_message(url: str, exc: Exception) -> str:
    return f"Failed to fetch {url}: {exc}"


@register_retriever
class HttpWebFetchRetriever(
    BaseWebFetch, HTTPRetrieverMixin, BaseRetriever, WebFetchRetriever
):
    """Fetch a URL directly and convert HTML responses to readable text."""

    provider = "http"

    def __init__(
        self,
        *,
        default_headers: Optional[Mapping[str, str]] = None,
        timeout: Optional[float] = None,
    ):
        """Initialize the direct HTTP fetch retriever.

        Args:
            default_headers:
                Headers sent with every page request. Defaults to none.
            timeout:
                Request timeout in seconds. Defaults to 30.
        """
        self.default_headers = dict(default_headers or {})
        self.timeout = timeout or 30.0
        self._initialize()

    def _fetch_one(self, url: str) -> Dict[str, Dict[str, str]]:
        target_url = normalize_url(url)
        try:
            response = self.client.get(
                target_url,
                headers=self.default_headers,
            )
            response.raise_for_status()
        except httpx2.HTTPError as exc:
            raise RuntimeError(_error_message(url, exc)) from exc

        content = response.text
        if not _is_markdown_response(response):
            content = html_to_text(content)
        if not content.strip():
            raise RuntimeError(f"Failed to fetch {url}: empty response body")
        return build_fetch_entry(target_url, content)

    async def _afetch_one(self, url: str) -> Dict[str, Dict[str, str]]:
        target_url = normalize_url(url)
        try:
            response = await self.aclient.get(
                target_url,
                headers=self.default_headers,
            )
            response.raise_for_status()
        except httpx2.HTTPError as exc:
            raise RuntimeError(_error_message(url, exc)) from exc

        content = response.text
        if not _is_markdown_response(response):
            content = html_to_text(content)
        if not content.strip():
            raise RuntimeError(f"Failed to fetch {url}: empty response body")
        return build_fetch_entry(target_url, content)

    def _fetch(self, urls: Union[str, List[str]]) -> List[Dict[str, Dict[str, str]]]:
        return [self._fetch_one(url) for url in urls]

    async def _afetch(
        self, urls: Union[str, List[str]]
    ) -> List[Dict[str, Dict[str, str]]]:
        results = []
        for url in urls:
            results.append(await self._afetch_one(url))
        return results
