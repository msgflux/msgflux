"""markdown.new fetch retriever."""

from typing import Any, Dict, List, Mapping, Optional, Union

import httpx2

from msgflux.data.retrievers.base import BaseRetriever, BaseWebFetch
from msgflux.data.retrievers.http import HTTPRetrieverMixin
from msgflux.data.retrievers.providers.http import build_fetch_entry, normalize_url
from msgflux.data.retrievers.registry import register_retriever
from msgflux.data.retrievers.types import WebFetchRetriever

DEFAULT_MARKDOWN_ENDPOINT = "https://markdown.new/"
MARKDOWN_METHODS = frozenset({"auto", "ai", "browser"})


@register_retriever
class MarkdownWebFetchRetriever(
    BaseWebFetch, HTTPRetrieverMixin, BaseRetriever, WebFetchRetriever
):
    """Fetch a page as Markdown using the markdown.new JSON endpoint.

    markdown.new is keyless. It accepts a target URL in a JSON body and
    supports ``method="auto"``, ``"ai"``, and ``"browser"``.
    """

    provider = "markdown"

    def __init__(
        self,
        *,
        endpoint: str = DEFAULT_MARKDOWN_ENDPOINT,
        method: str = "auto",
        retain_images: Optional[bool] = None,
        default_headers: Optional[Mapping[str, str]] = None,
        timeout: Optional[float] = None,
    ):
        """Initialize the markdown.new retriever.

        Args:
            endpoint:
                markdown.new-compatible endpoint URL. Defaults to
                ``https://markdown.new/``.
            method:
                Conversion method: ``auto``, ``ai``, or ``browser``.
            retain_images:
                When true, ask markdown.new to keep image references.
            default_headers:
                Additional headers sent to the endpoint.
            timeout:
                Request timeout in seconds. Defaults to 30.
        """
        method = (method or "auto").strip().lower()
        if method not in MARKDOWN_METHODS:
            raise ValueError("`method` must be one of: auto, ai, browser")

        self.endpoint = normalize_url(endpoint)
        self.method = method
        self.retain_images = bool(retain_images)
        self.default_headers = dict(default_headers or {})
        self.timeout = timeout or 30.0
        self._initialize()

    def _build_headers(self) -> Dict[str, str]:
        headers = {
            "Accept": "text/markdown, text/plain;q=0.9, */*;q=0.8",
            "Content-Type": "application/json",
        }
        headers.update(self.default_headers)
        return headers

    def _build_payload(self, url: str) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "url": normalize_url(url),
            "method": self.method,
        }
        if self.retain_images:
            payload["retain_images"] = True
        return payload

    def _parse_response(self, url: str, response: httpx2.Response) -> Dict[str, Any]:
        content = response.text
        if not content.strip():
            raise RuntimeError(f"markdown.new fetch failed for {url}: empty response")
        return build_fetch_entry(normalize_url(url), content)

    def _fetch_one(self, url: str) -> Dict[str, Dict[str, str]]:
        try:
            response = self.client.post(
                self.endpoint,
                headers=self._build_headers(),
                json=self._build_payload(url),
            )
            response.raise_for_status()
        except httpx2.HTTPError as exc:
            raise RuntimeError(f"markdown.new fetch failed for {url}: {exc}") from exc

        return self._parse_response(url, response)

    async def _afetch_one(self, url: str) -> Dict[str, Dict[str, str]]:
        try:
            response = await self.aclient.post(
                self.endpoint,
                headers=self._build_headers(),
                json=self._build_payload(url),
            )
            response.raise_for_status()
        except httpx2.HTTPError as exc:
            raise RuntimeError(f"markdown.new fetch failed for {url}: {exc}") from exc

        return self._parse_response(url, response)

    def _fetch(self, urls: Union[str, List[str]]) -> List[Dict[str, Dict[str, str]]]:
        return [self._fetch_one(url) for url in urls]

    async def _afetch(
        self, urls: Union[str, List[str]]
    ) -> List[Dict[str, Dict[str, str]]]:
        results = []
        for url in urls:
            results.append(await self._afetch_one(url))
        return results
