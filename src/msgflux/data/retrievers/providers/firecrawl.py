"""Firecrawl v2 search and scrape retrievers."""

import asyncio
from os import getenv
from typing import Any, Dict, List, Mapping, Optional, Union

import httpx2

from msgflux.core.dotdict import dotdict
from msgflux.data.retrievers.base import BaseRetriever, BaseWebFetch, BaseWebSearch
from msgflux.data.retrievers.http import HTTPRetrieverMixin
from msgflux.data.retrievers.providers.http import build_fetch_entry, normalize_url
from msgflux.data.retrievers.registry import register_retriever
from msgflux.data.retrievers.types import WebFetchRetriever, WebSearchRetriever

FIRECRAWL_API_KEY_ENV_NAME = "FIRECRAWL_API_KEY"
FIRECRAWL_API_URL = "https://api.firecrawl.dev/v2"


def _response_detail(response: Optional[httpx2.Response]) -> str:
    if response is None:
        return ""
    try:
        payload = response.json()
    except ValueError:
        return response.text.strip()[:300]
    if isinstance(payload, Mapping):
        for key in ("error", "message", "code"):
            if payload.get(key):
                return str(payload[key])
        errors = payload.get("errors")
        if errors:
            return str(errors)
    return response.text.strip()[:300]


def _http_failure_detail(exc: httpx2.HTTPError) -> str:
    response = getattr(exc, "response", None)
    if response is None:
        return str(exc)
    detail = f"HTTP {response.status_code}"
    body = _response_detail(response)
    return f"{detail}: {body}" if body else detail


def _unsuccessful_detail(payload: Mapping[str, Any]) -> str:
    for key in ("error", "message", "code"):
        if payload.get(key):
            return str(payload[key])
    if payload.get("errors"):
        return str(payload["errors"])
    return "response did not report success"


class _FirecrawlRequestMixin:
    """Shared Firecrawl v2 request and response handling."""

    def _init_firecrawl(self, *, base_url: str, timeout: Optional[float]) -> None:
        self.base_url = normalize_url(base_url).rstrip("/")
        self.timeout = timeout or 30.0
        self._initialize()

    def _build_headers(self) -> Dict[str, str]:
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        api_key = (getenv(FIRECRAWL_API_KEY_ENV_NAME) or "").strip()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def _parse_payload(
        self,
        response: httpx2.Response,
        *,
        action: str,
        target: str,
    ) -> Mapping[str, Any]:
        try:
            payload = response.json()
        except ValueError as exc:
            raise RuntimeError(
                f"Firecrawl {action} failed for {target}: invalid JSON response"
            ) from exc
        if not isinstance(payload, Mapping):
            raise RuntimeError(
                f"Firecrawl {action} failed for {target}: invalid response body"
            )
        if payload.get("success") is not True:
            detail = _unsuccessful_detail(payload)
            raise RuntimeError(f"Firecrawl {action} failed for {target}: {detail}")
        return payload

    def _post_json(
        self,
        path: str,
        payload: Mapping[str, Any],
        *,
        action: str,
        target: str,
    ) -> Mapping[str, Any]:
        try:
            response = self.client.post(
                f"{self.base_url}{path}",
                headers=self._build_headers(),
                json=dict(payload),
            )
            response.raise_for_status()
        except httpx2.HTTPError as exc:
            detail = _http_failure_detail(exc)
            raise RuntimeError(
                f"Firecrawl {action} failed for {target}: {detail}"
            ) from exc
        return self._parse_payload(response, action=action, target=target)

    async def _apost_json(
        self,
        path: str,
        payload: Mapping[str, Any],
        *,
        action: str,
        target: str,
    ) -> Mapping[str, Any]:
        try:
            response = await self.aclient.post(
                f"{self.base_url}{path}",
                headers=self._build_headers(),
                json=dict(payload),
            )
            response.raise_for_status()
        except httpx2.HTTPError as exc:
            detail = _http_failure_detail(exc)
            raise RuntimeError(
                f"Firecrawl {action} failed for {target}: {detail}"
            ) from exc
        return self._parse_payload(response, action=action, target=target)


@register_retriever
class FirecrawlWebFetchRetriever(
    BaseWebFetch,
    _FirecrawlRequestMixin,
    HTTPRetrieverMixin,
    BaseRetriever,
    WebFetchRetriever,
):
    """Scrape a page as Markdown with Firecrawl v2.

    Firecrawl is keyless by default. When ``FIRECRAWL_API_KEY`` is set at
    request time, it is sent as a Bearer token.
    """

    provider = "firecrawl"

    def __init__(
        self,
        *,
        base_url: str = FIRECRAWL_API_URL,
        timeout: Optional[float] = None,
    ):
        """Initialize the Firecrawl fetch retriever.

        Args:
            base_url:
                Firecrawl v2 API root. Defaults to
                ``https://api.firecrawl.dev/v2``.
            timeout:
                Request timeout in seconds. Defaults to 30.
        """
        self._init_firecrawl(base_url=base_url, timeout=timeout)

    def _build_payload(self, url: str) -> Dict[str, object]:
        return {"url": normalize_url(url), "formats": ["markdown"]}

    def _parse_scrape(
        self, url: str, payload: Mapping[str, Any]
    ) -> Dict[str, Dict[str, str]]:
        data = payload.get("data")
        if not isinstance(data, Mapping):
            raise RuntimeError(
                f"Firecrawl scrape failed for {url}: missing response data"
            )
        if data.get("error"):
            raise RuntimeError(f"Firecrawl scrape failed for {url}: {data['error']}")
        markdown = data.get("markdown")
        if not isinstance(markdown, str) or not markdown.strip():
            raise RuntimeError(
                f"Firecrawl scrape failed for {url}: empty markdown response"
            )
        return build_fetch_entry(normalize_url(url), markdown)

    def _fetch_one(self, url: str) -> Dict[str, Dict[str, str]]:
        payload = self._post_json(
            "/scrape",
            self._build_payload(url),
            action="scrape",
            target=url,
        )
        return self._parse_scrape(url, payload)

    async def _afetch_one(self, url: str) -> Dict[str, Dict[str, str]]:
        payload = await self._apost_json(
            "/scrape",
            self._build_payload(url),
            action="scrape",
            target=url,
        )
        return self._parse_scrape(url, payload)

    def _fetch(self, urls: Union[str, List[str]]) -> List[Dict[str, Dict[str, str]]]:
        return [self._fetch_one(url) for url in urls]

    async def _afetch(
        self, urls: Union[str, List[str]]
    ) -> List[Dict[str, Dict[str, str]]]:
        results = []
        for url in urls:
            results.append(await self._afetch_one(url))
        return results


@register_retriever
class FirecrawlWebRetriever(
    BaseWebSearch,
    _FirecrawlRequestMixin,
    HTTPRetrieverMixin,
    BaseRetriever,
    WebSearchRetriever,
):
    """Search the web with Firecrawl v2 and scrape result Markdown."""

    provider = "firecrawl"

    def __init__(
        self,
        *,
        base_url: str = FIRECRAWL_API_URL,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        tbs: Optional[str] = None,
        location: Optional[str] = None,
        country: Optional[str] = None,
        categories: Optional[List[str]] = None,
        timeout: Optional[float] = None,
    ):
        """Initialize the Firecrawl search retriever.

        Args:
            base_url:
                Firecrawl v2 API root. Defaults to
                ``https://api.firecrawl.dev/v2``.
            include_domains:
                Restrict results to these hostnames.
            exclude_domains:
                Exclude results from these hostnames.
            tbs:
                Firecrawl time-based search value, such as ``qdr:w``.
            location:
                Search location string.
            country:
                ISO country code for search results.
            categories:
                Search categories such as ``research`` or ``pdf``.
            timeout:
                Request timeout in seconds. Defaults to 30.
        """
        self.include_domains = list(include_domains or [])
        self.exclude_domains = list(exclude_domains or [])
        self.tbs = tbs
        self.location = location
        self.country = country
        self.categories = list(categories or [])
        self._init_firecrawl(base_url=base_url, timeout=timeout)

    def _build_payload(self, query: str, top_k: int) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "query": query,
            "limit": top_k,
            "scrapeOptions": {"formats": [{"type": "markdown"}]},
        }
        if self.include_domains:
            payload["includeDomains"] = self.include_domains
        if self.exclude_domains:
            payload["excludeDomains"] = self.exclude_domains
        if self.tbs:
            payload["tbs"] = self.tbs
        if self.location:
            payload["location"] = self.location
        if self.country:
            payload["country"] = self.country
        if self.categories:
            payload["categories"] = self.categories
        return payload

    def _parse_results(
        self, payload: Mapping[str, Any], top_k: int
    ) -> List[Dict[str, Dict[str, Any]]]:
        data = payload.get("data")
        if not isinstance(data, Mapping):
            raise RuntimeError("Firecrawl search failed: missing response data")
        web_results = data.get("web") or []
        if not isinstance(web_results, list):
            raise RuntimeError("Firecrawl search failed: invalid web results")

        results = []
        for result in web_results[:top_k]:
            if not isinstance(result, Mapping):
                continue
            content = result.get("markdown")
            if content is None:
                content = result.get("description")
            results.append(
                {
                    "data": {
                        "title": result.get("title"),
                        "content": content,
                        "url": result.get("url"),
                    }
                }
            )
        return results

    def _single_search(self, query: str, top_k: int) -> List[Dict[str, Dict[str, Any]]]:
        payload = self._post_json(
            "/search",
            self._build_payload(query, top_k),
            action="search",
            target=f"query '{query}'",
        )
        return self._parse_results(payload, top_k)

    async def _asingle_search(
        self, query: str, top_k: int
    ) -> List[Dict[str, Dict[str, Any]]]:
        payload = await self._apost_json(
            "/search",
            self._build_payload(query, top_k),
            action="search",
            target=f"query '{query}'",
        )
        return self._parse_results(payload, top_k)

    def _search(self, queries: List[str], top_k: int) -> List[dotdict]:
        results = []
        for query in queries:
            query_results = self._single_search(query, top_k)
            results.append(dotdict({"results": query_results}))
        return results

    async def _asearch(self, queries: List[str], top_k: int) -> List[dotdict]:
        tasks = [self._asingle_search(query, top_k) for query in queries]
        query_results = await asyncio.gather(*tasks)
        results = []
        for result in query_results:
            results.append(dotdict({"results": result}))
        return results
