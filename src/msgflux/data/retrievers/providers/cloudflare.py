"""Cloudflare Browser Run Markdown endpoint retriever."""

from os import getenv
from typing import Any, Dict, List, Mapping, Optional, Union
from urllib.parse import quote

import httpx2

from msgflux.data.retrievers.base import BaseRetriever, BaseWebFetch
from msgflux.data.retrievers.http import HTTPRetrieverMixin
from msgflux.data.retrievers.providers.http import build_fetch_entry, normalize_url
from msgflux.data.retrievers.registry import register_retriever
from msgflux.data.retrievers.types import WebFetchRetriever

CLOUDFLARE_API_BASE = "https://api.cloudflare.com/client/v4"
CLOUDFLARE_ACCOUNT_ID_ENV_NAME = "CLOUDFLARE_ACCOUNT_ID"
CLOUDFLARE_API_TOKEN_ENV_NAME = "CLOUDFLARE_API_TOKEN"  # noqa: S105 - env var name


def _format_error_item(item: Any) -> str:
    if isinstance(item, Mapping):
        detail = item.get("message") or item.get("code")
        return str(detail) if detail else ""
    return str(item) if item else ""


def _payload_detail(payload: Any) -> str:
    if not isinstance(payload, Mapping):
        return ""
    errors = payload.get("errors")
    if errors:
        if isinstance(errors, list):
            parts = [_format_error_item(item) for item in errors]
            detail = "; ".join(part for part in parts if part)
            if detail:
                return detail
        return str(errors)
    for key in ("messages", "error", "message"):
        value = payload.get(key)
        if value:
            return str(value)
    return ""


def _http_failure_detail(exc: httpx2.HTTPError) -> str:
    response = getattr(exc, "response", None)
    if response is None:
        return str(exc)
    detail = f"HTTP {response.status_code}"
    try:
        payload = response.json()
    except ValueError:
        payload = None
    body = _payload_detail(payload) or response.text.strip()[:300]
    return f"{detail}: {body}" if body else detail


@register_retriever
class CloudflareWebFetchRetriever(
    BaseWebFetch, HTTPRetrieverMixin, BaseRetriever, WebFetchRetriever
):
    """Convert a page to Markdown with Cloudflare Browser Run.

    The REST endpoint requires an account ID and a Browser Run API token. Both
    are resolved at request time from configuration or environment variables;
    the token is never stored on the retriever instance.
    """

    provider = "cloudflare"

    def __init__(
        self,
        account_id: Optional[str] = None,
        *,
        account_id_env: str = CLOUDFLARE_ACCOUNT_ID_ENV_NAME,
        api_token_env: str = CLOUDFLARE_API_TOKEN_ENV_NAME,
        base_url: str = CLOUDFLARE_API_BASE,
        timeout: Optional[float] = None,
    ):
        """Initialize the Cloudflare fetch retriever.

        Args:
            account_id:
                Cloudflare account ID. When omitted, ``CLOUDFLARE_ACCOUNT_ID``
                is read at request time.
            account_id_env:
                Environment variable used when ``account_id`` is not set.
            api_token_env:
                Environment variable holding the Browser Run API token.
            base_url:
                Cloudflare API root. Defaults to
                ``https://api.cloudflare.com/client/v4``.
            timeout:
                Request timeout in seconds. Defaults to 30.
        """
        if not account_id_env or not account_id_env.strip():
            raise ValueError("`account_id_env` must be a non-empty string")
        if not api_token_env or not api_token_env.strip():
            raise ValueError("`api_token_env` must be a non-empty string")

        self.account_id = account_id
        self.account_id_env = account_id_env
        self.api_token_env = api_token_env
        self.base_url = normalize_url(base_url).rstrip("/")
        self.timeout = timeout or 30.0
        self._initialize()

    def _get_account_id(self) -> str:
        account_id = self.account_id or getenv(self.account_id_env)
        if not account_id or not str(account_id).strip():
            raise ValueError(
                "The Cloudflare account ID is not available. Please set "
                f"`{self.account_id_env}` or pass `account_id`"
            )
        return quote(str(account_id).strip(), safe="")

    def _get_api_token(self) -> str:
        api_token = (getenv(self.api_token_env) or "").strip()
        if not api_token:
            raise ValueError(
                "The Cloudflare API token is not available. Please set "
                f"`{self.api_token_env}`"
            )
        return api_token

    def _build_url(self) -> str:
        return f"{self.base_url}/accounts/{self._get_account_id()}/browser-run/markdown"

    def _build_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self._get_api_token()}",
            "Content-Type": "application/json",
        }

    def _parse_payload(
        self, url: str, payload: Mapping[str, Any]
    ) -> Dict[str, Dict[str, str]]:
        if payload.get("success") is not True:
            detail = _payload_detail(payload) or "response did not report success"
            raise RuntimeError(f"Cloudflare markdown fetch failed for {url}: {detail}")

        result = payload.get("result")
        if not isinstance(result, str) or not result.strip():
            raise RuntimeError(
                f"Cloudflare markdown fetch failed for {url}: empty response"
            )
        return build_fetch_entry(normalize_url(url), result)

    def _fetch_one(self, url: str) -> Dict[str, Dict[str, str]]:
        target_url = normalize_url(url)
        try:
            response = self.client.post(
                self._build_url(),
                headers=self._build_headers(),
                json={"url": target_url},
            )
            response.raise_for_status()
        except httpx2.HTTPError as exc:
            detail = _http_failure_detail(exc)
            raise RuntimeError(
                f"Cloudflare markdown fetch failed for {url}: {detail}"
            ) from exc

        try:
            payload = response.json()
        except ValueError as exc:
            raise RuntimeError(
                f"Cloudflare markdown fetch failed for {url}: invalid JSON response"
            ) from exc
        if not isinstance(payload, Mapping):
            raise RuntimeError(
                f"Cloudflare markdown fetch failed for {url}: invalid response body"
            )
        return self._parse_payload(url, payload)

    async def _afetch_one(self, url: str) -> Dict[str, Dict[str, str]]:
        target_url = normalize_url(url)
        try:
            response = await self.aclient.post(
                self._build_url(),
                headers=self._build_headers(),
                json={"url": target_url},
            )
            response.raise_for_status()
        except httpx2.HTTPError as exc:
            detail = _http_failure_detail(exc)
            raise RuntimeError(
                f"Cloudflare markdown fetch failed for {url}: {detail}"
            ) from exc

        try:
            payload = response.json()
        except ValueError as exc:
            raise RuntimeError(
                f"Cloudflare markdown fetch failed for {url}: invalid JSON response"
            ) from exc
        if not isinstance(payload, Mapping):
            raise RuntimeError(
                f"Cloudflare markdown fetch failed for {url}: invalid response body"
            )
        return self._parse_payload(url, payload)

    def _fetch(self, urls: Union[str, List[str]]) -> List[Dict[str, Dict[str, str]]]:
        return [self._fetch_one(url) for url in urls]

    async def _afetch(
        self, urls: Union[str, List[str]]
    ) -> List[Dict[str, Dict[str, str]]]:
        results = []
        for url in urls:
            results.append(await self._afetch_one(url))
        return results
