"""API Key authentication for MCP."""

from typing import Dict
from msgflux.protocols.mcp.auth.base import BaseAuth


class APIKeyAuth(BaseAuth):
    """API Key authentication.

    Supports API keys in custom headers or as query parameters.

    Example:
        ```python
        # API key in header
        auth = APIKeyAuth(api_key="your-api-key", header_name="X-API-Key")

        # API key in header with prefix
        auth = APIKeyAuth(
            api_key="your-key",
            header_name="Authorization",
            key_prefix="ApiKey"
        )
        ```
    """

    def __init__(
        self,
        api_key: str,
        header_name: str = "X-API-Key",
        key_prefix: str = "",
    ):
        """Initialize API key authentication.

        Args:
            api_key: The API key.
            header_name: Name of the header to use (default: "X-API-Key").
            key_prefix: Optional prefix for the key value (e.g., "ApiKey ").
        """
        super().__init__()
        self._api_key = api_key
        self._header_name = header_name
        self._key_prefix = key_prefix

    def apply_auth(self, headers: Dict[str, str]) -> Dict[str, str]:
        """Apply API key to request headers.

        Args:
            headers: Existing request headers.

        Returns:
            Headers with API key added.
        """
        headers = headers.copy()

        if self._key_prefix:
            value = f"{self._key_prefix} {self._api_key}"
        else:
            value = self._api_key

        headers[self._header_name] = value
        return headers

    def update_api_key(self, api_key: str) -> None:
        """Update the API key.

        Args:
            api_key: New API key.
        """
        self._api_key = api_key
