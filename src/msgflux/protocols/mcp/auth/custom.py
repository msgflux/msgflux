"""Custom header authentication for MCP."""

from typing import Dict, Callable, Optional
from .base import BaseAuth


class CustomHeaderAuth(BaseAuth):
    """Custom header-based authentication.

    Allows arbitrary headers to be added for authentication.
    Useful for proprietary auth schemes or multiple headers.

    Example:
        ```python
        # Static headers
        auth = CustomHeaderAuth({
            "X-Custom-Auth": "secret-value",
            "X-Request-ID": "unique-id"
        })

        # Dynamic headers via callback
        def get_headers():
            return {
                "X-Signature": compute_signature(),
                "X-Timestamp": str(int(time.time()))
            }

        auth = CustomHeaderAuth(headers_callback=get_headers)
        ```
    """

    def __init__(
        self,
        headers: Optional[Dict[str, str]] = None,
        headers_callback: Optional[Callable[[], Dict[str, str]]] = None,
    ):
        """Initialize custom header authentication.

        Args:
            headers: Static headers to add (optional).
            headers_callback: Callback that returns headers dynamically (optional).
                            Takes precedence over static headers if both provided.
        """
        super().__init__()
        self._headers = headers or {}
        self._headers_callback = headers_callback

    def apply_auth(self, headers: Dict[str, str]) -> Dict[str, str]:
        """Apply custom headers to request.

        Args:
            headers: Existing request headers.

        Returns:
            Headers with custom auth headers added.
        """
        headers = headers.copy()

        # Use callback if provided, otherwise use static headers
        if self._headers_callback:
            auth_headers = self._headers_callback()
        else:
            auth_headers = self._headers

        # Merge auth headers
        headers.update(auth_headers)
        return headers

    def update_headers(self, headers: Dict[str, str]) -> None:
        """Update static headers.

        Args:
            headers: New headers to use.
        """
        self._headers = headers
