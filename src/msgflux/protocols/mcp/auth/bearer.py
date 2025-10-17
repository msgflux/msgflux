"""Bearer token authentication for MCP."""

from typing import Dict, Optional, Callable, Awaitable
from .base import BaseAuth
from datetime import datetime


class BearerTokenAuth(BaseAuth):
    """Bearer token authentication (JWT, etc).

    This auth provider adds an Authorization header with a Bearer token.
    Supports automatic token refresh via callback.

    Example:
        ```python
        auth = BearerTokenAuth(token="your-jwt-token")

        # With auto-refresh
        async def get_new_token():
            # Your token refresh logic
            return "new-token"

        auth = BearerTokenAuth(
            token="initial-token",
            expires_in=3600,
            refresh_callback=get_new_token
        )
        ```
    """

    def __init__(
        self,
        token: str,
        expires_in: Optional[int] = None,
        refresh_callback: Optional[Callable[[], Awaitable[str]]] = None,
        token_type: str = "Bearer",
    ):
        """Initialize bearer token authentication.

        Args:
            token: The bearer token.
            expires_in: Token expiration time in seconds (optional).
            refresh_callback: Async callback to get a new token when expired.
            token_type: Type of token (default: "Bearer").
        """
        super().__init__()
        self._token = token
        self._token_type = token_type
        self._refresh_callback = refresh_callback

        if expires_in:
            self.set_expiration(expires_in)

    def apply_auth(self, headers: Dict[str, str]) -> Dict[str, str]:
        """Apply bearer token to Authorization header.

        Args:
            headers: Existing request headers.

        Returns:
            Headers with Authorization added.
        """
        headers = headers.copy()
        headers["Authorization"] = f"{self._token_type} {self._token}"
        return headers

    async def refresh(self) -> bool:
        """Refresh the bearer token using the callback.

        Returns:
            True if refresh succeeded, False otherwise.
        """
        if not self._refresh_callback:
            return False

        try:
            new_token = await self._refresh_callback()
            if new_token:
                self._token = new_token
                self._last_refresh = datetime.now()
                return True
        except Exception:
            pass

        return False

    def update_token(self, token: str, expires_in: Optional[int] = None) -> None:
        """Update the bearer token.

        Args:
            token: New bearer token.
            expires_in: New expiration time in seconds (optional).
        """
        self._token = token
        if expires_in:
            self.set_expiration(expires_in)
