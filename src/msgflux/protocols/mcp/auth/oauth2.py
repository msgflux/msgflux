"""OAuth2 authentication for MCP."""

from typing import Dict, Optional, Callable, Awaitable, Any
from datetime import datetime
from msgflux.protocols.mcp.auth.base import BaseAuth


class OAuth2Auth(BaseAuth):
    """OAuth2 authentication with automatic token refresh.

    Manages OAuth2 access tokens and refresh tokens, automatically
    refreshing when expired.

    Example:
        ```python
        async def refresh_token_callback(refresh_token: str) -> dict:
            # Call your OAuth2 token endpoint
            response = await http_client.post(
                "https://auth.example.com/oauth/token",
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": refresh_token,
                    "client_id": "your-client-id",
                }
            )
            return {
                "access_token": response["access_token"],
                "refresh_token": response.get("refresh_token"),
                "expires_in": response["expires_in"],
            }

        auth = OAuth2Auth(
            access_token="initial-token",
            refresh_token="refresh-token",
            expires_in=3600,
            refresh_callback=refresh_token_callback
        )
        ```
    """

    def __init__(
        self,
        access_token: str,
        refresh_token: Optional[str] = None,
        expires_in: Optional[int] = None,
        refresh_callback: Optional[Callable[[str], Awaitable[Dict[str, Any]]]] = None,
        token_type: str = "Bearer",
    ):
        """Initialize OAuth2 authentication.

        Args:
            access_token: The OAuth2 access token.
            refresh_token: The refresh token (required for auto-refresh).
            expires_in: Token expiration time in seconds.
            refresh_callback: Async callback that takes refresh_token and returns
                            dict with 'access_token', 'refresh_token' (optional),
                            and 'expires_in'.
            token_type: Type of token (default: "Bearer").
        """
        super().__init__()
        self._access_token = access_token
        self._refresh_token = refresh_token
        self._token_type = token_type
        self._refresh_callback = refresh_callback

        if expires_in:
            self.set_expiration(expires_in)

    def apply_auth(self, headers: Dict[str, str]) -> Dict[str, str]:
        """Apply OAuth2 access token to Authorization header.

        Args:
            headers: Existing request headers.

        Returns:
            Headers with Authorization added.
        """
        headers = headers.copy()
        headers["Authorization"] = f"{self._token_type} {self._access_token}"
        return headers

    async def refresh(self) -> bool:
        """Refresh the OAuth2 tokens using the refresh token.

        Returns:
            True if refresh succeeded, False otherwise.
        """
        if not self._refresh_token or not self._refresh_callback:
            return False

        try:
            # Call refresh callback with current refresh token
            result = await self._refresh_callback(self._refresh_token)

            # Update tokens
            if "access_token" in result:
                self._access_token = result["access_token"]

                # Update refresh token if provided
                if "refresh_token" in result:
                    self._refresh_token = result["refresh_token"]

                # Update expiration
                if "expires_in" in result:
                    self.set_expiration(result["expires_in"])

                self._last_refresh = datetime.now()
                return True

        except Exception:
            pass

        return False

    def update_tokens(
        self,
        access_token: str,
        refresh_token: Optional[str] = None,
        expires_in: Optional[int] = None,
    ) -> None:
        """Update OAuth2 tokens.

        Args:
            access_token: New access token.
            refresh_token: New refresh token (optional).
            expires_in: New expiration time in seconds (optional).
        """
        self._access_token = access_token
        if refresh_token:
            self._refresh_token = refresh_token
        if expires_in:
            self.set_expiration(expires_in)
