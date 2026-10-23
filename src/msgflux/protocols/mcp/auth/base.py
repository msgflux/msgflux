"""Base authentication provider for MCP."""

import asyncio
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any, Dict, Optional


class BaseAuth(ABC):
    """Abstract base class for MCP authentication providers.

    All authentication providers must inherit from this class and implement
    the required methods for applying authentication to HTTP requests.
    """

    def __init__(self):
        """Initialize the base auth provider."""
        self._last_refresh: Optional[datetime] = None
        self._expires_at: Optional[datetime] = None
        self._refresh_lock = asyncio.Lock()

    @abstractmethod
    def apply_auth(self, headers: Dict[str, str]) -> Dict[str, str]:
        """Apply authentication to request headers.

        Args:
            headers: Existing request headers.

        Returns:
            Updated headers with authentication applied.
        """
        pass

    def is_expired(self) -> bool:
        """Check if the current authentication is expired.

        Returns:
            True if the expiration time has passed, False if the token is not
            expired or its expiration is unknown.
        """
        if self._expires_at is None:
            return False
        return datetime.now() >= self._expires_at  # noqa: DTZ005

    def set_expiration(self, expires_in: int) -> None:
        """Set token expiration time.

        Args:
            expires_in: Number of seconds until expiration.
        """
        self._expires_at = datetime.now() + timedelta(seconds=expires_in)  # noqa: DTZ005

    async def refresh_if_needed(self) -> bool:
        """Refresh authentication if expired.

        Callers that observe an expired credential serialize on a per-provider
        lock. Each caller re-checks expiration after acquiring the lock, so a
        caller that waited through a successful refresh with a future deadline
        returns False without refreshing again. If the refresh fails or the
        refreshed credential is already expired, each waiting caller attempts
        the refresh in turn; failures are not cached.

        Returns:
            True if the refresh succeeded, False if no refresh was needed, the
            provider does not support refresh, or the refresh failed.
        """
        if not self.is_expired():
            return False
        async with self._refresh_lock:
            if not self.is_expired():
                return False
            return await self.refresh()

    async def refresh(self) -> bool:
        """Refresh the authentication credentials.

        Override this method in subclasses that support credential refresh.

        Returns:
            True if refresh succeeded, False otherwise.
        """
        return False

    def get_auth_info(self) -> Dict[str, Any]:
        """Get information about current authentication state.

        Returns:
            Dictionary with auth state information.
        """
        return {
            "type": self.__class__.__name__,
            "expired": self.is_expired(),
            "expires_at": self._expires_at.isoformat() if self._expires_at else None,
            "last_refresh": self._last_refresh.isoformat()
            if self._last_refresh
            else None,
        }
