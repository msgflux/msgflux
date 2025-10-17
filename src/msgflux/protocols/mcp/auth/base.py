"""Base authentication provider for MCP."""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Any
from datetime import datetime, timedelta


class BaseAuth(ABC):
    """Abstract base class for MCP authentication providers.

    All authentication providers must inherit from this class and implement
    the required methods for applying authentication to HTTP requests.
    """

    def __init__(self):
        """Initialize the base auth provider."""
        self._last_refresh: Optional[datetime] = None
        self._expires_at: Optional[datetime] = None

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
            True if expired or expiration unknown, False otherwise.
        """
        if self._expires_at is None:
            return False
        return datetime.now() >= self._expires_at

    def set_expiration(self, expires_in: int) -> None:
        """Set token expiration time.

        Args:
            expires_in: Number of seconds until expiration.
        """
        self._expires_at = datetime.now() + timedelta(seconds=expires_in)

    async def refresh_if_needed(self) -> bool:
        """Refresh authentication if expired.

        Returns:
            True if refreshed, False if not needed or not supported.
        """
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
            "last_refresh": self._last_refresh.isoformat() if self._last_refresh else None,
        }
