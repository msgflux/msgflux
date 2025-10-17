"""Basic HTTP authentication for MCP."""

import base64
from typing import Dict
from .base import BaseAuth


class BasicAuth(BaseAuth):
    """HTTP Basic Authentication.

    Encodes username and password in base64 for the Authorization header.

    Example:
        ```python
        auth = BasicAuth(username="user", password="pass")
        ```
    """

    def __init__(self, username: str, password: str):
        """Initialize basic authentication.

        Args:
            username: Username for authentication.
            password: Password for authentication.
        """
        super().__init__()
        self._username = username
        self._password = password

    def apply_auth(self, headers: Dict[str, str]) -> Dict[str, str]:
        """Apply basic authentication to Authorization header.

        Args:
            headers: Existing request headers.

        Returns:
            Headers with Authorization added.
        """
        headers = headers.copy()

        # Encode credentials
        credentials = f"{self._username}:{self._password}"
        encoded = base64.b64encode(credentials.encode()).decode()

        headers["Authorization"] = f"Basic {encoded}"
        return headers

    def update_credentials(self, username: str, password: str) -> None:
        """Update authentication credentials.

        Args:
            username: New username.
            password: New password.
        """
        self._username = username
        self._password = password
