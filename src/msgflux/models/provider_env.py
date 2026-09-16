"""Environment-driven endpoint configuration for model providers.

Subclasses declare class variables only; the shared `_get_*` methods
resolve them. Providers that work without a real key (local servers)
set `api_key_default` instead of requiring the variable.
"""

from os import getenv
from typing import Optional


class ProviderEnvBase:
    """Configuration mixin for provider base URL and API key."""

    provider: str = ""
    display_name: str = ""
    api_key_env: str = ""
    api_key_default: Optional[str] = None
    base_url_env: Optional[str] = None
    base_url: Optional[str] = None

    def _get_base_url(self):
        """Load base URL from environment variable."""
        if self.base_url_env is None:
            return self.base_url
        base_url = getenv(self.base_url_env, self.base_url)
        if base_url is None:
            raise ValueError(f"Please set `{self.base_url_env}`")
        return base_url

    def _get_api_key(self):
        """Load API keys from environment variable."""
        key = getenv(self.api_key_env, self.api_key_default)
        if not key:
            raise ValueError(
                f"The {self.display_name} key is not available. "
                f"Please set `{self.api_key_env}`"
            )
        return key
