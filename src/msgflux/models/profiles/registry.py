"""Profile registry with lazy loading and background refresh.

Provides singleton access to model profiles with automatic caching.
"""

import threading
from typing import Dict, Optional

from msgflux.models.profiles.base import ModelProfile, ProviderProfile
from msgflux.models.profiles.loader import ProfileLoader


class ProfileRegistry:
    """Singleton registry for model profiles.

    Features:
    - Lazy loading on first access
    - Background refresh when cache expires
    - Thread-safe operations
    """

    _instance = None
    _profiles: Optional[Dict[str, ProviderProfile]] = None
    _loading: bool = False
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_instance(cls) -> "ProfileRegistry":
        """Get singleton instance.

        Returns:
            ProfileRegistry singleton
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def get_model_profile(
        self,
        model_id: str,
        provider_id: Optional[str] = None,
    ) -> Optional[ModelProfile]:
        """Get model profile by model_id and optionally provider_id.

        Auto-refreshes cache if expired (in background).

        Args:
            model_id: Model identifier (e.g., "gpt-4o", "llama-3-70b")
            provider_id: Provider identifier (e.g., "openai", "ollama")
                        If None, searches across all providers
                        Special case: vllm/ollama fallback to openrouter

        Returns:
            ModelProfile if found, None otherwise
        """
        # Lazy load on first access
        if self._profiles is None:
            self._load_profiles(background=True)

        # Check if cache is still valid
        if not ProfileLoader.is_cache_valid() and not self._loading:
            # Cache expired, refresh in background
            self._load_profiles(background=True)

        # Return from current cache (may be stale but still usable)
        if self._profiles is None:
            return None

        if provider_id:
            # Special case: vllm/ollama use openrouter data (open-source models)
            if provider_id in ("vllm", "ollama"):
                provider = self._profiles.get("openrouter")
                if provider:
                    return provider.models.get(model_id)
                return None

            # Direct lookup with provider
            provider = self._profiles.get(provider_id)
            if provider:
                return provider.models.get(model_id)
            return None

        # Search across all providers
        for provider in self._profiles.values():
            if model_id in provider.models:
                return provider.models[model_id]

        return None

    def get_provider_profile(self, provider_id: str) -> Optional[ProviderProfile]:
        """Get provider profile by provider_id.

        Args:
            provider_id: Provider identifier (e.g., "openai")

        Returns:
            ProviderProfile if found, None otherwise
        """
        # Lazy load on first access
        if self._profiles is None:
            self._load_profiles(background=True)

        # Check if cache is still valid
        if not ProfileLoader.is_cache_valid() and not self._loading:
            # Cache expired, refresh in background
            self._load_profiles(background=True)

        # Return from current cache
        if self._profiles is None:
            return None

        return self._profiles.get(provider_id)

    def ensure_loaded(self, background: bool = True):
        """Ensure profiles are loaded.

        Args:
            background: If True, load in background. If False, block until loaded.
        """
        if self._profiles is None:
            self._load_profiles(background=background)

    def _load_profiles(self, background: bool = True):
        """Load profiles from cache or fetch from API.

        Args:
            background: If True, fetch in background. If False, block.
        """
        with self._lock:
            if self._loading:
                return  # Already loading

            # Try cache first (fast)
            cached = ProfileLoader.load_from_cache()
            if cached:
                self._profiles = cached
                return

            # Cache miss/expired - need to fetch
            if background:
                self._loading = True
                from msgflux.nn.functional import background_task

                background_task(self._fetch_and_cache)
            else:
                self._fetch_and_cache()

    def _fetch_and_cache(self):
        """Fetch from API and save to cache."""
        try:
            profiles = ProfileLoader.fetch_profiles_sync()
            with self._lock:
                self._profiles = profiles
                ProfileLoader.save_to_cache(profiles)
        except Exception as e:
            # Log error but don't crash
            from msgflux.logger import logger

            logger.error(f"Failed to fetch model profiles: {e}")
        finally:
            self._loading = False


# Module-level singleton instance
_registry = ProfileRegistry.get_instance()


def get_model_profile(
    model_id: str,
    provider_id: Optional[str] = None,
) -> Optional[ModelProfile]:
    """Get model profile (module-level convenience function).

    Args:
        model_id: Model identifier
        provider_id: Optional provider identifier

    Returns:
        ModelProfile if found, None otherwise
    """
    return _registry.get_model_profile(model_id, provider_id)


def get_provider_profile(provider_id: str) -> Optional[ProviderProfile]:
    """Get provider profile (module-level convenience function).

    Args:
        provider_id: Provider identifier

    Returns:
        ProviderProfile if found, None otherwise
    """
    return _registry.get_provider_profile(provider_id)


def ensure_profiles_loaded(background: bool = True):
    """Ensure profiles are loaded (module-level convenience function).

    Args:
        background: If True, load in background
    """
    _registry.ensure_loaded(background=background)
