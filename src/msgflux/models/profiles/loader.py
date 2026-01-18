"""Profile loader for models.dev API.

Handles fetching, caching, and deserialization of model profiles.
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, Optional

from msgflux.envs import envs
from msgflux.models.profiles.base import (
    ModelCapabilities,
    ModelCost,
    ModelLimits,
    ModelModalities,
    ModelProfile,
    ProviderProfile,
)


def get_cache_dir() -> Path:
    """Get cache directory following HuggingFace convention.

    Priority:
    1. MSGFLUX_CACHE_HOME env var
    2. XDG_CACHE_HOME/msgflux
    3. ~/.cache/msgflux

    Returns:
        Path to cache directory
    """
    cache_home = os.environ.get("MSGFLUX_CACHE_HOME")
    if cache_home:
        return Path(cache_home)

    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache:
        return Path(xdg_cache) / "msgflux"

    return Path.home() / ".cache" / "msgflux"


class ProfileLoader:
    """Loader for model profiles from models.dev."""

    API_URL = "https://models.dev/api.json"
    CACHE_FILE = get_cache_dir() / "model_profiles.json"
    CACHE_TTL = envs.profile_cache_ttl

    @staticmethod
    def is_cache_valid() -> bool:
        """Check if cache exists and is not expired.

        Returns:
            True if cache is valid, False otherwise
        """
        if not ProfileLoader.CACHE_FILE.exists():
            return False

        try:
            with open(ProfileLoader.CACHE_FILE) as f:
                data = json.load(f)
                cached_at = data.get("_cached_at", 0)
                age = time.time() - cached_at
                return age < ProfileLoader.CACHE_TTL
        except Exception:
            return False

    @staticmethod
    def load_from_cache() -> Optional[Dict[str, ProviderProfile]]:
        """Load profiles from cache if valid.

        Returns:
            Dictionary of provider profiles if cache is valid, None otherwise
        """
        if not ProfileLoader.is_cache_valid():
            return None

        try:
            with open(ProfileLoader.CACHE_FILE) as f:
                data = json.load(f)
                # Remove metadata
                data.pop("_cached_at", None)
                # Convert dict to ProviderProfile objects
                return ProfileLoader._deserialize(data)
        except Exception:
            return None

    @staticmethod
    def save_to_cache(profiles: Dict[str, ProviderProfile]):
        """Save profiles to cache with timestamp.

        Args:
            profiles: Dictionary of provider profiles to cache
        """
        ProfileLoader.CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)

        # Serialize profiles + add timestamp
        data = ProfileLoader._serialize(profiles)
        data["_cached_at"] = time.time()

        with open(ProfileLoader.CACHE_FILE, "w") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def fetch_profiles_sync() -> Dict[str, ProviderProfile]:
        """Fetch profiles from models.dev API (sync).

        Returns:
            Dictionary of provider profiles

        Raises:
            Exception: If fetch fails
        """
        try:
            import httpx
        except ImportError as e:
            raise ImportError(
                "httpx is required for profile fetching. "
                "Install with: pip install msgflux[httpx]"
            ) from e

        response = httpx.get(ProfileLoader.API_URL, timeout=10.0)
        response.raise_for_status()
        raw_data = response.json()

        # Convert to ProviderProfile objects
        return ProfileLoader._deserialize(raw_data)

    @staticmethod
    def _deserialize(raw_data: dict) -> Dict[str, ProviderProfile]:
        """Convert raw JSON data to ProviderProfile objects.

        Args:
            raw_data: Raw JSON data from models.dev

        Returns:
            Dictionary of provider profiles
        """
        providers = {}

        for provider_id, provider_data in raw_data.items():
            # Skip metadata fields
            if provider_id.startswith("_"):
                continue

            models = {}
            for model_id, model_data in provider_data.get("models", {}).items():
                # Parse capabilities
                capabilities = ModelCapabilities(
                    tool_call=model_data.get("tool_call", False),
                    structured_output=model_data.get("structured_output", False),
                    reasoning=model_data.get("reasoning", False),
                    attachment=model_data.get("attachment", False),
                    temperature=model_data.get("temperature", False),
                )

                # Parse modalities
                modalities_data = model_data.get("modalities", {})
                modalities = ModelModalities(
                    input=modalities_data.get("input", ["text"]),
                    output=modalities_data.get("output", ["text"]),
                )

                # Parse cost
                cost_data = model_data.get("cost", {})
                cost = ModelCost(
                    input_per_million=cost_data.get("input", 0.0),
                    output_per_million=cost_data.get("output", 0.0),
                    cache_read_per_million=cost_data.get("cache_read"),
                )

                # Parse limits
                limit_data = model_data.get("limit", {})
                limits = ModelLimits(
                    context=limit_data.get("context", 0),
                    output=limit_data.get("output", 0),
                )

                # Create model profile
                model_profile = ModelProfile(
                    id=model_data.get("id", model_id),
                    name=model_data.get("name", model_id),
                    provider_id=provider_id,
                    capabilities=capabilities,
                    modalities=modalities,
                    cost=cost,
                    limits=limits,
                    knowledge=model_data.get("knowledge"),
                    release_date=model_data.get("release_date"),
                    last_updated=model_data.get("last_updated"),
                    open_weights=model_data.get("open_weights", False),
                )

                models[model_id] = model_profile

            # Create provider profile
            provider_profile = ProviderProfile(
                id=provider_data.get("id", provider_id),
                name=provider_data.get("name", provider_id),
                api_base=provider_data.get("api"),
                env_vars=provider_data.get("env", []),
                doc_url=provider_data.get("doc"),
                models=models,
            )

            providers[provider_id] = provider_profile

        return providers

    @staticmethod
    def _serialize(profiles: Dict[str, ProviderProfile]) -> dict:
        """Convert ProviderProfile objects to JSON-serializable dict.

        Args:
            profiles: Dictionary of provider profiles

        Returns:
            JSON-serializable dictionary
        """
        result = {}

        for provider_id, provider in profiles.items():
            models_dict = {}
            for model_id, model in provider.models.items():
                models_dict[model_id] = {
                    "id": model.id,
                    "name": model.name,
                    "tool_call": model.capabilities.tool_call,
                    "structured_output": model.capabilities.structured_output,
                    "reasoning": model.capabilities.reasoning,
                    "attachment": model.capabilities.attachment,
                    "temperature": model.capabilities.temperature,
                    "modalities": {
                        "input": model.modalities.input,
                        "output": model.modalities.output,
                    },
                    "cost": {
                        "input": model.cost.input_per_million,
                        "output": model.cost.output_per_million,
                        "cache_read": model.cost.cache_read_per_million,
                    },
                    "limit": {
                        "context": model.limits.context,
                        "output": model.limits.output,
                    },
                    "knowledge": model.knowledge,
                    "release_date": model.release_date,
                    "last_updated": model.last_updated,
                    "open_weights": model.open_weights,
                }

            result[provider_id] = {
                "id": provider.id,
                "name": provider.name,
                "api": provider.api_base,
                "env": provider.env_vars,
                "doc": provider.doc_url,
                "models": models_dict,
            }

        return result
