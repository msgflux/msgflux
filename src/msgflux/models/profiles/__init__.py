"""Model profiles from models.dev.

Provides access to model metadata, capabilities, pricing, and limits.

Usage:
    from msgflux.models.profiles import get_model_profile

    # Get profile for a specific model
    profile = get_model_profile("gpt-4o", provider_id="openai")

    # Check capabilities
    if profile and profile.capabilities.tool_call:
        print("Model supports tool calling")

    # Calculate cost
    if profile:
        cost = profile.cost.calculate(
            input_tokens=1000,
            output_tokens=500
        )
        print(f"Estimated cost: ${cost:.4f}")
"""

from msgflux.models.profiles.base import (
    ModelCapabilities,
    ModelCost,
    ModelLimits,
    ModelModalities,
    ModelProfile,
    ProviderProfile,
)
from msgflux.models.profiles.registry import (
    ensure_profiles_loaded,
    get_model_profile,
    get_provider_profile,
)

__all__ = [
    # Data structures
    "ModelProfile",
    "ProviderProfile",
    "ModelCapabilities",
    "ModelModalities",
    "ModelCost",
    "ModelLimits",
    # Registry functions
    "get_model_profile",
    "get_provider_profile",
    "ensure_profiles_loaded",
]
