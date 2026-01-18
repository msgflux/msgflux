"""Quick test for model profiles system."""

from msgflux.models.profiles import get_model_profile, get_provider_profile

def test_provider_profile():
    """Test getting provider profile."""
    provider = get_provider_profile("openai")

    if provider:
        print(f"\n=== OpenAI Provider ===")
        print(f"Name: {provider.name}")
        print(f"Models available: {len(provider.models)}")
        print(f"First 5 models: {list(provider.models.keys())[:5]}")
        print("✓ Provider profile retrieved successfully")
    else:
        print("⚠ Provider profile not found (cache may not be loaded yet)")

def test_openrouter_profile():
    """Test getting OpenRouter model profile."""
    provider = get_provider_profile("openrouter")

    if provider and provider.models:
        # Use first available model
        model_id = list(provider.models.keys())[0]
        profile = get_model_profile(model_id, provider_id="openrouter")

        if profile:
            print(f"\n=== OpenRouter Model ===")
            print(f"Model ID: {model_id}")
            print(f"Name: {profile.name}")
            print(f"Provider in profile: {profile.provider_id}")
            print(f"Open weights: {profile.open_weights}")
            print("✓ OpenRouter profile retrieved successfully")
        else:
            print("⚠ OpenRouter profile not found")
    else:
        print("⚠ OpenRouter provider not found (cache may not be loaded yet)")
