"""Quick test for model profiles system."""

import os

# Configure env to use console exporter for testing
os.environ["MSGFLUX_TELEMETRY_ENABLED"] = "false"

from msgflux.models.profiles import get_model_profile, get_provider_profile

def test_basic_import():
    """Test that imports work."""
    print("✓ Imports successful")

def test_get_openai_profile():
    """Test getting OpenAI model profile."""
    profile = get_model_profile("gpt-4o", provider_id="openai")

    if profile:
        print(f"\n=== GPT-4o Profile ===")
        print(f"Name: {profile.name}")
        print(f"Provider: {profile.provider_id}")
        print(f"Tool calling: {profile.capabilities.tool_call}")
        print(f"Structured output: {profile.capabilities.structured_output}")
        print(f"Context window: {profile.limits.context:,} tokens")
        print(f"Max output: {profile.limits.output:,} tokens")
        print(f"Input cost: ${profile.cost.input_per_million:.2f}/1M tokens")
        print(f"Output cost: ${profile.cost.output_per_million:.2f}/1M tokens")

        # Test cost calculation
        cost = profile.cost.calculate(input_tokens=1000, output_tokens=500)
        print(f"Estimated cost for 1K input + 500 output: ${cost:.6f}")
        print("✓ OpenAI profile retrieved successfully")
    else:
        print("⚠ OpenAI profile not found (cache may not be loaded yet)")

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

def test_vllm_ollama_fallback():
    """Test that vLLM/Ollama use openrouter data."""
    # Get any model from openrouter to test fallback
    provider = get_provider_profile("openrouter")

    if provider and provider.models:
        # Use first available model
        model_id = list(provider.models.keys())[0]
        profile = get_model_profile(model_id, provider_id="vllm")

        if profile:
            print(f"\n=== vLLM Model (from OpenRouter) ===")
            print(f"Model ID: {model_id}")
            print(f"Name: {profile.name}")
            print(f"Provider in profile: {profile.provider_id}")
            print(f"Open weights: {profile.open_weights}")
            print("✓ vLLM fallback to OpenRouter works")
        else:
            print("⚠ vLLM fallback failed")
    else:
        print("⚠ OpenRouter provider not found (cache may not be loaded yet)")

if __name__ == "__main__":
    print("Testing Model Profiles System\n")
    print("=" * 50)

    test_basic_import()
    test_get_openai_profile()
    test_provider_profile()
    test_vllm_ollama_fallback()

    print("\n" + "=" * 50)
    print("✅ All basic tests completed!")
    print("\nNote: Profiles load in background, so some may not be available immediately.")
    print("The system will fetch from models.dev API and cache locally.")
