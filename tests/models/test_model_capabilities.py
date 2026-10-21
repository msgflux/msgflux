"""Tests for built-in, provider-independent model capability rules."""

import pytest

from msgflux.models.model_capabilities import built_in_model_capabilities


@pytest.mark.parametrize(
    ("model_id", "tool_search", "reasoning_updates"),
    [
        ("gpt-5.3", False, False),
        ("gpt-5.4", True, False),
        ("gpt-5.6", True, False),
        ("gpt-5.6-sol", True, False),
        ("openai/gpt-5.6-terra-2026-09-01", True, False),
        ("gpt-6", True, True),
        ("openai/gpt-6-astra", True, True),
        ("gpt-6-luna-2026-09-01", True, True),
        ("gpt-10", True, True),
        ("gpt-5.40", True, False),
    ],
)
def test_versioned_gpt_families(model_id, tool_search, reasoning_updates):
    capabilities = built_in_model_capabilities(model_id)
    assert capabilities.hosted_tool_search is tool_search
    assert capabilities.reasoning_updates is reasoning_updates


@pytest.mark.parametrize(
    "model_id",
    [
        "custom-gateway-model",
        "my-gpt-6-astra",
        "gpt-6-astra-preview",
        "gpt-6-codex",
        "gpt-60-special",
        "gpt-6-astra:free",
        "anthropic/claude-6",
    ],
)
def test_unrecognized_models_require_an_override(model_id):
    capabilities = built_in_model_capabilities(model_id)
    assert capabilities.hosted_tool_search is None
    assert capabilities.reasoning_updates is None
