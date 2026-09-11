"""Tests for template functionality."""

from msgflux.generation.templates import PromptSpec


class TestPromptSpec:
    """Tests for PromptSpec class."""

    def test_prompt_spec_has_system_prompt(self):
        assert PromptSpec.SYSTEM_PROMPT == (
            "Instructions and stable context for the model"
        )
