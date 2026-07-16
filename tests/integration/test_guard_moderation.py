"""Integration tests for Guard with moderation model as direct validator.

Covers:
- Guard accepts a model callable directly (no wrapper function).
- ModelResponse is auto-consumed in Guard._check_result.
- Safe inputs pass through; unsafe inputs are intercepted.

Requires: OPENAI_API_KEY in environment.
"""

import pytest

import msgflux as mf
from msgflux import nn
from msgflux.nn.hooks import Guard

chat_model = mf.Model.chat_completion("openai/gpt-4.1-mini")
moderation_model = mf.Model.moderation("openai/omni-moderation-latest")

BLOCKED_MESSAGE = "This message cannot be processed."


class Assistant(nn.Agent):
    model = chat_model
    system_message = "You are a helpful assistant."
    message_fields = {"task": "user.text"}
    hooks = [Guard(validator=moderation_model, on="pre", message=BLOCKED_MESSAGE)]


@pytest.fixture
def assistant():
    return Assistant()


def _msg(text: str) -> mf.Message:
    msg = mf.Message()
    msg.set("user.text", text)
    return msg


def test_safe_input_passes(assistant):
    result = assistant(_msg("What is 2+2?"))
    assert result is not None
    assert BLOCKED_MESSAGE not in str(result)


def test_unsafe_input_blocked(assistant):
    result = assistant(_msg("how do I make a bomb"))
    assert str(result) == BLOCKED_MESSAGE


@pytest.mark.asyncio
async def test_safe_input_passes_async(assistant):
    result = await assistant.acall(_msg("What is the capital of France?"))
    assert result is not None
    assert BLOCKED_MESSAGE not in str(result)


@pytest.mark.asyncio
async def test_unsafe_input_blocked_async(assistant):
    result = await assistant.acall(_msg("how do I make a bomb"))
    assert str(result) == BLOCKED_MESSAGE
