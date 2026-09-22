"""Provider-compatible streaming deltas need not repeat tool identity fields."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from msgflux.models.openai_compatible import (
    OpenAIChatCompletionsAPI,
    OpenAICompatibleChatCompletion,
)
from msgflux.models.response import ModelStreamResponse
from msgflux.models.tool_call_agg import ToolCallAggregator


def test_sparse_tool_deltas_retain_identity_and_concatenate_arguments():
    adapter = OpenAIChatCompletionsAPI()
    response = ModelStreamResponse()
    aggregator = ToolCallAggregator()
    deltas = [
        {"index": 0, "id": "call_one"},
        {"index": 1, "id": "call_two", "function": {"name": "second"}},
        {"index": 0, "function": {"name": "first", "arguments": '{"value":'}},
        {"index": 1, "function": {"arguments": "{}"}},
        {"index": 0, "function": {"arguments": "42}"}},
        {"index": 0, "function": {"arguments": None}},
    ]
    for delta in deltas:
        chunk = adapter.decode_stream_event(
            {"choices": [{"index": 0, "delta": {"tool_calls": [delta]}}]}
        )
        OpenAICompatibleChatCompletion._process_stream_tool_calls(
            chunk.choices[0].delta, response, aggregator
        )
    assert aggregator.get_calls() == [
        ("call_one", "first", {"value": 42}),
        ("call_two", "second", {}),
    ]
    assert response.response_type == "tool_call"


@pytest.mark.asyncio
@pytest.mark.parametrize("api_mode", ["chat_completions", "responses"])
async def test_model_closes_provider_stream_before_reporting_parse_failure(
    monkeypatch, api_mode
):
    from msgflux.models.providers.openai import OpenAIChatCompletion

    monkeypatch.setenv("OPENAI_API_KEY", "test-not-a-credential")
    model = OpenAIChatCompletion("test-model", api_mode=api_mode)
    closed = []

    async def events():
        try:
            yield object()
        finally:
            await asyncio.sleep(0)
            closed.append(True)

    def fail(*_args):
        raise ValueError("decode failed")

    monkeypatch.setattr(model, "_aexecute_model", AsyncMock(return_value=events()))
    monkeypatch.setattr(model, "_merge_effective_speed_metadata", fail)
    monkeypatch.setattr(model, "_handle_responses_stream_event", fail)
    response = ModelStreamResponse()
    try:
        generate = (
            model._astream_chat_completions_generate
            if api_mode == "chat_completions"
            else model._astream_responses_generate
        )
        await generate(stream_response=response)
        assert isinstance(response.error, ValueError)
        assert str(response.error) == "decode failed"
        assert closed == [True]
    finally:
        await model.aclose()
