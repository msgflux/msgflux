"""Provider stream cancellation finalizes as interrupted without dispatching tools."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from msgflux.models.providers.openai import OpenAIChatCompletion
from msgflux.models.response import ModelStreamResponse


def _text_event(model, api_mode):
    if api_mode == "chat_completions":
        payload = {
            "choices": [{"index": 0, "delta": {"content": "partial-before-cancel"}}]
        }
    else:
        payload = {
            "type": "response.output_text.delta",
            "delta": "partial-before-cancel",
            "output_index": 0,
            "content_index": 0,
            "item_id": "message-1",
        }
    return model.api_adapter.decode_stream_event(payload)


def _tool_event(model):
    payload = {
        "choices": [
            {
                "index": 0,
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call-danger",
                            "function": {"name": "danger", "arguments": '{"x":'},
                        }
                    ]
                },
            }
        ]
    }
    return model.api_adapter.decode_stream_event(payload)


async def _cancel_generation(model, response, api_mode, *, task_cancel, tools=None):
    started = asyncio.Event()
    cancellation = asyncio.CancelledError("original stream cancellation")

    async def events():
        yield _text_event(model, api_mode)
        if tools:
            yield _tool_event(model)
        started.set()
        if task_cancel:
            await asyncio.Event().wait()
        raise cancellation

    monkeypatch = AsyncMock(return_value=events())
    model._aexecute_with_metadata = monkeypatch
    generate = (
        model._astream_chat_completions_generate
        if api_mode == "chat_completions"
        else model._astream_responses_generate
    )
    task = asyncio.create_task(generate(stream_response=response, tools=tools))
    await asyncio.wait_for(started.wait(), timeout=5)
    if task_cancel:
        task.cancel("task cancellation")
    with pytest.raises(asyncio.CancelledError) as raised:
        await asyncio.wait_for(task, timeout=5)
    return raised.value


@pytest.mark.asyncio
@pytest.mark.parametrize("api_mode", ["chat_completions", "responses"])
@pytest.mark.parametrize("task_cancel", [False, True])
async def test_async_chat_stream_cancellation_is_interrupted_and_preserved(
    monkeypatch, api_mode, task_cancel
):
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-cancellation-test-key")
    model = OpenAIChatCompletion("test-model", api_mode=api_mode, retry=False)
    response = ModelStreamResponse(mode="async")

    try:
        error = await _cancel_generation(
            model, response, api_mode, task_cancel=task_cancel
        )

        assert response._final_status == "interrupted"
        assert response.error is error
        assert isinstance(error, asyncio.CancelledError)
        assert "partial-before-cancel" in str(response.chat_accumulator.snapshot())
    finally:
        await model.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("task_cancel", [False, True])
async def test_cancelled_chat_stream_preserves_text_without_publishing_partial_tool(
    monkeypatch, task_cancel
):
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-cancellation-test-key")
    model = OpenAIChatCompletion("test-model", api_mode="chat_completions", retry=False)
    response = ModelStreamResponse(mode="async")

    try:
        await _cancel_generation(
            model,
            response,
            "chat_completions",
            task_cancel=task_cancel,
            tools=[{"type": "function", "function": {"name": "danger"}}],
        )

        assert response._final_status == "interrupted"
        assert "partial-before-cancel" in str(response.chat_accumulator.snapshot())
        assert response.data is None
        assert response.get_tool_intents() == ()
    finally:
        await model.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("api_mode", ["chat_completions", "responses"])
async def test_cancellation_during_stream_cleanup_still_finalizes_response(
    monkeypatch, api_mode
):
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-cancellation-test-key")
    model = OpenAIChatCompletion("test-model", api_mode=api_mode, retry=False)
    response = ModelStreamResponse(mode="async")
    cleanup_started = asyncio.Event()

    async def events():
        yield _text_event(model, api_mode)

    async def blocked_close(model_output, stream_response):
        cleanup_started.set()
        await asyncio.Event().wait()
        return True

    monkeypatch.setattr(
        model, "_aexecute_with_metadata", AsyncMock(return_value=events())
    )
    monkeypatch.setattr(model, "_aclose_model_stream", blocked_close)
    generate = (
        model._astream_chat_completions_generate
        if api_mode == "chat_completions"
        else model._astream_responses_generate
    )
    task = asyncio.create_task(generate(stream_response=response))

    try:
        await asyncio.wait_for(cleanup_started.wait(), timeout=5)
        task.cancel("cancelled during close")
        with pytest.raises(
            asyncio.CancelledError, match="cancelled during close"
        ) as exc:
            await asyncio.wait_for(task, timeout=5)

        assert response._final_status == "interrupted"
        assert response.error is exc.value
        assert "partial-before-cancel" in str(response.chat_accumulator.snapshot())
        assert await response.next_chunk() == "partial-before-cancel"
        with pytest.raises(asyncio.CancelledError):
            await response.next_chunk()
    finally:
        await model.aclose()


@pytest.mark.asyncio
async def test_cleanup_cancellation_preserves_primary_stream_cancellation(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-cancellation-test-key")
    model = OpenAIChatCompletion("test-model", retry=False)
    response = ModelStreamResponse(mode="async")
    started = asyncio.Event()
    cleanup_started = asyncio.Event()

    async def events():
        yield _text_event(model, "chat_completions")
        started.set()
        await asyncio.Event().wait()

    async def blocked_close(model_output, stream_response):
        cleanup_started.set()
        await asyncio.Event().wait()
        return True

    monkeypatch.setattr(
        model, "_aexecute_with_metadata", AsyncMock(return_value=events())
    )
    monkeypatch.setattr(model, "_aclose_model_stream", blocked_close)
    task = asyncio.create_task(
        model._astream_chat_completions_generate(stream_response=response)
    )

    try:
        await asyncio.wait_for(started.wait(), timeout=5)
        task.cancel("primary cancellation")
        await asyncio.wait_for(cleanup_started.wait(), timeout=5)
        task.cancel("secondary cleanup cancellation")
        with pytest.raises(asyncio.CancelledError, match="primary cancellation") as exc:
            await asyncio.wait_for(task, timeout=5)

        assert response._final_status == "interrupted"
        assert response.error is exc.value
    finally:
        await model.aclose()
