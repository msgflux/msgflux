"""Native discovery reaches direct model and Agent event consumers."""

from unittest.mock import MagicMock

import pytest

from msgflux.models.providers.openai import OpenAIChatCompletion
from msgflux.models.providers.openai_codex import OpenAICodexChatCompletion
from msgflux.models.response import ModelStreamResponse
from msgflux.models.tool_call_agg import ToolCallAggregator
from msgflux.models.tool_search_events import ToolSearchEvents
from msgflux.nn.events import emit_model_response_events
from msgflux.nn.modules.module import Module
from msgflux.runtime.events import _capture_events, _EventSink


CALL = {
    "type": "tool_search_call",
    "id": "ts_1",
    "execution": "server",
    "call_id": None,
    "status": "completed",
    "arguments": {"paths": ["crm"]},
}
OUTPUT = {
    "type": "tool_search_output",
    "id": "tso_1",
    "execution": "server",
    "call_id": None,
    "status": "completed",
    "tools": [
        {
            "type": "namespace",
            "name": "crm",
            "tools": [
                {
                    "type": "function",
                    "name": "lookup",
                    "parameters": {"secret_schema": True},
                }
            ],
        }
    ],
}


def test_hosted_discovery_without_call_ids_is_paired_and_deduplicated():
    tracker = ToolSearchEvents("openai", "responses")
    events = tracker.observe({**CALL, "arguments": {}}, index=0)
    events += tracker.observe(CALL, index=0, done=True)
    events += tracker.observe(OUTPUT, index=1, done=True)
    assert [event.type for event in events] == [
        "tool.start",
        "tool.update",
        "tools.updated",
        "tool.end",
    ]
    assert {event.data["tool_call_id"] for event in events} == {"ts_1"}
    assert all(event.data["execution"] == "provider" for event in events)
    assert events[-1].data["result"] == {"loaded_tools": ["crm.lookup"]}
    assert "parameters" not in repr(events)
    assert tracker.observe(CALL, index=0, done=True) == []
    assert tracker.observe(OUTPUT, index=1, done=True) == []
    assert tracker.close("failed") == []


def test_search_without_item_ids_keeps_generated_identity_and_output_order():
    tracker = ToolSearchEvents("openai-codex", "responses")
    call = {key: value for key, value in CALL.items() if key != "id"}
    start = tracker.observe({**call, "arguments": {}}, index=0)[0]
    update = tracker.observe(call, index=0, done=True)[0]
    output = tracker.observe(OUTPUT, index=1, done=True)[-1]
    assert (
        start.data["tool_call_id"]
        == update.data["tool_call_id"]
        == output.data["tool_call_id"]
    )


def test_explicit_ids_pair_parallel_searches():
    tracker = ToolSearchEvents("openai", "responses")
    tracker.observe({**CALL, "call_id": "first"}, index=0)
    tracker.observe({**CALL, "call_id": "second"}, index=1)
    result = tracker.observe({**OUTPUT, "call_id": "second"}, index=2, done=True)
    assert result[-1].data["tool_call_id"] == "second"
    assert tracker.close("transport failed")[0].data["tool_call_id"] == "first"


@pytest.mark.parametrize("status", ["failed", "incomplete", "cancelled"])
def test_failed_output_does_not_report_loaded_tools(status):
    tracker = ToolSearchEvents("openai", "responses")
    tracker.observe(CALL, index=0)
    events = tracker.observe({**OUTPUT, "status": status}, index=1, done=True)
    assert [event.type for event in events] == ["tool.end"]
    assert events[0].data["error"] == status
    assert events[0].data["result"] == {"loaded_tools": []}


def test_empty_search_and_orphan_output_do_not_fabricate_updates():
    tracker = ToolSearchEvents("openai", "responses")
    assert tracker.observe(OUTPUT, index=1, done=True) == []
    tracker.observe(CALL, index=0)
    events = tracker.observe({**OUTPUT, "tools": []}, index=2, done=True)
    assert [event.type for event in events] == ["tool.end"]
    assert events[0].data["error"] is None


def test_failed_call_and_transport_close_emit_one_terminal_event():
    tracker = ToolSearchEvents("openai", "responses")
    events = tracker.observe({**CALL, "status": "failed"}, index=0, done=True)
    assert [event.type for event in events] == ["tool.start", "tool.end"]
    assert tracker.close("transport failed") == []
    tracker.observe({**CALL, "id": "ts_2"}, index=1)
    assert tracker.close("transport failed")[0].data["error"] == "transport failed"
    assert tracker.close("transport failed") == []


@pytest.mark.parametrize("provider", [OpenAIChatCompletion, OpenAICodexChatCompletion])
@pytest.mark.asyncio
async def test_provider_stream_and_runtime_forward_search_before_answer(
    provider, monkeypatch
):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    model = provider(model_id="gpt-6-luna")
    response = ModelStreamResponse()
    state = model._new_responses_stream_state(MagicMock())
    aggregator = ToolCallAggregator(api_mode="responses")
    for index, item in enumerate([CALL, OUTPUT]):
        for kind in ["added", "done"]:
            model._handle_responses_stream_event(
                {
                    "type": f"response.output_item.{kind}",
                    "output_index": index,
                    "item": item,
                },
                response,
                aggregator,
                state,
            )
    response.set_response_type("text_generation")
    response.add("OK")
    response.finish()
    events = []
    with _capture_events(_EventSink(events.append)):
        await Module._aconsume_event_response(response)
    assert [event.type for event in events] == [
        "tool.start",
        "tools.updated",
        "tool.end",
        "message.delta",
    ]
    assert events[1].data["loaded_tools"] == ["crm.lookup"]
    assert events[0].data["provider"] == model.provider
    assert len(response.chat_accumulator.snapshot()) == 3
    await model.aclose()


@pytest.mark.parametrize("provider", [OpenAIChatCompletion, OpenAICodexChatCompletion])
def test_ordinary_response_forwards_events_and_preserves_history(provider, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    model = provider(model_id="gpt-6-luna")
    response = model._process_responses_model_output(
        {
            "status": "completed",
            "output": [
                CALL,
                OUTPUT,
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "OK"}],
                },
            ],
        }
    )
    events = []
    with _capture_events(_EventSink(events.append)):
        emit_model_response_events(response)
    assert [event.type for event in events] == [
        "tool.start",
        "tools.updated",
        "tool.end",
        "model.response",
    ]
    assert response.consume() == "OK"
    assert [item["provider_state"]["data"] for item in response.history_items[:2]] == [
        CALL,
        OUTPUT,
    ]
    model.close()


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_transport_failure_settles_pending_search(asynchronous, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    model = OpenAIChatCompletion(model_id="gpt-6-luna")
    added = {"type": "response.output_item.added", "output_index": 0, "item": CALL}

    def stream(**kwargs):
        yield added
        raise RuntimeError("connection lost")

    async def astream(**kwargs):
        yield added
        raise RuntimeError("connection lost")

    async def execute(**kwargs):
        return astream()

    response = ModelStreamResponse(mode="async" if asynchronous else "sync")
    if asynchronous:
        monkeypatch.setattr(model, "_aexecute_model", execute)
        await model._astream_responses_generate(stream_response=response)
    else:
        monkeypatch.setattr(model, "_execute_model", stream)
        model._stream_responses_generate(stream_response=response)
    events = []
    with pytest.raises(RuntimeError, match="connection lost"):
        async for event in response.consume_events():
            events.append(event)
    assert [event.type for event in events] == ["tool.start", "tool.end"]
    assert events[-1].data["error"] == "connection lost"
    await model.aclose()


def test_late_native_identifier_resolves_to_original_event_identity():
    tracker = ToolSearchEvents("openai", "responses")
    start = tracker.observe({"type": "tool_search_call"}, index=0)[0]
    update = tracker.observe(CALL, index=0, done=True)[0]
    events = tracker.observe(
        {**OUTPUT, "tool_search_call_id": "ts_1"}, index=1, done=True
    )
    assert (
        start.data["tool_call_id"]
        == update.data["tool_call_id"]
        == events[-1].data["tool_call_id"]
    )
