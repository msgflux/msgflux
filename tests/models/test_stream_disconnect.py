"""Received text survives provider disconnects without completing tool calls."""

import httpx2
import pytest

from msgflux.models.chat_transport import HTTPChatTransport
from msgflux.models.providers.openai import OpenAIChatCompletion
from msgflux.tools.definitions import ToolCatalog
from tests.models._stream_disconnect import partial_text, tool_call


class _DisconnectedBody(httpx2.SyncByteStream, httpx2.AsyncByteStream):
    def __init__(self, api_mode, incomplete_tool):
        self.api_mode = api_mode
        self.incomplete_tool = incomplete_tool
        self.closed = False
        self.error = httpx2.ReadError("synthetic upstream connection dropped")

    def __iter__(self):
        yield partial_text(self.api_mode)
        if self.incomplete_tool:
            yield tool_call(self.api_mode, complete=False)
        raise self.error

    async def __aiter__(self):
        for chunk in self:
            yield chunk

    def close(self):
        self.closed = True

    async def aclose(self):
        self.close()


@pytest.mark.parametrize("api_mode", ["chat_completions", "responses"])
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("incomplete_tool", [False, True])
@pytest.mark.asyncio
async def test_stream_drop_preserves_text_and_original_error(
    monkeypatch, api_mode, asynchronous, incomplete_tool
):
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-fault-test-key")
    body = _DisconnectedBody(api_mode, incomplete_tool)
    requests = []

    def handler(request):
        requests.append(request)
        return httpx2.Response(200, stream=body)

    catalog = ToolCatalog.from_function_schemas(
        [
            {
                "type": "function",
                "function": {
                    "name": "record_effect",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
    )
    http = (
        httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
        if asynchronous
        else httpx2.Client(transport=httpx2.MockTransport(handler))
    )
    model = OpenAIChatCompletion(
        model_id="gpt-4.1-mini",
        api_mode=api_mode,
        chat_transport=HTTPChatTransport(
            **{"async_client" if asynchronous else "client": http}, max_retries=0
        ),
        retry=False,
    )
    events = []
    try:
        response = (
            await model.acall("answer", stream=True, tool_catalog=catalog)
            if asynchronous
            else model("answer", stream=True, tool_catalog=catalog)
        )
        with pytest.raises(httpx2.ReadError) as raised:
            async for event in response.consume_events():
                events.append(event)
        assert raised.value is body.error
        assert response._final_status == "failed"
        assert "partial-before-drop" in "".join(str(event.data) for event in events)
        assert "partial-before-drop" in str(response.chat_accumulator.snapshot())
        if incomplete_tool:
            assert response.data is None
        assert len(requests) == 1 and body.closed
    finally:
        if asynchronous:
            await model.aclose()
            await http.aclose()
        else:
            model.close()
            http.close()
