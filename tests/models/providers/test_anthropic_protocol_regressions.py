"""Regressions for failures observed against the native Anthropic API."""

import asyncio
import json

import httpx2
import pytest

from msgflux.models.http_transport import HTTPTransport
from msgflux.models.providers.anthropic import AnthropicChatCompletion
from msgflux.tools import ToolCatalog


@pytest.fixture(autouse=True)
def credentials(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-regression-key")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")


def catalog():
    return ToolCatalog.from_function_schemas(
        [
            {
                "type": "function",
                "function": {
                    "name": "submit_order",
                    "description": "Submit a synthetic order.",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {"quantity": {"type": "integer"}},
                        "required": ["quantity"],
                        "additionalProperties": False,
                    },
                },
            }
        ],
        choice="submit_order",
    )


def test_named_tool_choice_preserves_strict_schema_on_wire():
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx2.Response(
            200,
            json={
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "model": "claude-haiku-5-5",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "submit_order",
                        "input": {"quantity": 2},
                    }
                ],
                "stop_reason": "tool_use",
                "usage": {"input_tokens": 4, "output_tokens": 8},
            },
        )

    with httpx2.Client(transport=httpx2.MockTransport(handler)) as client:
        model = AnthropicChatCompletion(model_id="claude-haiku-5-5", retry=False)
        model.http_transport = HTTPTransport(client=client, max_retries=0)
        response = model("Submit two.", tool_catalog=catalog())
    assert list(response.get_tool_intents())[0].arguments == {"quantity": 2}
    assert requests[0]["tool_choice"] == {"type": "tool", "name": "submit_order"}
    assert requests[0]["tools"][0]["strict"] is True


@pytest.mark.parametrize("restore", [False, True])
def test_canonical_agent_tool_history_replays_after_model_restore(restore):
    from msgflux import Model

    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx2.Response(
            200,
            json={
                "id": "msg_final",
                "type": "message",
                "role": "assistant",
                "model": "claude-haiku-5-5",
                "content": [{"type": "text", "text": "DONE"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 4, "output_tokens": 2},
            },
        )

    model = AnthropicChatCompletion(model_id="claude-haiku-5-5", retry=False)
    if restore:
        state = model.serialize()
        model = Model.from_serialized(
            state["provider"], state["model_type"], state["state"]
        )
    with httpx2.Client(transport=httpx2.MockTransport(handler)) as client:
        model.http_transport = HTTPTransport(client=client, max_retries=0)
        response = model(
            [
                {"type": "message", "role": "user", "content": "Submit two."},
                {
                    "type": "function_call",
                    "call_id": "toolu_1",
                    "name": "submit_order",
                    "arguments": {"quantity": 2},
                },
                {
                    "type": "function_call_output",
                    "call_id": "toolu_1",
                    "output": "Order accepted",
                },
            ]
        )
    assert response.consume() == "DONE"
    messages = requests[0]["messages"]
    assert messages[1] == {
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "submit_order",
                "input": {"quantity": 2},
            }
        ],
    }
    assert messages[2] == {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "toolu_1",
                "content": "Order accepted",
            }
        ],
    }


def test_loading_deferred_tool_preserves_signed_system_and_tools_prefix():
    import msgflux as mf
    from msgflux import nn
    from msgflux.nn.extensions.prompt import ToolUsageGuidanceExtension
    from msgflux.nn.hooks.events import ModelContext
    from msgflux.runtime.context import ExecutionScope

    @mf.tool_config(defer_loading=True, usage_guidance="Use product SKUs.")
    def inventory_lookup(sku: str) -> str:
        """Look up inventory for a product SKU."""
        return sku

    model = AnthropicChatCompletion(model_id="claude-haiku-5-5", retry=False)
    guidance = ToolUsageGuidanceExtension()
    agent = nn.Agent(
        name="stable_prefix",
        model=model,
        tools=[inventory_lookup],
        extensions=[guidance],
    )
    history = mf.ChatMessages(thread_id="stable-prefix")
    views = [agent.tool_library.get_tool_catalog_view(history)]
    agent.tool_library.load_tools(history, ["inventory_lookup"])
    views.append(agent.tool_library.get_tool_catalog_view(history))
    assert views[0].has_deferred is True
    assert views[1].has_deferred is False
    bodies = []
    for view in views:
        context = ModelContext(
            scope=ExecutionScope(thread_id=history.thread_id),
            system_prompt="Check inventory.",
            tool_catalog=view,
        )
        prompt = guidance._add_guidance(context).system_prompt
        params = model._build_generation_params("Find BOLT-17", prompt, None, view)
        bodies.append(model._to_anthropic_body({**model.sampling_run_params, **params}))
    assert bodies[0]["system"] == bodies[1]["system"]
    assert "Use product SKUs." in bodies[0]["system"]
    assert 'name="tool_search"' not in bodies[0]["system"]
    assert bodies[0]["tools"] == bodies[1]["tools"]
    assert bodies[1]["tools"][1]["defer_loading"] is True


@pytest.mark.parametrize("arguments", ['{"quantity":', "[]", '"value"', "null"])
def test_incomplete_or_nonobject_tool_history_is_rejected_without_replacing_arguments(
    arguments,
):
    model = AnthropicChatCompletion(model_id="claude-haiku-5-5", retry=False)
    with pytest.raises(ValueError, match="JSON object"):
        model(
            [
                {"type": "message", "role": "user", "content": "Submit two."},
                {
                    "type": "function_call",
                    "call_id": "toolu_1",
                    "name": "submit_order",
                    "arguments": arguments,
                },
                {
                    "type": "function_call_output",
                    "call_id": "toolu_1",
                    "output": "Accepted",
                },
            ]
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "initial,fragments,expected",
    [
        ({}, ['{"quantity":', "2}"], {"quantity": 2}),
        ({"quantity": 3}, [], {"quantity": 3}),
        ({}, [], {}),
    ],
)
async def test_public_tool_stream_has_one_complete_argument_object(
    asynchronous, initial, fragments, expected
):
    events = [
        ("message_start", {"message": {"usage": {"input_tokens": 4}}}),
        (
            "content_block_start",
            {
                "index": 0,
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "submit_order",
                    "input": initial,
                },
            },
        ),
        *[
            (
                "content_block_delta",
                {
                    "index": 0,
                    "delta": {"type": "input_json_delta", "partial_json": part},
                },
            )
            for part in fragments
        ],
        ("content_block_stop", {"index": 0}),
        (
            "message_delta",
            {"delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 8}},
        ),
        ("message_stop", {}),
    ]
    data = "".join(
        f"event: {name}\ndata: {json.dumps(payload)}\n\n" for name, payload in events
    )

    def handler(_request):
        return httpx2.Response(
            200, text=data, headers={"content-type": "text/event-stream"}
        )

    with httpx2.Client(transport=httpx2.MockTransport(handler)) as client:
        async with httpx2.AsyncClient(
            transport=httpx2.MockTransport(handler)
        ) as aclient:
            model = AnthropicChatCompletion(model_id="claude-haiku-5-5", retry=False)
            model.http_transport = HTTPTransport(
                client=client, async_client=aclient, max_retries=0
            )
            response = (
                await model.acall("Submit order", stream=True)
                if asynchronous
                else model("Submit order", stream=True)
            )

            async def consume():
                return [chunk async for chunk in response.consume()]

            await asyncio.wait_for(consume(), timeout=2)
            intents = list(response.get_tool_intents())
            assert len(intents) == 1
            assert intents[0].name == "submit_order"
            assert intents[0].arguments == expected


@pytest.mark.parametrize("mode", ["portable", "missing_capability_method"])
def test_stable_prompt_portable_fallback_keeps_hidden_guidance_hidden(mode):
    import msgflux as mf
    from msgflux import nn
    from msgflux.nn.extensions.prompt import ToolUsageGuidanceExtension
    from msgflux.nn.hooks.events import ModelContext
    from msgflux.runtime.context import ExecutionScope

    @mf.tool_config(defer_loading=True, usage_guidance="Use product SKUs.")
    def inventory_lookup(sku: str) -> str:
        """Look up inventory."""
        return sku

    model = AnthropicChatCompletion(model_id="claude-haiku-5-5", native_tools=False)
    if mode == "missing_capability_method":
        model.supports_native_tool_search = None
    guidance = ToolUsageGuidanceExtension()
    agent = nn.Agent(
        name="portable", model=model, tools=[inventory_lookup], extensions=[guidance]
    )
    messages = mf.ChatMessages(thread_id="portable")
    view = agent.tool_library.get_tool_catalog_view(messages)
    ctx = ModelContext(
        scope=ExecutionScope(thread_id=messages.thread_id),
        system_prompt="Check stock.",
        tool_catalog=view,
    )
    rendered = guidance._add_guidance(ctx).system_prompt
    assert "Use product SKUs." not in rendered
    assert "Load a match with `select`" in rendered


def test_server_pause_is_explicit_and_never_becomes_a_client_tool_call():
    model = AnthropicChatCompletion(model_id="claude-haiku-5-5")
    native = {
        "id": "msg_paused",
        "type": "message",
        "role": "assistant",
        "model": model.model_id,
        "content": [
            {
                "type": "server_tool_use",
                "id": "srvtoolu_1",
                "name": "tool_search_tool_bm25",
                "input": {"query": "inventory"},
            }
        ],
        "stop_reason": "pause_turn",
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }
    response = model._process_completion_model_output(
        model._anthropic_to_completion(native, stream=False)
    )
    assert response.metadata.stop_reason == "pause_turn"
    assert response.response_type == "text_generation"
    assert list(response.get_tool_intents()) == []
    assert any(item["type"] == "tool_search_call" for item in response.history_items)
