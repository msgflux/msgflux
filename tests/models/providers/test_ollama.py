"""Tests for the native and OpenAI-compatible Ollama transports."""

from unittest.mock import MagicMock, patch

import pytest

from msgflux.chat_messages import ChatMessages
from msgflux.models.reasoning import OllamaReasoningCodec


@pytest.fixture(autouse=True)
def ollama_env(monkeypatch):
    monkeypatch.setenv("OLLAMA_API_KEY", "ollama")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")


@pytest.fixture
def mock_native_clients():
    client = MagicMock()
    aclient = MagicMock()
    with (
        patch("msgflux.models.providers.ollama.httpx2.Client", return_value=client),
        patch(
            "msgflux.models.providers.ollama.httpx2.AsyncClient",
            return_value=aclient,
        ),
    ):
        yield client, aclient


def test_native_chat_is_default(mock_native_clients):
    from msgflux.models.providers.ollama import OllamaChatCompletion

    model = OllamaChatCompletion(model_id="qwen3:8b")

    assert model.api_mode == "ollama_chat"
    assert model.supported_api_modes == ("ollama_chat", "chat_completions")
    assert isinstance(model.reasoning_codec, OllamaReasoningCodec)
    assert model.sampling_params["base_url"] == "http://localhost:11434"
    assert model._native_url() == "http://localhost:11434/api/chat"


def test_native_chat_replays_thinking(mock_native_clients):
    from msgflux.models.providers.ollama import OllamaChatCompletion

    model = OllamaChatCompletion(model_id="qwen3:8b", enable_thinking=True)
    messages = ChatMessages()
    messages.add_reasoning("private chain")
    messages.add_assistant("first answer")
    messages.add_user("follow up")

    params = model._build_generation_params(
        messages,
        system_prompt=None,
        prefilling=None,
        tool_catalog=None,
    )
    adapted = model._adapt_params(params)

    assert adapted["messages"] == [
        {
            "role": "assistant",
            "content": "first answer",
            "thinking": "private chain",
        },
        {"role": "user", "content": "follow up"},
    ]
    assert adapted["think"] is True


@pytest.mark.parametrize("level", ["low", "medium", "high"])
def test_native_chat_accepts_thinking_levels(mock_native_clients, level):
    from msgflux.models.providers.ollama import OllamaChatCompletion

    model = OllamaChatCompletion(model_id="gpt-oss:20b", enable_thinking=level)

    adapted = model._adapt_params({"messages": []})

    assert adapted["think"] == level
    assert "reasoning_effort" not in model._build_response_metadata(None).model


def test_native_params_map_sampling_schema_tools_and_images(mock_native_clients):
    from msgflux.models.providers.ollama import OllamaChatCompletion

    model = OllamaChatCompletion(
        model_id="qwen3-vl:2b",
        max_tokens=42,
        temperature=0.25,
        top_p=0.8,
        extra_body={"keep_alive": "10m", "options": {"seed": 7}},
    )
    params = model._adapt_params(
        {
            **model.sampling_run_params,
            "model": model.model_id,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,YWJj"},
                        },
                    ],
                }
            ],
            "tools": [{"type": "function", "function": {"name": "lookup"}}],
            "tool_choice": "auto",
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "answer",
                    "schema": {"type": "object"},
                },
            },
        }
    )

    assert params["messages"] == [
        {"role": "user", "content": "describe", "images": ["YWJj"]}
    ]
    assert params["options"] == {
        "seed": 7,
        "num_predict": 42,
        "temperature": 0.25,
        "top_p": 0.8,
    }
    assert params["format"] == {"type": "object"}
    assert params["keep_alive"] == "10m"
    assert params["stream"] is False
    assert params["tools"][0]["function"]["name"] == "lookup"


def test_native_response_is_parsed_with_reasoning_tools_and_usage(
    mock_native_clients,
):
    from msgflux.models.providers.ollama import OllamaChatCompletion

    model = OllamaChatCompletion(model_id="qwen3:8b", enable_thinking=True)
    native = {
        "model": "qwen3:8b",
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 11,
        "eval_count": 7,
        "message": {
            "role": "assistant",
            "content": "",
            "thinking": "I should call lookup.",
            "tool_calls": [
                {
                    "function": {
                        "name": "lookup",
                        "arguments": {"sku": "1842"},
                    }
                }
            ],
        },
    }

    response = model._process_model_output(
        model._native_to_completion(native, stream=False)
    )

    assert response.response_type == "tool_call"
    assert response.reasoning == "I should call lookup."
    call_id, name, arguments = response.data.get_calls()[0]
    assert call_id.startswith("ollama_call_")
    assert (name, arguments) == ("lookup", {"sku": "1842"})
    assert response.metadata.usage.input_tokens == 11
    assert response.metadata.usage.output_tokens == 7
    assert response.metadata.usage.total_tokens == 18
    assert response.history_items[0]["text"] == "I should call lookup."


def test_native_chat_normalizes_canonical_tool_loop_history(mock_native_clients):
    from msgflux.models.providers.ollama import OllamaChatCompletion

    model = OllamaChatCompletion(model_id="qwen3:8b")
    params = model._build_generation_params(
        [
            {"role": "user", "content": "Add 2 and 3"},
            {
                "type": "function_call",
                "call_id": "call_add",
                "name": "add",
                "arguments": '{"a":2,"b":3}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_add",
                "output": "5",
            },
        ],
        system_prompt=None,
        prefilling=None,
        tool_catalog=None,
    )

    adapted = model._adapt_params(params)

    assert adapted["messages"] == [
        {"role": "user", "content": "Add 2 and 3"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "id": "call_add",
                    "function": {
                        "name": "add",
                        "arguments": {"a": 2, "b": 3},
                    },
                }
            ],
        },
        {"role": "tool", "content": "5", "tool_name": "add"},
    ]


def test_native_missing_tool_ids_are_unique_across_responses():
    from msgflux.models.providers.ollama import OllamaChatCompletion

    first = OllamaChatCompletion._native_to_completion(
        {"message": {"tool_calls": [{"function": {"name": "first"}}]}},
        stream=False,
    )
    second = OllamaChatCompletion._native_to_completion(
        {"message": {"tool_calls": [{"function": {"name": "second"}}]}},
        stream=False,
    )

    first_id = first.choices[0].message.tool_calls[0].id
    second_id = second.choices[0].message.tool_calls[0].id
    assert first_id != second_id


def test_native_execute_posts_to_api_chat(mock_native_clients):
    from msgflux.models.providers.ollama import OllamaChatCompletion

    client, _ = mock_native_clients
    http_response = MagicMock()
    http_response.json.return_value = {
        "model": "smollm2:135m",
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 3,
        "eval_count": 1,
        "message": {"role": "assistant", "content": "ok"},
    }
    client.post.return_value = http_response
    model = OllamaChatCompletion(model_id="smollm2:135m")

    response = model("say ok")

    assert response.data == "ok"
    client.post.assert_called_once()
    assert client.post.call_args.args == ("http://localhost:11434/api/chat",)
    assert client.post.call_args.kwargs["json"]["messages"] == [
        {"role": "user", "content": "say ok"}
    ]


def test_openai_compatible_chat_does_not_replay_thinking():
    from msgflux.models.providers.ollama import OllamaChatCompletion

    model = OllamaChatCompletion(
        model_id="qwen3:8b",
        api_mode="chat_completions",
        enable_thinking=True,
    )
    messages = ChatMessages()
    messages.add_reasoning("private chain")
    messages.add_assistant("first answer")
    messages.add_user("follow up")

    params = model._build_generation_params(
        messages,
        system_prompt=None,
        prefilling=None,
        tool_catalog=None,
    )

    assert params["messages"] == [
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "follow up"},
    ]


def test_openai_compatible_chat_keeps_thinking_request_control():
    from msgflux.models.providers.ollama import OllamaChatCompletion

    model = OllamaChatCompletion(
        model_id="qwen3:8b",
        api_mode="chat_completions",
        enable_thinking=True,
    )

    adapted = model._adapt_params({"messages": [], "extra_body": {}})

    assert adapted["extra_body"]["think"] is True


def test_ollama_does_not_audit_discarded_reasoning_effort(mock_native_clients):
    from msgflux.models.providers.ollama import OllamaChatCompletion

    with pytest.warns(UserWarning, match="does not support.*reasoning_effort"):
        model = OllamaChatCompletion(
            model_id="gpt-oss:20b",
            reasoning_effort="medium",
        )

    assert "reasoning_effort" not in model._build_response_metadata(None).model


def test_ollama_warns_and_ignores_runtime_reasoning_effort(mock_native_clients):
    from msgflux.models.providers.ollama import OllamaChatCompletion

    model = OllamaChatCompletion(model_id="gpt-oss:20b")

    with pytest.warns(UserWarning, match="does not support.*reasoning_effort"):
        result = model.set_reasoning_effort("high")

    assert result is model
    assert "reasoning_effort" not in model.sampling_run_params
