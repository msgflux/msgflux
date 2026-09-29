"""System notices remain ordered while using OpenAI Responses developer roles."""

from copy import deepcopy

import pytest

from msgflux.chat_messages import ChatMessages
from msgflux.models import Model


@pytest.fixture(autouse=True)
def _test_credentials(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")


def _model(provider, tmp_path, **kwargs):
    if provider == "openai-codex":
        auth_file = tmp_path / "auth.json"
        auth_file.write_text(
            '{"tokens":{"access_token":"test-token","account_id":"test-account"}}'
        )
        kwargs["auth_file"] = auth_file
    model_id = "openai/gpt-6-luna" if provider == "openrouter" else "gpt-6-luna"
    return Model.chat_completion(f"{provider}/{model_id}", **kwargs)


def _history():
    return ChatMessages(
        [
            {"role": "user", "content": "Run Bash in background"},
            {
                "type": "function_call",
                "call_id": "background-call",
                "name": "bash",
                "arguments": '{"run_in_background":true}',
            },
            {
                "type": "function_call_output",
                "call_id": "background-call",
                "output": "Task started",
            },
            {"role": "system", "content": "Task completed: background-ok"},
            {"role": "developer", "content": "Keep the existing developer message"},
        ]
    )


@pytest.mark.parametrize(
    "provider",
    ["openai", "openai-codex", "openrouter", "custom-router", "custom-openai"],
)
def test_responses_translates_notices_in_place_and_keeps_canonical_history(
    provider, tmp_path
):
    route = {"custom-router": "openrouter", "custom-openai": "openai"}.get(
        provider, provider
    )
    model = _model(route, tmp_path, api_mode="responses")
    model.provider = provider
    history = _history()
    original = deepcopy(history.to_items())
    params = model._build_generation_params(
        history, system_prompt="Stable instructions", prefilling=None, tool_catalog=None
    )
    request = model.api_adapter.prepare_request(model, params)
    items = request.json["input"]
    if provider == "openai-codex":
        assert request.json["instructions"] == "Stable instructions"
    else:
        assert items[0]["role"] == "developer"
        assert items[0]["content"] == "Stable instructions"
        items = items[1:]
    assert [item.get("role") or item["type"] for item in items] == [
        "user",
        "function_call",
        "function_call_output",
        "developer",
        "developer",
    ]
    assert request.json["input"][-2]["content"] == "Task completed: background-ok"
    assert request.json["input"][-1]["content"] == "Keep the existing developer message"
    assert history.to_items() == original


@pytest.mark.parametrize("provider", ["openai", "openai-codex"])
@pytest.mark.parametrize("operation", ["prepare_token_count", "prepare_compaction"])
def test_context_operations_use_same_roles(provider, operation, tmp_path):
    model = _model(provider, tmp_path)
    history = _history()
    adapter = model.api_mode_capabilities.context_adapter
    request = getattr(adapter, operation)(
        model, history, system_prompt="Stable instructions"
    )
    assert request.json["instructions"] == "Stable instructions"
    assert request.json["input"][-2]["role"] == "developer"
    assert history.to_items()[-2]["role"] == "system"


@pytest.mark.parametrize(
    "provider, mode", [("openai", "chat_completions"), ("openrouter", "responses")]
)
def test_other_transports_keep_system_roles(provider, mode, tmp_path):
    if provider == "openrouter":
        model = Model.chat_completion(
            "openrouter/anthropic/claude-opus-4.6", api_mode=mode
        )
    else:
        model = _model(provider, tmp_path, api_mode=mode)
    params = model._build_generation_params(
        _history(),
        system_prompt="Stable instructions",
        prefilling=None,
        tool_catalog=None,
    )
    request = model.api_adapter.prepare_request(model, params)
    items = request.json["input" if mode == "responses" else "messages"]
    assert items[-2]["role"] == "system"


def test_openrouter_chat_completions_preserves_system_role(tmp_path):
    model = _model("openrouter", tmp_path, api_mode="chat_completions")
    params = model._build_generation_params(
        _history(),
        system_prompt="Stable instructions",
        prefilling=None,
        tool_catalog=None,
    )
    request = model.api_adapter.prepare_request(model, params)
    assert request.json["messages"][-2]["role"] == "system"
