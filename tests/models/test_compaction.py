from types import SimpleNamespace

import pytest

from msgflux.chat_messages import ChatMessages
from msgflux.models.response import ModelResponse
from msgflux.models.types import ChatCompletionModel


class PortableCompactionModel(ChatCompletionModel):
    provider = "portable"
    api_mode = "messages"
    model_id = "summary-model"

    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        response = ModelResponse()
        response.set_response_type("text_generation")
        response.add("Keep decision A and unresolved action B.")
        response.metadata = {"usage": {"input_tokens": 50, "output_tokens": 10}}
        return response

    async def acall(self, **kwargs):
        return self(**kwargs)


def test_context_capacity_prefers_explicit_configuration_over_profile():
    model = PortableCompactionModel()
    model.context_length = 200_000
    model.profile = SimpleNamespace(limits=SimpleNamespace(context=100_000))

    assert model.context_capacity == 200_000


def test_context_capacity_uses_profile_and_treats_zero_as_unknown():
    model = PortableCompactionModel()
    model.profile = SimpleNamespace(limits=SimpleNamespace(context=100_000))
    assert model.context_capacity == 100_000

    model.profile = SimpleNamespace(limits=SimpleNamespace(context=0))
    assert model.context_capacity is None


def test_generic_compaction_returns_complete_portable_summary_view():
    model = PortableCompactionModel()
    messages = ChatMessages([{"role": "user", "content": "Long history"}])

    compacted = model.compact_context(messages, system_prompt="Follow policy X.")

    assert compacted.format == "messages"
    assert compacted.items == [
        {
            "role": "system",
            "content": (
                "<conversation_summary>\n"
                "Keep decision A and unresolved action B.\n"
                "</conversation_summary>"
            ),
        }
    ]
    assert compacted.usage == {"input_tokens": 50, "output_tokens": 10}
    assert model.calls[0]["messages"] is messages
    assert model.calls[0]["stream"] is False
    assert (
        "Original agent instructions:\nFollow policy X."
        in model.calls[0]["system_prompt"]
    )
    assert "tool_catalog" not in model.calls[0]


def test_generic_token_estimate_counts_materialized_view_not_canonical_history():
    model = PortableCompactionModel()
    messages = ChatMessages()
    messages.begin_turn()
    messages.add_user("x" * 3_000)
    messages.add_assistant("y" * 3_000)
    messages.end_turn()
    boundary = messages.latest_completed_turn_boundary()["item_id"]
    before = model.count_context_tokens(messages)
    messages.add_compaction(
        compacted_through_item_id=boundary,
        views=[
            {
                "format": "messages",
                "items": [{"role": "system", "content": "short summary"}],
            }
        ],
    )

    after = model.count_context_tokens(messages)

    assert after.input_tokens < before.input_tokens


@pytest.mark.asyncio
async def test_generic_async_compaction_uses_same_portable_contract():
    model = PortableCompactionModel()

    compacted = await model.acompact_context(
        ChatMessages([{"role": "user", "content": "History"}])
    )

    assert compacted.format == "messages"
    assert compacted.model_id == "summary-model"


def test_generic_compaction_rejects_empty_or_non_text_summary():
    class EmptyCompactionModel(PortableCompactionModel):
        def __call__(self, **_kwargs):
            return ModelResponse()

    with pytest.raises(TypeError, match="non-empty text"):
        EmptyCompactionModel().compact_context([])
