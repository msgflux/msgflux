import pytest

import msgflux as mf
from msgflux.models.providers.fake import (
    FakeChatCompletion,
    FakeImageEmbedder,
    FakeImageTextToImage,
    FakeModelExecutionError,
    FakeModeration,
    FakeSpeechToText,
    FakeTextEmbedder,
    FakeTextReranker,
    FakeTextToImage,
    FakeTextToSpeech,
)
from msgflux.models.registry import register_model


@pytest.fixture(autouse=True)
def register_fake_models():
    for model_cls in (
        FakeChatCompletion,
        FakeImageEmbedder,
        FakeImageTextToImage,
        FakeModeration,
        FakeSpeechToText,
        FakeTextEmbedder,
        FakeTextReranker,
        FakeTextToImage,
        FakeTextToSpeech,
    ):
        register_model(model_cls)


@pytest.mark.parametrize(
    ("factory", "model_type", "class_name"),
    [
        (mf.Model.chat_completion, "chat_completion", "FakeChatCompletion"),
        (mf.Model.image_embedder, "image_embedder", "FakeImageEmbedder"),
        (mf.Model.image_text_to_image, "image_text_to_image", "FakeImageTextToImage"),
        (mf.Model.moderation, "moderation", "FakeModeration"),
        (mf.Model.speech_to_text, "speech_to_text", "FakeSpeechToText"),
        (mf.Model.text_embedder, "text_embedder", "FakeTextEmbedder"),
        (mf.Model.text_reranker, "text_reranker", "FakeTextReranker"),
        (mf.Model.text_to_image, "text_to_image", "FakeTextToImage"),
        (mf.Model.text_to_speech, "text_to_speech", "FakeTextToSpeech"),
    ],
)
def test_fake_models_serialize_and_never_execute(factory, model_type, class_name):
    model = factory("fake/placeholder", reason="test")

    serialized = model.serialize()

    assert model.__class__.__name__ == class_name
    assert serialized["msgflux_type"] == "model"
    assert serialized["provider"] == "fake"
    assert serialized["model_type"] == model_type
    assert serialized["state"]["model_id"] == "placeholder"
    assert serialized["state"]["kwargs"] == {"reason": "test"}

    with pytest.raises(
        FakeModelExecutionError,
        match=f"{class_name} cannot execute requests",
    ):
        model()
