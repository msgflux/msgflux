import pytest

from msgflux.models.usage import UsageCodec


def test_usage_codec_normalizes_chat_completions():
    raw = {
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "total_tokens": 120,
        "prompt_tokens_details": {
            "cached_tokens": 80,
            "cache_write_tokens": 10,
        },
        "completion_tokens_details": {"reasoning_tokens": 12},
        "cost": 0.01,
    }

    usage = UsageCodec().normalize(raw)

    assert usage.input_tokens == 100
    assert usage.output_tokens == 20
    assert usage.total_tokens == 120
    assert usage.cache_hit_percentage == 80.0
    assert usage.input_tokens_details.cached_tokens == 80
    assert usage.input_tokens_details.cache_write_tokens == 10
    assert usage.output_tokens_details.reasoning_tokens == 12
    assert usage.cost == 0.01
    assert usage.raw == raw
    assert usage.raw is not raw


def test_usage_codec_normalizes_responses_and_calculates_total():
    usage = UsageCodec().normalize(
        {
            "input_tokens": 40,
            "output_tokens": 8,
            "input_tokens_details": {"cached_tokens": 32},
            "output_tokens_details": {"reasoning_tokens": 5},
        }
    )

    assert usage.input_tokens == 40
    assert usage.output_tokens == 8
    assert usage.total_tokens == 48
    assert usage.cache_hit_percentage == 80.0
    assert usage.input_tokens_details.cached_tokens == 32
    assert usage.output_tokens_details.reasoning_tokens == 5


def test_usage_codec_preserves_image_token_details():
    usage = UsageCodec().normalize(
        {
            "input_tokens": 12,
            "output_tokens": 20,
            "total_tokens": 32,
            "input_tokens_details": {"text_tokens": 4, "image_tokens": 8},
            "output_tokens_details": {"text_tokens": 0, "image_tokens": 20},
        }
    )

    assert usage.input_tokens_details.text_tokens == 4
    assert usage.input_tokens_details.image_tokens == 8
    assert usage.output_tokens_details.text_tokens == 0
    assert usage.output_tokens_details.image_tokens == 20


def test_usage_codec_normalizes_transcription_details_and_duration():
    token_usage = UsageCodec().normalize(
        {
            "input_tokens": 14,
            "input_token_details": {"audio_tokens": 12, "text_tokens": 2},
            "output_tokens": 3,
            "total_tokens": 17,
        }
    )
    duration_usage = UsageCodec().normalize({"type": "duration", "seconds": 9})

    assert token_usage.input_tokens_details.audio_tokens == 12
    assert token_usage.input_tokens_details.text_tokens == 2
    assert duration_usage.duration_seconds == 9


def test_usage_codec_normalizes_anthropic_and_google_aliases():
    anthropic = UsageCodec().normalize(
        {
            "input_tokens": 50,
            "output_tokens": 10,
            "cache_read_input_tokens": 30,
            "cache_creation_input_tokens": 20,
        }
    )
    google = UsageCodec().normalize(
        {
            "prompt_token_count": 70,
            "candidates_token_count": 9,
            "total_token_count": 85,
            "cached_content_token_count": 60,
            "thoughts_token_count": 6,
        }
    )

    assert anthropic.input_tokens_details.cached_tokens == 30
    assert anthropic.cache_hit_percentage == 60.0
    assert anthropic.input_tokens_details.cache_write_tokens == 20
    assert google.input_tokens == 70
    assert google.output_tokens == 9
    assert google.total_tokens == 85
    assert google.input_tokens_details.cached_tokens == 60
    assert google.cache_hit_percentage == pytest.approx(85.7142857143)
    assert google.output_tokens_details.reasoning_tokens == 6


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ({"input_tokens": 100}, 0.0),
        ({"output_tokens": 10}, None),
        ({"input_tokens": 0, "input_tokens_details": {"cached_tokens": 0}}, None),
        (
            {"input_tokens": 10, "input_tokens_details": {"cached_tokens": 11}},
            None,
        ),
        (
            {"input_tokens": 10, "input_tokens_details": {"cached_tokens": -1}},
            None,
        ),
    ],
)
def test_usage_codec_calculates_valid_cache_hit_percentage(raw, expected):
    usage = UsageCodec().normalize(raw)

    assert usage.cache_hit_percentage == expected
