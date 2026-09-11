from __future__ import annotations

import pytest

from msgflux.models.multipart import (
    aprepare_multipart_file,
    prepare_multipart_file,
    prepare_multipart_data,
)


def test_prepare_multipart_file_preserves_name_content_and_type(tmp_path):
    path = tmp_path / "sample.wav"
    path.write_bytes(b"audio")

    assert prepare_multipart_file(
        str(path),
        default_filename="audio.wav",
    ) == ("sample.wav", b"audio", "audio/x-wav")


@pytest.mark.asyncio
async def test_aprepare_multipart_file_uses_default_for_raw_bytes():
    assert await aprepare_multipart_file(
        b"audio",
        default_filename="audio.wav",
    ) == ("audio.wav", b"audio", "audio/x-wav")


def test_prepare_multipart_data_encodes_scalars_and_repeated_values():
    assert prepare_multipart_data(
        {
            "model": "gpt-4o-transcribe",
            "stream": True,
            "temperature": 0,
            "timestamp_granularities": ["word", "segment"],
            "prompt": None,
        }
    ) == {
        "model": "gpt-4o-transcribe",
        "stream": "true",
        "temperature": "0",
        "timestamp_granularities[]": ["word", "segment"],
    }
