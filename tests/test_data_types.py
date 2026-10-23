"""
Tests for multimodal data types (Image, Audio, Video, File).

These tests validate that the dtype classes correctly process files/URLs
and return properly formatted ChatBlocks.
"""

import base64
import tempfile
from pathlib import Path

import pytest

from msgflux.data.types import Audio, File, Image, Video


class TestImage:
    """Tests for Image dtype."""

    def test_image_from_url_no_encode(self):
        """Test that URLs are kept as-is when force_encode=False."""
        img = Image("https://example.com/photo.jpg")
        result = img()

        assert result["type"] == "image_url"
        assert result["image_url"]["url"] == "https://example.com/photo.jpg"

    def test_image_with_kwargs(self):
        """Test that kwargs are passed to ChatBlock."""
        img = Image("https://example.com/photo.jpg", detail="high")
        result = img()

        assert result["image_url"]["detail"] == "high"

    def test_image_from_local_file(self):
        """Test that local files are encoded to base64."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"\xff\xd8\xff")  # JPEG magic bytes
            temp_path = f.name

        try:
            img = Image(temp_path)
            result = img()

            assert result["type"] == "image_url"
            assert result["image_url"]["url"].startswith("data:image/")
        finally:
            Path(temp_path).unlink()

    def test_image_caches_result(self):
        """Test that result is cached after first call."""
        img = Image("https://example.com/photo.jpg")

        result1 = img()
        result2 = img()

        assert result1 is result2

    @pytest.mark.asyncio
    async def test_image_async(self):
        """Test async processing."""
        img = Image("https://example.com/photo.jpg")
        result = await img.acall()

        assert result["type"] == "image_url"
        assert result["image_url"]["url"] == "https://example.com/photo.jpg"


class TestAudio:
    """Tests for Audio dtype."""

    def test_audio_from_local_file(self):
        """Test that audio files are encoded to base64."""
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(b"ID3\x04\x00\x00")  # MP3 magic bytes
            temp_path = f.name

        try:
            aud = Audio(temp_path)
            result = aud()

            assert result["type"] == "input_audio"
            assert "data" in result["input_audio"]
            assert result["input_audio"]["format"] == "mp3"
        finally:
            Path(temp_path).unlink()

    def test_audio_format_detection(self):
        """Test that audio format is detected from extension."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"RIFF\x00\x00\x00\x00WAVE")  # WAV header
            temp_path = f.name

        try:
            aud = Audio(temp_path)
            result = aud()

            assert result["input_audio"]["format"] == "wav"
        finally:
            Path(temp_path).unlink()

    @pytest.mark.asyncio
    async def test_audio_async(self):
        """Test async processing."""
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(b"ID3\x04\x00\x00")
            temp_path = f.name

        try:
            aud = Audio(temp_path)
            result = await aud.acall()

            assert result["type"] == "input_audio"
        finally:
            Path(temp_path).unlink()


class TestVideo:
    """Tests for Video dtype."""

    def test_video_from_url_no_encode(self):
        """Test that URLs are kept as-is when force_encode=False."""
        vid = Video("https://example.com/video.mp4", force_encode=False)
        result = vid()

        assert result["type"] == "video_url"
        assert result["video_url"]["url"] == "https://example.com/video.mp4"

    def test_video_with_kwargs(self):
        """Test that kwargs are passed to ChatBlock."""
        vid = Video("https://example.com/video.mp4", force_encode=False, format="mp4")
        result = vid()

        assert result["video_url"]["format"] == "mp4"

    def test_video_from_local_file(self):
        """Test that local files are encoded to base64."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"\x00\x00\x00\x1c")  # MP4 header bytes
            temp_path = f.name

        try:
            vid = Video(temp_path)
            result = vid()

            assert result["type"] == "video_url"
            assert result["video_url"]["url"].startswith("data:video/")
        finally:
            Path(temp_path).unlink()

    @pytest.mark.asyncio
    async def test_video_async(self):
        """Test async processing."""
        vid = Video("https://example.com/video.mp4", force_encode=False)
        result = await vid.acall()

        assert result["type"] == "video_url"


class TestFile:
    """Tests for File dtype."""

    def test_file_from_local_file(self):
        """Test that files are encoded to base64."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"%PDF-1.4")  # PDF magic bytes
            temp_path = f.name

        try:
            doc = File(temp_path)
            result = doc()

            assert result["type"] == "file"
            assert "filename" in result["file"]
            assert result["file"]["filename"].endswith(".pdf")
            assert result["file"]["file_data"].startswith("data:application/pdf")
        finally:
            Path(temp_path).unlink()

    def test_file_filename_extraction(self):
        """Test that filename is correctly extracted."""
        with tempfile.NamedTemporaryFile(
            suffix=".pdf", delete=False, prefix="test_doc_"
        ) as f:
            f.write(b"%PDF-1.4")
            temp_path = f.name

        try:
            doc = File(temp_path)
            result = doc()

            assert "test_doc_" in result["file"]["filename"]
        finally:
            Path(temp_path).unlink()

    @pytest.mark.asyncio
    async def test_file_async(self):
        """Test async processing."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"%PDF-1.4")
            temp_path = f.name

        try:
            doc = File(temp_path)
            result = await doc.acall()

            assert result["type"] == "file"
        finally:
            Path(temp_path).unlink()


MEDIA_CASES = [
    (Image, ".png", lambda block: block["image_url"]["url"]),
    (Audio, ".mp3", lambda block: block["input_audio"]["data"]),
    (Video, ".mp4", lambda block: block["video_url"]["url"]),
    (File, ".pdf", lambda block: block["file"]["file_data"]),
]


@pytest.mark.parametrize(("media_cls", "suffix", "extract"), MEDIA_CASES)
def test_sync_encode_reflects_current_file_contents(
    media_cls, suffix, extract, tmp_path
):
    """A new media object reads the current file contents, not a stale value."""
    path = tmp_path / f"media{suffix}"
    path.write_bytes(b"first version")

    first = extract(media_cls(str(path))())
    path.write_bytes(b"second version")
    second = extract(media_cls(str(path))())

    assert first.endswith(base64.b64encode(b"first version").decode())
    assert second.endswith(base64.b64encode(b"second version").decode())


@pytest.mark.asyncio
@pytest.mark.parametrize(("media_cls", "suffix", "extract"), MEDIA_CASES)
async def test_async_encode_reflects_current_file_contents(
    media_cls, suffix, extract, tmp_path
):
    """Async processing keeps the same freshness contract as sync."""
    path = tmp_path / f"media{suffix}"
    path.write_bytes(b"first version")

    first = extract(await media_cls(str(path)).acall())
    path.write_bytes(b"second version")
    second = extract(await media_cls(str(path)).acall())

    assert first.endswith(base64.b64encode(b"first version").decode())
    assert second.endswith(base64.b64encode(b"second version").decode())


class TestMediaTypeBase:
    """Tests for MediaType base class behavior."""

    def test_is_url_detection(self):
        """Test that URLs are correctly detected."""
        img_url = Image("https://example.com/photo.jpg")
        img_local = Image("/path/to/photo.jpg")

        assert img_url._is_url(img_url.source)
        assert not img_local._is_url(img_local.source)

    def test_force_encode_parameter(self):
        """Test that force_encode works correctly."""
        # Default for Image is False
        img = Image("https://example.com/photo.jpg")
        assert img.force_encode is False

        # Default for Audio is True
        aud = Audio("/path/to/audio.mp3")
        assert aud.force_encode is True

        # Override default
        img_forced = Image("https://example.com/photo.jpg", force_encode=True)
        assert img_forced.force_encode is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
