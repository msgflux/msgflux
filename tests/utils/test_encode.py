import base64
import io
import os

import httpx2
import pytest

from msgflux.utils.encode import (
    aencode_data_to_base64,
    aencode_data_to_bytes,
    encode_base64_from_url,
    encode_data_to_base64,
    encode_data_to_bytes,
    encode_local_file_in_base64,
    encode_to_io_object,
)


@pytest.fixture
def mock_httpx_get(mocker):
    mock_response = mocker.MagicMock()
    mock_response.status_code = 200
    mock_response.content = b"hello world"
    mock_response.raise_for_status.return_value = None
    mock_response.url = "http://example.com/test.txt"
    mocker.patch("httpx2.get", return_value=mock_response)


@pytest.fixture
def temp_file(tmp_path):
    file_path = tmp_path / "test.txt"
    file_path.write_text("hello world")
    return str(file_path)


def test_encode_base64_from_url(mock_httpx_get):
    encoded = encode_base64_from_url("http://example.com")
    assert base64.b64decode(encoded) == b"hello world"


def test_encode_local_file_in_base64(temp_file):
    encoded = encode_local_file_in_base64(temp_file)
    assert base64.b64decode(encoded) == b"hello world"


def test_encode_data_to_base64(mock_httpx_get, temp_file):
    assert encode_data_to_base64("not a path or url") == "not a path or url"
    assert base64.b64decode(encode_data_to_base64(temp_file)) == b"hello world"
    assert (
        base64.b64decode(encode_data_to_base64("http://example.com")) == b"hello world"
    )


def test_encode_data_to_base64_keeps_local_path_with_http_in_name(tmp_path, mocker):
    file_path = tmp_path / "http-notes.txt"
    file_path.write_text("hello world")
    http_get = mocker.patch(
        "httpx2.get",
        side_effect=AssertionError("local path must not be fetched over HTTP"),
    )

    encoded = encode_data_to_base64(str(file_path))

    http_get.assert_not_called()
    assert base64.b64decode(encoded) == b"hello world"


@pytest.mark.asyncio
async def test_aencode_data_to_base64_keeps_local_path_with_http_in_name(
    tmp_path, monkeypatch
):
    file_path = tmp_path / "http-notes.txt"
    file_path.write_text("hello world")

    def fail_async_client(*args, **kwargs):
        raise AssertionError("local path must not be fetched over HTTP")

    monkeypatch.setattr(httpx2, "AsyncClient", fail_async_client)

    encoded = await aencode_data_to_base64(str(file_path))

    assert base64.b64decode(encoded) == b"hello world"


def test_encode_to_io_object(mock_httpx_get, temp_file):
    assert isinstance(encode_to_io_object(b"hello"), io.BytesIO)
    assert isinstance(encode_to_io_object(temp_file), io.IOBase)
    assert isinstance(encode_to_io_object("http://example.com"), io.BytesIO)
    b64_string = base64.b64encode(b"hello").decode()
    assert isinstance(encode_to_io_object(b64_string), io.BytesIO)
    with pytest.raises(ValueError):
        encode_to_io_object("not a valid input")


def test_encode_data_to_bytes(mock_httpx_get, temp_file):
    assert isinstance(encode_data_to_bytes(b"hello"), io.BytesIO)
    assert isinstance(encode_data_to_bytes(temp_file), io.BytesIO)
    assert isinstance(encode_data_to_bytes("http://example.com"), io.BytesIO)
    b64_string = base64.b64encode(b"hello").decode()
    assert isinstance(encode_data_to_bytes(b64_string), io.BytesIO)
    with pytest.raises(ValueError):
        encode_data_to_bytes("not a valid input")


def test_encode_data_to_bytes_preserves_fallback_filename():
    buffer = encode_data_to_bytes(b"audio", filename="audio.wav")

    assert buffer.name == "audio.wav"


def test_encode_data_to_bytes_accepts_base64_data_url():
    value = "data:image/png;base64," + base64.b64encode(b"image").decode()

    buffer = encode_data_to_bytes(value)

    assert buffer.name == "upload.png"
    assert buffer.read() == b"image"


@pytest.mark.asyncio
async def test_aencode_data_to_bytes_downloads_without_blocking(monkeypatch):
    async def handler(request):
        return httpx2.Response(200, content=b"audio", request=request)

    client = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    monkeypatch.setattr(httpx2, "AsyncClient", lambda **kwargs: client)

    buffer = await aencode_data_to_bytes(
        "https://example.com/sample.wav",
        filename="audio.wav",
    )

    assert buffer.name == "sample.wav"
    assert buffer.read() == b"audio"
