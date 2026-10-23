import base64

from msgflux.utils.validation import (
    is_base64,
    is_builtin_type,
    is_http_url,
    is_subclass_of,
)


class A:
    pass


class B(A):  # B is a subclass of A
    pass


def test_is_subclass_of():
    assert is_subclass_of(B, A)
    assert not is_subclass_of(A, B)
    assert not is_subclass_of("not a class", A)


def test_is_builtin_type():
    assert is_builtin_type(1)
    assert is_builtin_type("hello")
    assert is_builtin_type([1, 2])
    assert not is_builtin_type(A())


def test_is_base64():
    valid_b64 = base64.b64encode(b"hello").decode()
    assert is_base64(valid_b64)
    assert not is_base64("not base64")


def test_is_http_url():
    assert is_http_url("http://example.com/file.txt")
    assert is_http_url("https://example.com/file.txt")
    assert is_http_url(b"https://example.com/file.txt")
    assert not is_http_url("http-photo.jpg")
    assert not is_http_url("data/http-photo.jpg")
    assert not is_http_url("http")
    assert not is_http_url(b"http-photo.jpg")
    assert not is_http_url(None)
