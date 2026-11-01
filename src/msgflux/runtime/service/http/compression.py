"""Negotiated response compression using Litestar's compression interfaces."""

from __future__ import annotations

import sys
from contextlib import contextmanager
from importlib import import_module
from io import BytesIO
from typing import TYPE_CHECKING

from litestar.config.compression import CompressionConfig
from litestar.datastructures import MutableScopeHeaders
from litestar.middleware.compression import CompressionMiddleware
from litestar.middleware.compression.facade import CompressionFacade
from litestar.middleware.compression.gzip_facade import GzipCompression

if TYPE_CHECKING:
    from litestar.types import Receive, Scope, Send


def _zstd_module():
    try:
        return import_module(
            "compression.zstd" if sys.version_info >= (3, 14) else "backports.zstd"
        )
    except ImportError as exc:
        raise ImportError(
            "HTTP compression requires zstd support; install msgflux[service]."
        ) from exc


class _ZstdCompression(CompressionFacade):
    """Flush each write while keeping one zstd frame for the response."""

    encoding = "zstd"

    def __init__(
        self,
        buffer: BytesIO,
        compression_encoding: str,
        config: CompressionConfig,
    ) -> None:
        self.buffer = buffer
        self.compression_encoding = compression_encoding
        self._compressor = _zstd_module().ZstdCompressor(level=config.backend_config)

    def write(self, body: bytes) -> None:
        compressor = self._compressor
        if compressor is None:
            raise ValueError("Compression stream is closed")
        self.buffer.write(compressor.compress(body, compressor.FLUSH_BLOCK))

    def close(self) -> None:
        compressor = self._compressor
        if compressor is not None:
            self._compressor = None
            self.buffer.write(compressor.flush(compressor.FLUSH_FRAME))


def _select_encoding(value: str) -> tuple[str | None, bool]:
    """Select an offered encoding or identity, respecting explicit exclusions."""
    if not value.strip():
        return None, True
    qualities: dict[str, float] = {}
    for item in value.split(","):
        coding, _, parameters = item.partition(";")
        coding = coding.strip().lower()
        if not coding:
            continue
        quality = 1.0
        for parameter in parameters.split(";"):
            key, separator, raw = parameter.partition("=")
            if separator and key.strip().lower() == "q":
                try:
                    quality = float(raw.strip())
                except ValueError:
                    quality = 0.0
        qualities[coding] = quality if 0 <= quality <= 1 else 0.0
    wildcard = qualities.get("*", 0.0)
    offered = [(coding, qualities.get(coding, wildcard)) for coding in ("zstd", "gzip")]
    # An implicit identity fallback does not displace an accepted coding;
    # an explicit identity preference participates in quality selection.
    identity = qualities.get("identity")
    coding, quality = max(offered, key=lambda item: item[1])
    identity_allowed = (identity is not None and identity > 0) or (
        identity is None and qualities.get("*") != 0
    )
    if quality > 0 and (identity is None or quality >= identity):
        return coding, identity_allowed
    if identity_allowed:
        return None, True
    raise ValueError("No acceptable response encoding")


@contextmanager
def _compression_sender(*, send, encoding, config, identity_allowed):
    """Own the facade until send/cancellation finishes, then close the buffer.

    Litestar's send-wrapper has no cancellation cleanup hook. Retain its
    facade interface and header helpers while explicitly owning this lifetime.
    """
    buffer = BytesIO()
    facade_type = GzipCompression if encoding == "gzip" else config.compression_facade
    facade = facade_type(buffer=buffer, compression_encoding=encoding, config=config)
    initial = None
    started = False
    passthrough = False

    async def compressed_send(message):
        nonlocal initial, started, passthrough
        if message["type"] == "http.response.start":
            initial = message
            headers = MutableScopeHeaders(initial)
            headers.extend_header_value("vary", "Accept-Encoding")
            passthrough = "content-encoding" in headers
            return
        if message["type"] != "http.response.body":
            await send(message)
            return
        body = message.get("body", b"")
        more = message.get("more_body", False)
        if not started:
            started = True
            passthrough = passthrough or (
                identity_allowed and not more and len(body) < config.minimum_size
            )
        if passthrough:
            if initial is not None:
                await send(initial)
                initial = None
            await send(message)
            return
        facade.write(body)
        if not more:
            facade.close()
        encoded = buffer.getvalue()
        buffer.seek(0)
        buffer.truncate()
        if initial is not None:
            headers = MutableScopeHeaders(initial)
            headers["content-encoding"] = encoding
            if more:
                del headers["content-length"]
            else:
                headers["content-length"] = str(len(encoded))
            await send(initial)
            initial = None
        await send({**message, "body": encoded})

    try:
        yield compressed_send
    finally:
        try:
            facade.close()
        finally:
            buffer.close()


class _NegotiatedCompressionMiddleware(CompressionMiddleware):
    """Negotiate encodings and scope framework facades to each response."""

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        value = ",".join(
            value.decode("latin-1")
            for name, value in scope.get("headers", ())
            if name.lower() == b"accept-encoding"
        )
        try:
            encoding, identity_allowed = _select_encoding(value)
        except ValueError:
            await send(
                {
                    "type": "http.response.start",
                    "status": 406,
                    "headers": [
                        (b"content-length", b"0"),
                        (b"vary", b"Accept-Encoding"),
                    ],
                }
            )
            await send({"type": "http.response.body", "body": b""})
            return
        if encoding is not None:
            with _compression_sender(
                send=send,
                encoding=encoding,
                config=self.config,
                identity_allowed=identity_allowed,
            ) as wrapped:
                await self.app(scope, receive, wrapped)
            return

        async def send_identity(message):
            if message["type"] == "http.response.start":
                MutableScopeHeaders(message).extend_header_value(
                    "vary", "Accept-Encoding"
                )
            await send(message)

        await self.app(scope, receive, send_identity)


def service_compression_config(*, enabled: bool) -> CompressionConfig | None:
    """Build the framework policy unless the host disables compression."""
    if not enabled:
        return None
    _zstd_module()
    return CompressionConfig(
        backend="zstd",
        compression_facade=_ZstdCompression,
        middleware_class=_NegotiatedCompressionMiddleware,
        backend_config=1,
        gzip_compress_level=1,
    )
