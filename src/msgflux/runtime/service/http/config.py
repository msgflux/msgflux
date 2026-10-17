"""Dependency-free HTTP policy for embedded and local AgentService hosts."""

from urllib.parse import urlsplit

import msgspec


class ServiceHTTPConfig(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """Allow explicit browser origins or a standalone wildcard; empty disables CORS."""

    cors_allow_origins: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if "*" in self.cors_allow_origins:
            if len(self.cors_allow_origins) != 1:
                raise ValueError("The CORS wildcard must be the only allowed origin")
            return
        for origin in self.cors_allow_origins:
            if not isinstance(origin, str):
                raise ValueError("CORS origins must be HTTP(S) origin strings")
            try:
                parsed = urlsplit(origin)
                port = parsed.port
            except ValueError as error:
                raise ValueError(f"Invalid CORS origin: {origin!r}") from error
            if (
                parsed.scheme not in {"http", "https"}
                or not parsed.hostname
                or parsed.username is not None
                or parsed.password is not None
                or parsed.path
                or parsed.query
                or parsed.fragment
                or "*" in origin
                or any(character.isspace() for character in origin)
                or (port is not None and port == 0)
            ):
                raise ValueError(
                    f"Invalid CORS origin {origin!r}; use an exact HTTP(S) origin "
                    "such as http://localhost:3000"
                )
