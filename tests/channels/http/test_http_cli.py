from pathlib import Path

import pytest

from msgflux.channels.http import cli


def test_resolve_server_target_local_path_unchanged():
    target = "server.py:registry"
    resolved = cli._resolve_server_target(target, trust_remote_code=False)
    assert resolved == target


def test_resolve_server_target_remote_requires_trust():
    with pytest.raises(ValueError, match="--trust-remote-code"):
        cli._resolve_server_target(
            "https://example.com/server.py",
            trust_remote_code=False,
        )


def test_resolve_server_target_remote_downloads_when_trusted(monkeypatch, tmp_path):
    expected = tmp_path / "downloaded.py"
    expected.write_text("registry = object()", encoding="utf-8")

    monkeypatch.setattr(cli, "_download_remote_target", lambda _url: expected)

    resolved = cli._resolve_server_target(
        "https://example.com/server.py",
        trust_remote_code=True,
    )

    assert resolved == str(expected)


def test_resolve_server_target_remote_preserves_registry_attr(monkeypatch, tmp_path):
    expected = tmp_path / "downloaded.py"
    expected.write_text("registry = object()", encoding="utf-8")

    monkeypatch.setattr(cli, "_download_remote_target", lambda _url: expected)

    resolved = cli._resolve_server_target(
        "https://example.com/server.py:custom_registry",
        trust_remote_code=True,
    )

    assert resolved == f"{expected}:custom_registry"


def test_download_remote_target_writes_cached_file(monkeypatch, tmp_path):
    class FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def read(self, _n):
            return b"from msgflux.channels import ChannelRegistry\nregistry = ChannelRegistry()\n"

    monkeypatch.setattr(cli, "_CACHE_DIR", tmp_path)
    monkeypatch.setattr(cli, "urlopen", lambda _url, timeout: FakeResponse())

    path = cli._download_remote_target("https://example.com/server.py")

    assert isinstance(path, Path)
    assert path.exists()
    assert path.read_text(encoding="utf-8").startswith("from msgflux.channels")
