from __future__ import annotations

import asyncio
import base64
import multiprocessing
import json
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from msgflux.accounts import Accounts
from msgflux.accounts import oauth


def jwt(exp: int) -> str:
    claims = {
        "exp": exp,
        "https://api.openai.com/auth": {"chatgpt_account_id": "acct-1"},
    }
    payload = base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip("=")
    return f"header.{payload}.signature"


def native_payload(exp: int, *, refresh: str = "refresh-1") -> dict:
    return {
        "auth_mode": "chatgpt",
        "tokens": {
            "access_token": jwt(exp),
            "refresh_token": refresh,
            "account_id": "acct-1",
        },
        "last_refresh": "2026-01-01T00:00:00Z",
    }


def _multiprocess_refresh(root: str, endpoint: str, barrier, queue) -> None:
    oauth.OAUTH_ENDPOINT = endpoint
    try:
        barrier.wait(timeout=10)
        result = oauth.resolve_codex_credentials(Accounts(root=root), "work")
        queue.put(("ok", result.account_id))
    except Exception as exc:
        queue.put(("error", str(exc)))


class _RefreshHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", "0")))
        self.server.refresh_requests += 1
        time.sleep(0.2)
        body = json.dumps(
            {
                "access_token": jwt(int(time.time()) + 3600),
                "refresh_token": "refresh-from-http-server",
                "expires_in": 3600,
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        return


class MemoryAccounts:
    def __init__(self, payload: dict | None = None, tmp_path: Path | None = None):
        self.payload = payload
        self.tmp_path = tmp_path or Path(".")
        self.lock = threading.Lock()
        self.writes = 0

    def account_dir(self, _name: str) -> Path:
        self.tmp_path.mkdir(parents=True, exist_ok=True)
        return self.tmp_path

    @contextmanager
    def locked(self, _provider: str, *, account_name: str):
        with self.lock:
            yield

    def get_oauth(self, _provider: str, *, account_name: str):
        if self.payload is None:
            raise ValueError("missing credentials")
        return SimpleNamespace(payload=self.payload)

    def set_oauth(self, _provider: str, payload: dict, *, account_name: str):
        self.payload = payload
        self.writes += 1
        return SimpleNamespace(provider=_provider, kind="oauth")


def test_resolve_refreshes_expired_token_and_persists_rotation(monkeypatch):
    accounts = MemoryAccounts(native_payload(int(time.time()) - 20))

    class Client:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def post(self, url, *, data, headers):
            assert url == oauth.OAUTH_ENDPOINT
            assert data["refresh_token"] == "refresh-1"
            assert headers["Content-Type"] == "application/x-www-form-urlencoded"
            return SimpleNamespace(
                status_code=200,
                json=lambda: {
                    "access_token": "opaque-access-2",
                    "refresh_token": "refresh-2",
                    "id_token": "id-2",
                    "expires_in": 3600,
                },
            )

    monkeypatch.setitem(
        __import__("sys").modules, "httpx2", SimpleNamespace(Client=Client)
    )
    resolved = oauth.resolve_codex_credentials(accounts)

    assert resolved.account_id == "acct-1"
    assert accounts.writes == 1
    assert accounts.payload["tokens"]["refresh_token"] == "refresh-2"
    assert accounts.payload["tokens"]["id_token"] == "id-2"
    assert accounts.payload["tokens"]["expires_at"] > time.time() + 3500
    assert resolved.expires_at == accounts.payload["tokens"]["expires_at"]
    assert accounts.payload["last_refresh"].endswith("Z")


def test_concurrent_refresh_rereads_after_lock(monkeypatch):
    accounts = MemoryAccounts(native_payload(int(time.time()) - 20))
    refresh_calls = 0
    guard = threading.Lock()

    class Client:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def post(self, *_args, **_kwargs):
            nonlocal refresh_calls
            with guard:
                refresh_calls += 1
            time.sleep(0.03)
            return SimpleNamespace(
                status_code=200,
                json=lambda: {
                    "access_token": jwt(int(time.time()) + 3600),
                    "refresh_token": "refresh-2",
                    "expires_in": 3600,
                },
            )

    monkeypatch.setitem(
        __import__("sys").modules, "httpx2", SimpleNamespace(Client=Client)
    )
    results: list[object] = []
    threads = [
        threading.Thread(
            target=lambda: results.append(oauth.resolve_codex_credentials(accounts))
        )
        for _ in range(2)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)

    assert len(results) == 2
    assert refresh_calls == 1
    assert accounts.writes == 1


def test_multiprocess_refresh_uses_interprocess_lock_and_rereads(tmp_path, monkeypatch):
    accounts = Accounts(root=tmp_path / "accounts")
    accounts.set_oauth(
        "openai-codex", native_payload(int(time.time()) - 20), account_name="work"
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), _RefreshHandler)
    server.refresh_requests = 0
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")

    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(2)
    queue = context.Queue()
    endpoint = f"http://127.0.0.1:{server.server_port}/oauth/token"
    workers = [
        context.Process(
            target=_multiprocess_refresh,
            args=(str(accounts.root), endpoint, barrier, queue),
        )
        for _ in range(2)
    ]
    try:
        for worker in workers:
            worker.start()
        results = [queue.get(timeout=15) for _ in workers]
        for worker in workers:
            worker.join(timeout=5)
        assert all(worker.exitcode == 0 for worker in workers)
        assert results == [("ok", "acct-1"), ("ok", "acct-1")]
        assert server.refresh_requests == 1
        saved = accounts.get_oauth("openai-codex", account_name="work").payload
        assert saved["tokens"]["refresh_token"] == "refresh-from-http-server"
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=2)
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=2)


def test_revoked_token_error_is_redacted(monkeypatch):
    accounts = MemoryAccounts(
        native_payload(int(time.time()) - 20, refresh="private-refresh")
    )

    class Client:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def post(self, *_args, **_kwargs):
            return SimpleNamespace(status_code=401)

    monkeypatch.setitem(
        __import__("sys").modules, "httpx2", SimpleNamespace(Client=Client)
    )
    with pytest.raises(RuntimeError, match="run account login") as error:
        oauth.resolve_codex_credentials(accounts)
    assert "private-refresh" not in str(error.value)
    assert accounts.writes == 0


def test_login_uses_isolated_child_home_and_keeps_old_grant(monkeypatch, tmp_path):
    external_home = tmp_path / "user-codex"
    external_home.mkdir()
    external_auth = external_home / "auth.json"
    external_auth.write_text("untouched")
    old = native_payload(int(time.time()) + 3600)
    accounts = MemoryAccounts(old, tmp_path / "managed")
    monkeypatch.setattr(oauth.shutil, "which", lambda _cmd: "/usr/bin/codex")
    captured: dict[str, str] = {}
    for name in (
        "OPENAI_API_KEY",
        "CODEX_API_KEY",
        "CODEX_ACCESS_TOKEN",
        "OPENAI_FEDERATION_RULE_ID",
        "OPENAI_IDENTITY_TOKEN_FILE",
    ):
        monkeypatch.setenv(name, "inherited-test-value")

    class Process:
        async def wait(self):
            home = Path(captured["CODEX_HOME"])
            (home / "auth.json").write_text(
                json.dumps(native_payload(int(time.time()) + 3600))
            )
            return 0

    async def fake_exec(*args, env, stdout, **_kwargs):
        assert args[0] == "/usr/bin/codex"
        assert "cli_auth_credentials_store=file" in args
        assert stdout is oauth.sys.stderr
        assert not set(oauth._AUTH_OVERRIDES) & env.keys()
        captured.update(env)
        return Process()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    result = asyncio.run(oauth.login(accounts, "openai-codex"))

    assert captured["CODEX_HOME"] != str(external_home)
    assert external_auth.read_text() == "untouched"
    assert accounts.writes == 1
    assert result.provider == "openai-codex"


def test_login_publication_waits_for_account_lock(monkeypatch, tmp_path):
    accounts = MemoryAccounts(
        native_payload(int(time.time()) + 3600), tmp_path / "managed"
    )
    raw_lock = threading.Lock()
    publication_waiting = threading.Event()

    @contextmanager
    def observed_lock(_provider: str, *, account_name: str):
        publication_waiting.set()
        with raw_lock:
            yield

    accounts.locked = observed_lock
    raw_lock.acquire()
    monkeypatch.setattr(oauth.shutil, "which", lambda _cmd: "/usr/bin/codex")

    class Process:
        async def wait(self):
            (Path(captured["CODEX_HOME"]) / "auth.json").write_text(
                json.dumps(native_payload(int(time.time()) + 7200, refresh="new"))
            )
            return 0

    captured = {}

    async def fake_exec(*_args, env, **_kwargs):
        captured.update(env)
        return Process()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    async def scenario():
        task = asyncio.create_task(oauth.login_codex(accounts))
        try:
            assert await asyncio.to_thread(publication_waiting.wait, 2)
            assert accounts.writes == 0
        finally:
            raw_lock.release()
        await task

    asyncio.run(scenario())
    assert accounts.writes == 1
    assert accounts.payload["tokens"]["refresh_token"] == "new"


def test_failed_login_does_not_replace_existing_credentials(monkeypatch, tmp_path):
    old = native_payload(int(time.time()) + 3600)
    accounts = MemoryAccounts(old, tmp_path / "managed")
    monkeypatch.setattr(oauth.shutil, "which", lambda _cmd: "/usr/bin/codex")

    class Process:
        async def wait(self):
            return 1

    async def fake_exec(*_args, **_kwargs):
        return Process()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    with pytest.raises(RuntimeError, match="existing account credentials were kept"):
        asyncio.run(oauth.login_codex(accounts))

    assert accounts.payload is old
    assert accounts.writes == 0
    assert list((tmp_path / "managed").iterdir()) == []


def test_cancelled_login_kills_unresponsive_cli_and_keeps_previous_grant(
    monkeypatch, tmp_path
):
    old = native_payload(int(time.time()) + 3600)
    accounts = MemoryAccounts(old, tmp_path / "managed")
    monkeypatch.setattr(oauth.shutil, "which", lambda _cmd: "/usr/bin/codex")
    waiting = threading.Event()
    state = {"killed": False, "wait_calls": 0}

    class Process:
        async def wait(self):
            state["wait_calls"] += 1
            if state["wait_calls"] == 1:
                waiting.set()
                await asyncio.Event().wait()
            return 0

        def terminate(self):
            return None

        def kill(self):
            state["killed"] = True

    async def fake_exec(*_args, **_kwargs):
        return Process()

    async def immediate_timeout(awaitable, *, timeout):
        assert timeout == 5
        awaitable.close()
        raise asyncio.TimeoutError

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(asyncio, "wait_for", immediate_timeout)

    async def scenario():
        task = asyncio.create_task(oauth.login_codex(accounts))
        await asyncio.to_thread(waiting.wait, 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    assert state["killed"] is True
    assert accounts.payload is old
    assert accounts.writes == 0


def test_async_refresh_cancellation_waits_for_commit(monkeypatch):
    accounts = MemoryAccounts(native_payload(int(time.time()) - 20))
    started = threading.Event()
    release = threading.Event()

    def fake_refresh(payload):
        started.set()
        release.wait(timeout=2)
        updated = json.loads(json.dumps(payload))
        updated["tokens"]["access_token"] = jwt(int(time.time()) + 3600)
        updated["tokens"]["refresh_token"] = "rotated"
        return updated

    monkeypatch.setattr(oauth, "_refresh", fake_refresh)

    async def scenario():
        task = asyncio.create_task(oauth.aresolve_codex_credentials(accounts))
        await asyncio.to_thread(started.wait, 1)
        task.cancel()
        asyncio.get_running_loop().call_later(0.03, release.set)
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    assert accounts.payload["tokens"]["refresh_token"] == "rotated"
    assert accounts.writes == 1
