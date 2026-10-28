from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from msgflux.accounts import Accounts
from msgflux.accounts import cli as accounts_cli
from msgflux.accounts.providers import ProviderAuthSpec


def _run_cli(args: list[str], *, input_text: str | None = None, env=None):
    # Invoke only the current Python interpreter with fixed module/command
    # arguments; args in these tests are local literals and temporary paths.
    return subprocess.run(  # noqa: S603
        [sys.executable, "-m", "msgflux.cli", "accounts", *args],
        input=input_text,
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )


def _fake_codex_script(path: Path) -> Path:
    executable = path / "codex"
    executable.parent.mkdir(parents=True, exist_ok=True)
    executable.write_text(
        f"#!{sys.executable}\n"
        "import base64, json, os, time\n"
        "claims = {'exp': int(time.time()) + 3600, "
        "'https://api.openai.com/auth': {'chatgpt_account_id': 'acct-cli'}}\n"
        "part = base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip('=')\n"
        "payload = {'auth_mode': 'chatgpt', 'tokens': {"
        "'access_token': 'header.' + part + '.signature', "
        "'refresh_token': 'fake-cli-refresh', 'account_id': 'acct-cli'}}\n"
        "with open(os.path.join(os.environ['CODEX_HOME'], 'auth.json'), 'w') as f:\n"
        "    json.dump(payload, f)\n"
        "print('fake Codex progress')\n",
        encoding="utf-8",
    )
    executable.chmod(0o700)
    return executable


def test_api_key_uses_hidden_prompt_and_never_emits_secret(
    monkeypatch, tmp_path, capsys
):
    secret = "sk-fake-hidden-prompt-value"
    prompts: list[str] = []
    monkeypatch.setattr(
        accounts_cli,
        "getpass",
        SimpleNamespace(getpass=lambda prompt: prompts.append(prompt) or secret),
    )

    accounts_cli.main(["login", "openai", "--accounts-dir", str(tmp_path), "--json"])

    output = capsys.readouterr().out
    assert json.loads(output) == {
        "status": "configured",
        "account_name": "default",
        "provider": "openai",
        "method": "api_key",
    }
    assert prompts and "OPENAI_API_KEY" in prompts[0]
    assert secret not in output
    assert Accounts(root=tmp_path).get("openai").value == secret


def test_stdin_import_keeps_two_accounts_and_providers_separate(
    tmp_path, capsys, monkeypatch
):
    accounts_dir = str(tmp_path / "accounts")
    account_store = Accounts(root=accounts_dir)
    entries = [
        ("openai", "default", "sk-openai-default"),
        ("groq", "default", "gsk-groq-default"),
        ("openai", "work", "sk-openai-work"),
    ]
    outputs = []
    for provider, account_name, secret in entries:
        monkeypatch.setattr(accounts_cli.sys, "stdin", io.StringIO(secret + "\n"))
        accounts_cli.main(
            [
                "login",
                provider,
                "--account-name",
                account_name,
                "--stdin",
                "--accounts-dir",
                accounts_dir,
                "--json",
            ]
        )
        outputs.append(capsys.readouterr().out)

    assert account_store.list() == ["default", "work"]
    assert account_store.get("openai", "default").value == "sk-openai-default"
    assert account_store.get("groq", "default").value == "gsk-groq-default"
    assert account_store.get("openai", "work").value == "sk-openai-work"
    assert account_store.info("default").keys() == {"openai", "groq"}
    assert account_store.info("work").keys() == {"openai"}
    assert all(secret not in "".join(outputs) for _, _, secret in entries)


def test_from_env_stores_unmodified_value_without_echoing_it(
    monkeypatch, tmp_path, capsys
):
    secret = "  sk-fake raw env value  "
    monkeypatch.setenv("OPENAI_API_KEY", secret)

    accounts_cli.main(
        ["login", "openai", "--from-env", "--accounts-dir", str(tmp_path), "--json"]
    )

    output = capsys.readouterr().out
    assert secret not in output
    assert Accounts(root=tmp_path).get("openai").value == secret


@pytest.mark.parametrize(
    ("provider", "env_name", "other_env"),
    [
        ("opencode-go", "OPENCODE_GO_API_KEY", "OPENCODE_ZEN_API_KEY"),
        ("opencode-zen", "OPENCODE_ZEN_API_KEY", "OPENCODE_GO_API_KEY"),
    ],
)
def test_opencode_account_import_uses_its_own_key(
    monkeypatch, tmp_path, capsys, provider, env_name, other_env
):
    monkeypatch.setenv(env_name, "own-plan-key")
    monkeypatch.setenv(other_env, "other-plan-key")
    monkeypatch.setenv("OPENCODE_API_KEY", "generic-decoy-key")
    args = ["login", provider, "--from-env", "--accounts-dir", str(tmp_path), "--json"]

    accounts_cli.main(args)
    credential = Accounts(root=tmp_path).get(provider)
    assert credential.env_name == env_name
    assert credential.value == "own-plan-key"
    output = capsys.readouterr().out
    assert json.loads(output)["provider"] == provider
    assert "own-plan-key" not in output
    assert "other-plan-key" not in output
    assert "generic-decoy-key" not in output

    monkeypatch.delenv(env_name)
    with pytest.raises(SystemExit, match="1"):
        accounts_cli.main(args)
    assert json.loads(capsys.readouterr().out)["status"] == "failed"
    assert Accounts(root=tmp_path).get(provider).value == "own-plan-key"


def test_from_env_for_imagerouter_uses_provider_key_when_both_are_set(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-openai-key")
    monkeypatch.setenv("IMAGEROUTER_API_KEY", "ir-fake-imagerouter-key")

    accounts_cli.main(
        [
            "login",
            "imagerouter",
            "--from-env",
            "--accounts-dir",
            str(tmp_path),
            "--json",
        ]
    )

    output = capsys.readouterr().out
    credential = Accounts(root=tmp_path).get("imagerouter")
    assert credential.value == "ir-fake-imagerouter-key"
    assert credential.env_name == "IMAGEROUTER_API_KEY"
    assert "ir-fake-imagerouter-key" not in output
    assert "sk-fake-openai-key" not in output


def test_from_env_for_imagerouter_does_not_fall_back_to_openai_key(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-openai-key")
    monkeypatch.delenv("IMAGEROUTER_API_KEY", raising=False)

    with pytest.raises(SystemExit, match="1"):
        accounts_cli.main(
            [
                "login",
                "imagerouter",
                "--from-env",
                "--accounts-dir",
                str(tmp_path),
                "--json",
            ]
        )

    output = capsys.readouterr().out
    result = json.loads(output)
    assert result["status"] == "failed"
    assert "IMAGEROUTER_API_KEY" in result["error"]
    assert "sk-fake-openai-key" not in output
    assert not Accounts(root=tmp_path).credential_path("imagerouter").exists()


@pytest.mark.parametrize("source_flag", ["--stdin", "--from-env"])
def test_oauth_provider_rejects_api_key_source_flags(
    monkeypatch, tmp_path, capsys, source_flag
):
    called = False

    async def login(self, provider, account_name="default"):
        nonlocal called
        called = True

    monkeypatch.setattr(Accounts, "login", login)
    with pytest.raises(SystemExit, match="1"):
        accounts_cli.main(
            [
                "login",
                "openai-codex",
                source_flag,
                "--accounts-dir",
                str(tmp_path),
                "--json",
            ]
        )
    output = json.loads(capsys.readouterr().out)
    assert "OAuth" in output["error"]
    assert "API-key input" in output["error"]
    assert called is False


def test_oauth_provider_routes_through_accounts_login(monkeypatch, tmp_path, capsys):
    calls: list[tuple[str, str]] = []

    async def login(self, provider, account_name="default"):
        calls.append((provider, account_name))
        return SimpleNamespace(provider=provider, kind="oauth")

    monkeypatch.setattr(Accounts, "login", login)
    accounts_cli.main(
        [
            "login",
            "openai-codex",
            "--account-name",
            "work",
            "--accounts-dir",
            str(tmp_path),
            "--json",
        ]
    )

    assert calls == [("openai-codex", "work")]
    assert json.loads(capsys.readouterr().out)["method"] == "oauth"


def test_known_oauth_without_adapter_returns_actionable_redacted_error(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.setattr(
        accounts_cli,
        "get_provider_auth",
        lambda _provider: ProviderAuthSpec("future-oauth", "oauth"),
    )

    async def login(self, provider, account_name="default"):
        raise ValueError(
            f"OAuth login is not implemented for provider {provider!r}; "
            "use a provider with a supported account login flow"
        )

    monkeypatch.setattr(Accounts, "login", login)
    with pytest.raises(SystemExit, match="1"):
        accounts_cli.main(
            ["login", "future-oauth", "--accounts-dir", str(tmp_path), "--json"]
        )
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "failed"
    assert "not implemented" in result["error"]
    assert "supported account login flow" in result["error"]


def test_unknown_provider_error_is_actionable(tmp_path, capsys):
    with pytest.raises(SystemExit, match="1"):
        accounts_cli.main(
            [
                "login",
                "anthropic-claude-code",
                "--accounts-dir",
                str(tmp_path),
                "--json",
            ]
        )
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "failed"
    assert "Unsupported account provider" in result["error"]


def test_accounts_login_subprocess_reads_only_stdin_and_emits_clean_json(tmp_path):
    environment = {
        "PATH": os.pathsep.join([str(Path(sys.executable).parent), os.defpath]),
        "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src"),
        "HOME": str(tmp_path / "home"),
        "PYTHONUNBUFFERED": "1",
    }
    (tmp_path / "home").mkdir()
    accounts_dir = tmp_path / "accounts"
    fake_secret = "sk-fake-subprocess-key"
    result = _run_cli(
        [
            "login",
            "openai",
            "--stdin",
            "--accounts-dir",
            str(accounts_dir),
            "--json",
        ],
        input_text=fake_secret + "\n",
        env=environment,
    )

    assert result.returncode == 0
    assert json.loads(result.stdout) == {
        "status": "configured",
        "account_name": "default",
        "provider": "openai",
        "method": "api_key",
    }
    assert fake_secret not in result.stdout
    assert Accounts(root=accounts_dir).get("openai").value == fake_secret


def test_accounts_login_subprocess_isolates_fake_codex_and_keeps_json_stdout(
    tmp_path,
):
    fake_bin = tmp_path / "bin"
    _fake_codex_script(fake_bin)
    environment = {
        "PATH": os.pathsep.join(
            [str(fake_bin), str(Path(sys.executable).parent), os.defpath]
        ),
        "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src"),
        "HOME": str(tmp_path / "home"),
        "PYTHONUNBUFFERED": "1",
    }
    (tmp_path / "home").mkdir()
    accounts_dir = tmp_path / "accounts"
    result = _run_cli(
        [
            "login",
            "openai-codex",
            "--account-name",
            "work",
            "--accounts-dir",
            str(accounts_dir),
            "--json",
        ],
        env=environment,
    )

    assert result.returncode == 0
    assert json.loads(result.stdout) == {
        "status": "configured",
        "account_name": "work",
        "provider": "openai-codex",
        "method": "oauth",
    }
    assert "fake Codex progress" in result.stderr
    assert result.stdout.count("\n") == 1
    assert not (tmp_path / "home" / ".codex" / "auth.json").exists()
    stored = json.loads(
        (accounts_dir / "work" / "openai-codex" / "auth.json").read_text()
    )
    assert stored["tokens"]["refresh_token"] == "fake-cli-refresh"
    assert "fake-cli-refresh" not in result.stdout + result.stderr
