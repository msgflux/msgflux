"""Dynamic provider auth: catalog-aware specs, namespaces, CLI config."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from msgflux.accounts import Accounts
from msgflux.accounts import cli as accounts_cli
from msgflux.accounts.providers import (
    api_key_providers,
    get_api_key_env,
    get_provider_auth,
    oauth_providers,
)
from msgflux.models.catalog import ProviderCatalog, ProviderConfig
from msgflux.models.provider_env import ProviderEnvBase
from msgflux.runtime.service.local.manifest import load_provider_catalog


def jwt(exp: int) -> str:
    claims = {
        "exp": exp,
        "https://api.openai.com/auth": {"chatgpt_account_id": "acct-1"},
    }
    payload = base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip("=")
    return f"header.{payload}.signature"


def native_payload(exp: int) -> dict:
    return {
        "auth_mode": "chatgpt",
        "tokens": {
            "access_token": jwt(exp),
            "refresh_token": "refresh-1",
            "account_id": "acct-1",
        },
    }


def test_explicit_api_key_env_wins_even_for_codex_base():
    catalog = ProviderCatalog(
        {"my-codex": ProviderConfig(base="openai-codex", api_key_env="MY_CODEX_KEY")}
    )
    spec = get_provider_auth("my-codex", provider_catalog=catalog)
    assert spec.method == "api_key"
    assert spec.env_name == "MY_CODEX_KEY"
    assert get_api_key_env("my-codex", provider_catalog=catalog) == "MY_CODEX_KEY"


def test_api_key_alias_derives_own_env_not_base_env():
    catalog = ProviderCatalog({"nim-live": ProviderConfig(base="nvidia")})
    spec = get_provider_auth("nim-live", provider_catalog=catalog)
    assert spec.method == "api_key"
    # Base nvidia uses NVIDIA_API_KEY; the alias must not silently reuse it.
    assert spec.env_name == "NIM_LIVE_API_KEY"
    assert spec.env_name != get_api_key_env("nvidia")


def test_oauth_alias_inherits_codex_mechanism_with_own_namespace(tmp_path):
    catalog = ProviderCatalog({"my-codex": ProviderConfig(base="openai-codex")})
    spec = get_provider_auth("my-codex", provider_catalog=catalog)
    assert spec.method == "oauth"
    assert spec.oauth_adapter == "codex"

    accounts = Accounts(tmp_path / "accounts", provider_catalog=catalog)
    accounts.set_oauth("my-codex", native_payload(int(time.time()) + 3600))
    # Stored under the alias, never under the base provider.
    assert (tmp_path / "accounts" / "default" / "my-codex" / "auth.json").exists()
    assert not (tmp_path / "accounts" / "default" / "openai-codex").exists()
    with pytest.raises(Exception, match="does not use an API key"):
        get_api_key_env("my-codex", provider_catalog=catalog)

    resolver = accounts.resolver("my-codex")
    owner = SimpleNamespace(provider="my-codex")
    headers = resolver.resolve(owner).headers
    assert headers["chatgpt-account-id"] == "acct-1"
    assert headers["Authorization"].startswith("Bearer header.")


def test_oauth_refresh_stays_in_alias_namespace(tmp_path, monkeypatch):
    from msgflux.accounts import oauth

    catalog = ProviderCatalog({"my-codex": ProviderConfig(base="openai-codex")})
    accounts = Accounts(tmp_path / "accounts", provider_catalog=catalog)
    accounts.set_oauth("my-codex", native_payload(int(time.time()) - 20))

    def fake_refresh(payload):
        updated = json.loads(json.dumps(payload))
        updated["tokens"]["access_token"] = jwt(int(time.time()) + 3600)
        updated["tokens"]["refresh_token"] = "rotated-alias"
        return updated

    monkeypatch.setattr(oauth, "_refresh", fake_refresh)
    resolved = oauth.resolve_codex_credentials(accounts, provider="my-codex")
    assert resolved.account_id == "acct-1"
    stored = accounts.get_oauth("my-codex").payload
    assert stored["tokens"]["refresh_token"] == "rotated-alias"
    assert not (tmp_path / "accounts" / "default" / "openai-codex").exists()
    assert (
        asyncio.run(
            oauth.aresolve_codex_credentials(accounts, provider="my-codex")
        ).account_id
        == "acct-1"
    )


def test_oauth_login_publishes_under_alias_namespace(tmp_path, monkeypatch):
    from msgflux.accounts import oauth

    catalog = ProviderCatalog({"my-codex": ProviderConfig(base="openai-codex")})
    accounts = Accounts(tmp_path / "accounts", provider_catalog=catalog)
    monkeypatch.setattr(oauth.shutil, "which", lambda _cmd: "/usr/bin/codex")

    class Process:
        async def wait(self):
            return 0

    async def fake_exec(*_args, env, **_kwargs):
        (Path(env["CODEX_HOME"]) / "auth.json").write_text(
            json.dumps(native_payload(int(time.time()) + 3600))
        )
        return Process()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    result = asyncio.run(accounts.login("my-codex"))
    assert result.provider == "my-codex"
    assert (tmp_path / "accounts" / "default" / "my-codex" / "auth.json").exists()
    assert not (tmp_path / "accounts" / "default" / "openai-codex").exists()


def test_registered_class_auth_follows_provider_env_metadata():
    from msgflux.models.base import BaseModel

    class FactoryModel(ProviderEnvBase, BaseModel):
        model_type = "chat_completion"
        provider = "ext-widget"
        api_key_env = "EXT_WIDGET_KEY"

    catalog = ProviderCatalog()
    catalog.register(FactoryModel)
    # Programmatic registration resolves through the catalog's class identity.
    spec = get_provider_auth("ext-widget", provider_catalog=catalog)
    assert (spec.method, spec.env_name) == ("api_key", "EXT_WIDGET_KEY")


def test_codex_factory_descendant_inherits_oauth_unless_env_override():
    from msgflux.models.providers.openai_codex import OpenAICodexChatCompletion

    class CodexAlias(OpenAICodexChatCompletion):
        provider = "codex-alias"

    class CodexKeyAlias(OpenAICodexChatCompletion):
        provider = "codex-key-alias"
        api_key_env = "CODEX_KEY_ALIAS_KEY"

    catalog = ProviderCatalog()
    catalog.register(CodexAlias)
    catalog.register(CodexKeyAlias)
    assert get_provider_auth("codex-alias", provider_catalog=catalog).method == "oauth"
    spec = get_provider_auth("codex-key-alias", provider_catalog=catalog)
    assert (spec.method, spec.env_name) == ("api_key", "CODEX_KEY_ALIAS_KEY")


def test_static_listing_does_not_import_external_factory():
    catalog = ProviderCatalog(
        {
            "ext-static": ProviderConfig(factory="uninstalled_plugin:Provider"),
            "nim-live": ProviderConfig(base="nvidia"),
        }
    )
    listed = api_key_providers(provider_catalog=catalog)
    assert listed["nim-live"].env_name == "NIM_LIVE_API_KEY"
    assert "ext-static" not in listed
    assert "ext-static" not in oauth_providers(provider_catalog=catalog)


def test_anthropic_alias_resolver_sends_native_x_api_key_header(tmp_path):
    catalog = ProviderCatalog(
        {
            "acme": ProviderConfig(
                base="anthropic",
                base_url="https://acme.invalid",
                api_key_env="ACME_KEY",
            )
        }
    )
    accounts = Accounts(tmp_path / "accounts", provider_catalog=catalog)
    accounts.set_api_key("acme", "managed-alias-key")
    resolver = accounts.resolver("acme")

    from msgflux.models.providers.anthropic import AnthropicChatCompletion

    owner = AnthropicChatCompletion.__new__(AnthropicChatCompletion)
    owner.provider = "acme"
    assert resolver.resolve(owner).headers == {"x-api-key": "managed-alias-key"}


def test_builtin_auth_unchanged_without_catalog():
    assert get_provider_auth("openai-codex").method == "oauth"
    assert get_api_key_env("openai") == "OPENAI_API_KEY"


def test_cli_config_loads_only_provider_declarations(tmp_path, monkeypatch, capsys):
    config = tmp_path / "config.toml"
    # No defaults.model / agents: login must still work from declarations.
    config.write_text("[providers.acme]\nbase = 'openai'\napi_key_env = 'ACME_KEY'\n")
    catalog = load_provider_catalog(config)
    assert catalog.get_config("acme").api_key_env == "ACME_KEY"

    accounts_dir = str(tmp_path / "accounts")
    monkeypatch.setattr(accounts_cli.sys, "stdin", io.StringIO("cli-alias-key\n"))
    accounts_cli.main(
        [
            "login",
            "acme",
            "--stdin",
            "--config",
            str(config),
            "--accounts-dir",
            accounts_dir,
            "--json",
        ]
    )
    result = json.loads(capsys.readouterr().out)
    assert result["provider"] == "acme"
    assert result["method"] == "api_key"
    assert "cli-alias-key" not in json.dumps(result)
    stored = Accounts(root=accounts_dir, provider_catalog=catalog).get("acme")
    assert stored.env_name == "ACME_KEY"
    assert stored.value == "cli-alias-key"


def test_cli_default_config_used_when_present(tmp_path, monkeypatch, capsys):
    home = tmp_path / "home"
    (home / ".msgflux").mkdir(parents=True)
    (home / ".msgflux" / "config.toml").write_text(
        "[providers.acme]\nbase = 'openai'\napi_key_env = 'ACME_KEY'\n"
    )
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(accounts_cli.sys, "stdin", io.StringIO("default-config-key\n"))
    accounts_cli.main(
        [
            "login",
            "acme",
            "--stdin",
            "--accounts-dir",
            str(tmp_path / "accounts"),
            "--json",
        ]
    )
    result = json.loads(capsys.readouterr().out)
    assert result["provider"] == "acme"
    assert result["method"] == "api_key"


def test_cli_no_config_login_still_works(tmp_path, monkeypatch, capsys):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(accounts_cli.sys, "stdin", io.StringIO("builtin-key\n"))
    accounts_cli.main(
        [
            "login",
            "openai",
            "--stdin",
            "--accounts-dir",
            str(tmp_path / "accounts"),
            "--json",
        ]
    )
    assert json.loads(capsys.readouterr().out)["provider"] == "openai"
