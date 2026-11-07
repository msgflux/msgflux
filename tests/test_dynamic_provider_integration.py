"""Configured providers compose with the coding host without global mutation."""

from pathlib import Path

import pytest

from msgflux.coding.factory import _create_model
from msgflux.models.catalog import ProviderCatalog, ProviderConfig
from msgflux.models.model import Model
from msgflux.runtime.service.local.manifest import (
    load_provider_catalog,
    load_service_manifest,
)


def _manifest(tmp_path: Path, declarations: str, *, provider="acme"):
    path = tmp_path / "config.toml"
    path.write_text(
        "[defaults]\nagent = 'coding'\n"
        f"model = '{provider}/custom-model'\n"
        "[agents.coding]\npreset = 'builder'\n" + declarations
    )
    return path


def test_manifest_derives_workspace_defaults_without_importing_factory(tmp_path):
    path = _manifest(
        tmp_path,
        "[providers.acme]\nbase = 'openai'\napi_key_env = 'ACME_KEY'\n"
        "[providers.unused]\nfactory = 'uninstalled_plugin:Provider'\n",
    )
    manifest = load_service_manifest(path)
    options = manifest.coding_catalog.agents["coding"]
    assert options.tools_active == ("bash", "read", "apply_patch")
    assert manifest.provider_catalog.get_config("unused").factory == (
        "uninstalled_plugin:Provider"
    )


def test_auth_declarations_load_before_defaults_model_is_selected(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(
        "[defaults]\nagent = 'coding'\n[agents.coding]\npreset = 'coding'\n"
        "[providers.acme]\nbase = 'anthropic'\napi_key_env = 'ACME_KEY'\n"
    )
    catalog = load_provider_catalog(path)
    assert catalog.get_config("acme").base == "anthropic"
    with pytest.raises(ValueError, match="requires a model"):
        load_service_manifest(path)


def test_coding_model_uses_managed_alias_credentials(tmp_path, monkeypatch):
    from msgflux.accounts import Accounts

    monkeypatch.setenv("OPENAI_API_KEY", "wrong-base-key")
    path = _manifest(
        tmp_path,
        "[providers.acme]\nbase = 'openai'\n"
        "base_url = 'https://acme.invalid/v1'\napi_key_env = 'ACME_KEY'\n",
    )
    manifest = load_service_manifest(path)
    catalog = manifest.provider_catalog
    accounts = Accounts(tmp_path / "accounts", provider_catalog=catalog)
    accounts.set_api_key("acme", "alias-account-key")
    model = _create_model(
        manifest.coding_catalog.agents["coding"],
        accounts=accounts,
        provider_catalog=catalog,
    )
    try:
        assert model.provider == "acme"
        assert model.credential_resolver.resolve(model).headers == {
            "Authorization": "Bearer alias-account-key"
        }
        assert model._get_base_url() == "https://acme.invalid/v1"
        serialized = str(model.serialize())
        assert "alias-account-key" not in serialized
        assert "wrong-base-key" not in serialized
    finally:
        model.close()


@pytest.mark.asyncio
async def test_custom_async_factory_keeps_catalog_context_across_await(tmp_path):
    import asyncio

    from msgflux.runtime.service.local.runner import _with_factory_import_root

    catalog = ProviderCatalog({"acme": ProviderConfig(base="nvidia")})

    async def factory(_thread):
        await asyncio.sleep(0)
        return Model._get_model_class("chat_completion", "acme")

    wrapped = _with_factory_import_root(factory, tmp_path, catalog)
    cls = await wrapped(None)
    assert cls.provider == "acme"
    assert "acme" not in Model.providers()["chat_completion"]


@pytest.mark.parametrize("managed", [False, True])
def test_anthropic_alias_preserves_native_wire_and_signed_history(
    tmp_path, monkeypatch, managed
):
    import json

    import httpx2

    from msgflux.accounts import Accounts
    from msgflux.chat_messages import ChatMessages
    from msgflux.models.http_transport import HTTPTransport

    monkeypatch.setenv("ACME_KEY", "alias-key")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://wrong-base.invalid")
    catalog = ProviderCatalog(
        {
            "acme": ProviderConfig(
                base="anthropic",
                base_url="https://acme.invalid",
                api_key_env="ACME_KEY",
            )
        }
    )
    requests = []

    def handler(request):
        requests.append(request)
        return httpx2.Response(
            200,
            json={
                "id": "msg_alias",
                "type": "message",
                "role": "assistant",
                "model": "custom-model",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "Checked.",
                        "signature": "sig-alias",
                    },
                    {"type": "text", "text": "OK"},
                ],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 4, "output_tokens": 3},
            },
        )

    kwargs = {}
    if managed:
        accounts = Accounts(tmp_path / "accounts", provider_catalog=catalog)
        accounts.set_api_key("acme", "managed-alias-key")
        kwargs["credential_resolver"] = accounts.resolver("acme")
    with httpx2.Client(transport=httpx2.MockTransport(handler)) as client:
        model = Model.chat_completion(
            "acme/custom-model", provider_catalog=catalog, retry=False, **kwargs
        )
        model.http_transport = HTTPTransport(client=client, max_retries=0)
        try:
            response = model("Check")
            assert response.consume() == "OK"
            state = response.history_items[0]["provider_state"]
            assert state["provider"] == "acme"
            assert state["data"]["signature"] == "sig-alias"
            messages = ChatMessages()
            messages.add_user("Check")
            for item in response.history_items:
                messages.append(item)
            messages.add_assistant("OK")
            messages.add_user("Again")
            assert model(messages).consume() == "OK"
        finally:
            model.close()
    assert str(requests[0].url) == "https://acme.invalid/v1/messages"
    expected_key = "managed-alias-key" if managed else "alias-key"
    assert requests[0].headers["x-api-key"] == expected_key
    assert requests[0].headers["anthropic-version"] == "2023-06-01"
    replayed = json.loads(requests[1].content)["messages"]
    assert replayed[1]["content"][0] == {
        "type": "thinking",
        "thinking": "Checked.",
        "signature": "sig-alias",
    }


def test_dynamic_factory_roundtrip_in_fresh_process_requires_trusted_config(
    tmp_path, monkeypatch
):
    import os
    import subprocess
    import sys

    plugin = tmp_path / "test_dynamic_plugin.py"
    plugin.write_text(
        "from msgflux import register_model\n"
        "from msgflux.models.providers.nvidia import NVIDIAChatCompletion\n"
        "@register_model\n"
        "class AcmeChat(NVIDIAChatCompletion):\n"
        "    provider = 'acme'\n"
        "    api_key_env = 'ACME_KEY'\n"
        "    base_url = 'https://acme.invalid/v1'\n"
        "    base_url_env = None\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("ACME_KEY", "test-only-key")
    catalog = ProviderCatalog(
        {"acme": ProviderConfig(factory="test_dynamic_plugin:AcmeChat")}
    )
    model = Model.chat_completion("acme/my-model", provider_catalog=catalog)
    try:
        serialized = model.serialize()
    finally:
        model.close()
    state_path = tmp_path / "model.json"
    import msgspec

    state_path.write_bytes(msgspec.json.encode(serialized))
    script = """
import json, sys
from pathlib import Path
from msgflux import Model, ProviderCatalog
from msgflux.models.catalog import ProviderConfig
state = json.loads(Path(sys.argv[1]).read_text())
try:
    Model.from_serialized(state['provider'], state['model_type'], state['state'])
except ValueError:
    pass
else:
    raise AssertionError('Restored untrusted factory')
assert 'test_dynamic_plugin' not in sys.modules
catalog = ProviderCatalog({'acme': ProviderConfig(factory='test_dynamic_plugin:AcmeChat')})
model = Model.from_serialized(state['provider'], state['model_type'], state['state'], provider_catalog=catalog)
assert model.provider == 'acme'
assert model.model_id == 'my-model'
assert model._get_base_url() == 'https://acme.invalid/v1'
assert 'acme' not in Model.providers()['chat_completion']
model.close()
print('fresh-process-ok')
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        (str(tmp_path), str(Path(__file__).resolve().parents[1] / "src"))
    )
    result = subprocess.run(  # noqa: S603 - trusted, fixed test script
        [sys.executable, "-c", script, str(state_path)],
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "fresh-process-ok"


def test_oauth_alias_uses_own_managed_namespace_after_restore(tmp_path, monkeypatch):
    import time

    from msgflux.accounts import Accounts
    from tests.accounts.test_dynamic_providers import native_payload

    monkeypatch.setenv("HOME", str(tmp_path))
    catalog = ProviderCatalog({"acme": ProviderConfig(base="openai-codex")})
    accounts = Accounts(provider_catalog=catalog)
    accounts.set_oauth("acme", native_payload(int(time.time()) + 3600))
    model = Model.chat_completion("acme/my-model", provider_catalog=catalog)
    try:
        assert model.credential_resolver.provider == "acme"
        credentials = model.credential_resolver.resolve(model)
        assert credentials.headers["chatgpt-account-id"] == "acct-1"
        serialized = model.serialize()
        restored = Model.from_serialized(
            "acme", "chat_completion", serialized["state"], provider_catalog=catalog
        )
        try:
            assert restored.credential_resolver.provider == "acme"
            assert (
                restored.credential_resolver.resolve(restored).headers
                == credentials.headers
            )
        finally:
            restored.close()
    finally:
        model.close()


def test_factory_cannot_relabel_provider_via_endpoint_override(monkeypatch):
    from msgflux.models.providers.nvidia import NVIDIAChatCompletion

    monkeypatch.setattr(
        "msgflux.models.catalog.load_factory", lambda _path: NVIDIAChatCompletion
    )
    catalog = ProviderCatalog(
        {
            "acme": ProviderConfig(
                factory="some_module:NVIDIAChatCompletion",
                base_url="https://acme.invalid",
            )
        }
    )
    with pytest.raises(ValueError, match="does not match"):
        catalog.get_model_class("chat_completion", "acme")


def test_provider_enumeration_keeps_factories_lazy():
    catalog = ProviderCatalog(
        {"acme": ProviderConfig(factory="uninstalled_plugin:AcmeChatCompletion")}
    )
    assert "acme" in Model.providers(provider_catalog=catalog)["chat_completion"]
    assert "acme" not in Model.providers()["chat_completion"]


def test_ollama_alias_uses_own_native_endpoint_and_key(monkeypatch):
    monkeypatch.setenv("ACME_KEY", "alias-key")
    monkeypatch.setenv("OLLAMA_API_KEY", "wrong-key")
    monkeypatch.setenv("OLLAMA_BASE_URL", "https://wrong-base.invalid")
    catalog = ProviderCatalog(
        {
            "acme": ProviderConfig(
                base="ollama",
                base_url="https://acme.invalid/v1",
                api_key_env="ACME_KEY",
            )
        }
    )
    model = Model.chat_completion("acme/my-model", provider_catalog=catalog)
    try:
        assert model._get_base_url() == "https://acme.invalid"
        assert model.credential_resolver.resolve(model).headers == {
            "Authorization": "Bearer alias-key"
        }
    finally:
        model.close()


def test_derived_alias_looks_up_base_model_profile(monkeypatch):
    from msgflux.models import openai_compatible

    calls = []
    sentinel = object()

    def profile(model_id, provider_id=None):
        calls.append((model_id, provider_id))
        return sentinel

    monkeypatch.setattr(openai_compatible, "get_model_profile", profile)
    monkeypatch.setenv("ACME_KEY", "test-key")
    catalog = ProviderCatalog(
        {"acme": ProviderConfig(base="anthropic", api_key_env="ACME_KEY")}
    )
    model = Model.chat_completion("acme/custom-model", provider_catalog=catalog)
    try:
        assert model.profile is sentinel
        assert calls[-1] == ("custom-model", "anthropic")
        assert model.provider == "acme"
    finally:
        model.close()


def test_gateway_restores_configured_models_in_trusted_scope(monkeypatch):
    from msgflux import ModelGateway

    monkeypatch.setenv("ACME_KEY", "test-key")
    catalog = ProviderCatalog(
        {"acme": ProviderConfig(base="nvidia", api_key_env="ACME_KEY")}
    )
    with catalog.use():
        gateway = ModelGateway(
            [{"model_name": "fast", "model": "acme/custom-model"}], fallback=False
        )
    state = gateway.serialize()
    for model in gateway.models:
        model.close()
    with pytest.raises(ValueError, match="trusted"):
        ModelGateway.from_serialized(state)
    with catalog.use():
        restored = ModelGateway.from_serialized(state)
    try:
        assert restored.serialize()["state"]["models"][0]["model"]["provider"] == "acme"
        assert restored.fallback is False
    finally:
        for model in restored.models:
            model.close()


def test_options_reject_lossy_serialization():
    from datetime import datetime, timezone

    with pytest.raises(ValueError, match="plain"):
        ProviderConfig(base="nvidia", options={"timestamp": datetime.now(timezone.utc)})


def test_auth_listing_never_imports_factory_through_derived_alias(monkeypatch):
    from msgflux.accounts.providers import api_key_providers, oauth_providers

    def unexpected_import(_path):
        raise AssertionError("Static auth listing imported a factory")

    monkeypatch.setattr("msgflux.models.catalog.load_factory", unexpected_import)
    catalog = ProviderCatalog(
        {
            "external": ProviderConfig(factory="uninstalled_plugin:AcmeChatCompletion"),
            "acme": ProviderConfig(base="external"),
        }
    )
    assert "acme" not in api_key_providers(provider_catalog=catalog)
    assert "acme" not in oauth_providers(provider_catalog=catalog)


@pytest.mark.asyncio
async def test_unsupported_oauth_adapter_does_not_launch_codex(monkeypatch):
    from msgflux.accounts import Accounts, oauth, providers
    from msgflux.accounts.providers import ProviderAuthSpec

    monkeypatch.setattr(
        providers,
        "get_provider_auth",
        lambda *_args, **_kwargs: ProviderAuthSpec("future-oauth", "oauth"),
    )

    async def unexpected_login(*_args, **_kwargs):
        raise AssertionError("An unsupported OAuth provider launched Codex")

    monkeypatch.setattr(oauth, "login_codex", unexpected_login)
    with pytest.raises(ValueError, match="not implemented"):
        await oauth.login(Accounts(), "future-oauth")


def test_scoped_backend_classes_are_collectable(monkeypatch):
    import gc
    import weakref

    monkeypatch.setenv("ACME_KEY", "test-key")
    catalog = ProviderCatalog(
        {"acme": ProviderConfig(base="opencode-zen", api_key_env="ACME_KEY")}
    )
    facade = catalog.get_model_class("chat_completion", "acme")
    facade_ref = weakref.ref(facade)
    model = Model.chat_completion(
        "acme/custom-model", provider_catalog=catalog, api_mode="anthropic_messages"
    )
    backend_ref = weakref.ref(type(model))
    model.close()
    del facade, model, catalog
    gc.collect()
    assert facade_ref() is None
    assert backend_ref() is None


def test_replacing_scoped_base_invalidates_future_alias_derivations():
    from msgflux.models.providers.nvidia import NVIDIAChatCompletion

    class First(NVIDIAChatCompletion):
        provider = "custom-base"

    class Second(NVIDIAChatCompletion):
        provider = "custom-base"

    catalog = ProviderCatalog({"acme": ProviderConfig(base="custom-base")})
    catalog.register(First)
    assert issubclass(catalog.get_model_class("chat_completion", "acme"), First)
    catalog.register(Second, replace=True)
    assert issubclass(catalog.get_model_class("chat_completion", "acme"), Second)


@pytest.mark.parametrize("name", [".", "..", "../escape", "a/b", "a:b"])
def test_catalog_rejects_unsafe_account_names(name):
    with pytest.raises(ValueError, match="provider name"):
        ProviderCatalog({name: ProviderConfig(base="nvidia")})


def test_accounts_validate_programmatic_provider_namespace(tmp_path, monkeypatch):
    from msgflux.accounts import Accounts
    from msgflux.accounts.providers import ProviderAuthSpec

    accounts = Accounts(tmp_path / "accounts")
    monkeypatch.setattr(
        accounts, "_auth", lambda _provider: ProviderAuthSpec("..", "api_key", "KEY")
    )
    with pytest.raises(ValueError, match="provider name"):
        accounts.credential_path("..")
    assert not accounts.root.exists()
