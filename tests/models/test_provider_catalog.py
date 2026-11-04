"""Tests for msgflux.models.catalog dynamic providers."""

import copy
import importlib
import sys
import textwrap

import msgspec
import pytest

from msgflux.models.base import BaseModel
from msgflux.models.catalog import (
    ProviderCatalog,
    ProviderConfig,
    get_current_provider_catalog,
)
from msgflux.models.model import Model
from msgflux.models.registry import model_registry, register_model


class FakeBase(BaseModel):
    model_type = "fake_type"
    provider = "fake-base"
    base_url = "https://base.example/v1"
    api_key_env = "FAKE_BASE_API_KEY"
    display_name = "Fake Base"

    def __init__(
        self,
        model_id,
        *,
        marker="base",
        temperature=None,
        base_url=None,
        api_key_env=None,
        fail=False,
    ):
        self.model_id = model_id
        self.marker = marker
        self.temperature = temperature
        self.base_url_value = base_url or self.base_url
        if api_key_env is not None:
            self.api_key_env = api_key_env
        current = get_current_provider_catalog()
        self.ctor_catalog_id = id(current) if current is not None else None
        if fail:
            raise RuntimeError("constructor failed")
        self.runtime = None

    def _initialize(self):
        current = get_current_provider_catalog()
        self.runtime = id(current) if current is not None else None

    def __call__(self):
        return "fake"


@pytest.fixture(autouse=True)
def _register_fake_base():
    register_model(FakeBase)
    try:
        yield
    finally:
        providers = model_registry.get("fake_type", {})
        providers.pop("fake-base", None)
        if not providers:
            model_registry.pop("fake_type", None)


def _factory_module(tmp_path, monkeypatch, name, source):
    path = tmp_path / f"{name}.py"
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    return name


class TestProviderConfig:
    def test_exactly_one_of_base_or_factory_on_construction(self):
        with pytest.raises(ValueError, match="exactly one"):
            ProviderConfig()
        with pytest.raises(ValueError, match="exactly one"):
            ProviderConfig(base="openai", factory="pkg.mod:Cls")
        assert ProviderConfig(base="openai").base == "openai"

    def test_exactly_one_of_base_or_factory_on_decode(self):
        with pytest.raises(msgspec.ValidationError):
            msgspec.convert({}, type=ProviderConfig)
        with pytest.raises(msgspec.ValidationError):
            msgspec.convert(
                {"base": "openai", "factory": "pkg.mod:Cls"}, type=ProviderConfig
            )
        decoded = msgspec.convert({"base": "openai"}, type=ProviderConfig)
        assert decoded.base == "openai"

    def test_frozen_normalization_uses_force_setattr(self):
        # msgspec frozen structs reject object.__setattr__; normalization
        # (strip) must succeed through msgspec.structs.force_setattr.
        config = ProviderConfig(base=" openai ", base_url=" https://x.example/v1 ")
        assert config.base == "openai"
        assert config.base_url == "https://x.example/v1"
        with pytest.raises(AttributeError):
            config.base = "other"

    @pytest.mark.parametrize(
        "options",
        [
            {"api_key": "secret"},
            {"nested": {"access_token": "x"}},
            {"nested": [{"refresh_token": "x"}]},
            {"headers": {"Authorization": "Bearer x"}},
            {"headers": {"x-api-key": "x"}},
            {"password": "x"},
        ],
    )
    def test_options_reject_credential_bearing_fields_recursively(self, options):
        with pytest.raises(ValueError, match="credential"):
            ProviderConfig(base="openai", options=options)

    @pytest.mark.parametrize(
        "options",
        [
            {"model_id": "x"},
            {"provider": "x"},
            {"credential_resolver": "x"},
            {"provider_catalog": "x"},
            {"base_url": "https://x.example"},
            {"api_key_env": "X_API_KEY"},
        ],
    )
    def test_options_reject_reserved_keys(self, options):
        with pytest.raises(ValueError, match="reserved"):
            ProviderConfig(base="openai", options=options)

    def test_invalid_factory_and_url_and_env(self):
        with pytest.raises(ValueError, match="factory"):
            ProviderConfig(factory="not-a-path")
        with pytest.raises(ValueError, match="base_url"):
            ProviderConfig(base="openai", base_url="ftp://x.example")
        with pytest.raises(ValueError, match="api_key_env"):
            ProviderConfig(base="openai", api_key_env="not a env")


class TestDerivedProviders:
    def test_derived_named_subclass_inherits_and_renames(self):
        catalog = ProviderCatalog(
            {
                "my-alias": ProviderConfig(
                    base="fake-base", base_url="https://alias.example/v1"
                )
            }
        )
        cls = catalog.get_model_class("fake_type", "my-alias")
        assert issubclass(cls, FakeBase)
        assert cls is not FakeBase
        assert cls.provider == "my-alias"
        assert cls.__name__ not in {"FakeBase", ""}
        assert "<locals>" not in cls.__qualname__
        assert cls.base_url == "https://alias.example/v1"
        # Cached per catalog, not globally registered.
        assert catalog.get_model_class("fake_type", "my-alias") is cls
        assert "my-alias" not in model_registry["fake_type"]
        # Own account namespace, not the builtin base env.
        assert cls.api_key_env == "MY_ALIAS_API_KEY"

    def test_derived_explicit_env_and_options(self):
        catalog = ProviderCatalog(
            {
                "my-alias": ProviderConfig(
                    base="fake-base",
                    api_key_env="ALIAS_KEY",
                    options={"marker": "from-options", "temperature": 0.5},
                )
            }
        )
        model = catalog.create_model("fake_type", "my-alias", "m1")
        assert model.marker == "from-options"
        assert model.temperature == 0.5
        assert model.api_key_env == "ALIAS_KEY"

    def test_unknown_base_and_cycle(self):
        unknown = ProviderCatalog({"a": ProviderConfig(base="nope")})
        with pytest.raises(ValueError, match="Unknown base"):
            unknown.get_model_class("fake_type", "a")

        cyclic = ProviderCatalog(
            {"a": ProviderConfig(base="b"), "b": ProviderConfig(base="a")}
        )
        with pytest.raises(ValueError, match="cycle"):
            cyclic.get_model_class("fake_type", "a")

    def test_chain_options_leaf_overrides_and_source_chain(self):
        catalog = ProviderCatalog(
            {
                "mid": ProviderConfig(
                    base="fake-base",
                    base_url="https://mid.example/v1",
                    options={"marker": "mid", "temperature": 0.1},
                ),
                "leaf": ProviderConfig(
                    base="mid",
                    options={"temperature": 0.9},
                ),
            }
        )
        model = catalog.create_model("fake_type", "leaf", "m1")
        # Mid options apply, leaf wins on conflicts; mid URL is inherited.
        assert model.marker == "mid"
        assert model.temperature == 0.9
        assert model.base_url_value == "https://mid.example/v1"
        source = model.serialize()["state"]["_provider_source"]
        assert [entry["provider"] for entry in source["chain"]] == ["leaf", "mid"]
        assert source["chain"][1]["base_url"] == "https://mid.example/v1"

    def test_constructor_failure_cleanup(self):
        catalog = ProviderCatalog({"my-alias": ProviderConfig(base="fake-base")})
        with pytest.raises(RuntimeError, match="constructor failed"):
            catalog.create_model("fake_type", "my-alias", "m1", fail=True)
        assert "my-alias" not in model_registry["fake_type"]
        model = catalog.create_model("fake_type", "my-alias", "m1")
        assert model.model_id == "m1"

    def test_independent_catalogs_same_alias(self):
        cat_a = ProviderCatalog(
            {"alias": ProviderConfig(base="fake-base", base_url="https://a.example/v1")}
        )
        cat_b = ProviderCatalog(
            {"alias": ProviderConfig(base="fake-base", base_url="https://b.example/v1")}
        )
        class_a = cat_a.get_model_class("fake_type", "alias")
        class_b = cat_b.get_model_class("fake_type", "alias")
        assert class_a is not class_b
        model_a = cat_a.create_model("fake_type", "alias", "m")
        model_b = cat_b.create_model("fake_type", "alias", "m")
        assert model_a.base_url_value == "https://a.example/v1"
        assert model_b.base_url_value == "https://b.example/v1"
        assert (
            model_a.serialize()["state"]["_provider_source"]
            != (model_b.serialize()["state"]["_provider_source"])
        )
        # Cross-restore must fail.
        with pytest.raises(ValueError, match="does not match"):
            cat_b.from_serialized("alias", "fake_type", model_a.serialize()["state"])

    def test_configured_names_and_get_config(self):
        config = ProviderConfig(base="fake-base", options={"marker": "x"})
        catalog = ProviderCatalog({"alias": config})
        assert catalog.configured_names == ("alias",)
        assert isinstance(catalog.configured_names, tuple)
        # Configs are snapshots: mutating a returned copy (or the original
        # options dict) must not alter the catalog's trusted definition.
        assert catalog.get_config("alias") == config
        assert catalog.get_config("alias") is not config
        assert catalog.get_config("missing") is None
        assert catalog.configured == {"alias": config}
        assert catalog.configured is not catalog.configured
        config.options["marker"] = "tampered"
        assert catalog.get_config("alias").options == {"marker": "x"}
        catalog.get_config("alias").options["marker"] = "tampered-again"
        assert catalog.get_config("alias").options == {"marker": "x"}


class TestFactoryProviders:
    def test_factory_lazy_scoped_and_not_global(self, tmp_path, monkeypatch):
        module = _factory_module(
            tmp_path,
            monkeypatch,
            "ext_factory_mod",
            """
            from msgflux.models.base import BaseModel
            from msgflux.models.registry import register_model

            @register_model
            class ExtModel(BaseModel):
                model_type = "fake_type"
                provider = "ext-provider"
                base_url = "https://ext.example/v1"
                api_key_env = "EXT_API_KEY"

                def __init__(self, model_id, *, base_url=None, api_key_env=None):
                    self.model_id = model_id
                    self.base_url_value = base_url or self.base_url
                    if api_key_env is not None:
                        self.api_key_env = api_key_env

                def _initialize(self):
                    pass

                def __call__(self):
                    return "ext"
            """,
        )
        catalog = ProviderCatalog(
            {"ext-provider": ProviderConfig(factory=f"{module}:ExtModel")}
        )
        assert module not in sys.modules  # lazy: config alone imports nothing
        cls = catalog.get_model_class("fake_type", "ext-provider")
        assert module in sys.modules
        assert cls.provider == "ext-provider"
        model = catalog.create_model("fake_type", "ext-provider", "m1")
        assert model.base_url_value == "https://ext.example/v1"
        # Scoped decorator registration must not contaminate global registry
        # nor a second catalog.
        assert "ext-provider" not in model_registry["fake_type"]
        other = ProviderCatalog()
        with pytest.raises(ValueError, match="not registered"):
            other.get_model_class("fake_type", "ext-provider")
        # But a second catalog with the same trusted factory resolves it.
        second = ProviderCatalog(
            {"ext-provider": ProviderConfig(factory=f"{module}:ExtModel")}
        )
        assert second.get_model_class("fake_type", "ext-provider") is cls

    def test_factory_builtin_autoload_before_scoped_import(self, tmp_path, monkeypatch):
        module = _factory_module(
            tmp_path,
            monkeypatch,
            "ext_factory_autoload",
            """
            from msgflux.models.base import BaseModel
            from msgflux.models.registry import model_registry, register_model

            # Touching the registry inside the factory import must see
            # builtins already autoloaded globally.
            assert "openai" in model_registry["chat_completion"]

            @register_model
            class ExtModel(BaseModel):
                model_type = "fake_type"
                provider = "ext2"

                def __init__(self, model_id):
                    self.model_id = model_id

                def _initialize(self):
                    pass

                def __call__(self):
                    return "ext2"
            """,
        )
        catalog = ProviderCatalog(
            {"ext2": ProviderConfig(factory=f"{module}:ExtModel")}
        )
        catalog.get_model_class("fake_type", "ext2")
        assert "openai" in model_registry["chat_completion"]
        assert "ext2" not in model_registry["fake_type"]
        # The scoped catalog must not have absorbed builtins as its own.
        assert catalog._registrations.get("chat_completion", {}).get("openai") is None

    def test_factory_must_resolve_directly_to_class(self, tmp_path, monkeypatch):
        module = _factory_module(
            tmp_path,
            monkeypatch,
            "ext_factory_callable",
            """
            from msgflux.models.base import BaseModel

            class InnerModel(BaseModel):
                model_type = "fake_type"
                provider = "ext3"

                def __init__(self, model_id):
                    self.model_id = model_id

                def _initialize(self):
                    pass

                def __call__(self):
                    return "inner"

            def make():
                return InnerModel
            """,
        )
        catalog = ProviderCatalog({"ext3": ProviderConfig(factory=f"{module}:make")})
        with pytest.raises(TypeError, match="directly to a model class"):
            catalog.get_model_class("fake_type", "ext3")

    def test_factory_provider_mismatch_and_incompatible(self, tmp_path, monkeypatch):
        module = _factory_module(
            tmp_path,
            monkeypatch,
            "ext_factory_mismatch",
            """
            from msgflux.models.base import BaseModel

            class MismatchModel(BaseModel):
                model_type = "fake_type"
                provider = "someone-else"

                def __init__(self, model_id):
                    self.model_id = model_id

                def _initialize(self):
                    pass

                def __call__(self):
                    return "x"

            class NotAModel:
                pass
            """,
        )
        mismatch = ProviderCatalog(
            {"declared": ProviderConfig(factory=f"{module}:MismatchModel")}
        )
        with pytest.raises(ValueError, match="does not match"):
            mismatch.get_model_class("fake_type", "declared")
        incompatible = ProviderCatalog(
            {"declared": ProviderConfig(factory=f"{module}:NotAModel")}
        )
        with pytest.raises(TypeError, match="BaseModel"):
            incompatible.get_model_class("fake_type", "declared")

    def test_factory_url_env_overrides_subclass_and_restore(
        self, tmp_path, monkeypatch
    ):
        module = _factory_module(
            tmp_path,
            monkeypatch,
            "ext_factory_override",
            """
            from msgflux.models.base import BaseModel

            class PlainModel(BaseModel):
                model_type = "fake_type"
                provider = "ext4"
                base_url = "https://factory.example/v1"
                api_key_env = "FACTORY_API_KEY"

                def __init__(self, model_id):
                    self.model_id = model_id
                    self.url = self.base_url
                    self.env = self.api_key_env

                def _initialize(self):
                    pass

                def __call__(self):
                    return "plain"
            """,
        )
        catalog = ProviderCatalog(
            {
                "ext4": ProviderConfig(
                    factory=f"{module}:PlainModel",
                    base_url="https://override.example/v1",
                    api_key_env="OVERRIDE_API_KEY",
                )
            }
        )
        cls = catalog.get_model_class("fake_type", "ext4")
        plain = importlib.import_module(module).PlainModel
        assert cls is not plain and issubclass(cls, plain)
        assert cls.__name__ not in {"PlainModel", ""}
        model = catalog.create_model("fake_type", "ext4", "m1")
        assert model.url == "https://override.example/v1"
        assert model.env == "OVERRIDE_API_KEY"
        # Shared factory class must not be mutated.
        assert plain.base_url == "https://factory.example/v1"
        restored = catalog.from_serialized(
            "ext4", "fake_type", model.serialize()["state"]
        )
        assert restored.url == "https://override.example/v1"
        assert restored.env == "OVERRIDE_API_KEY"


class TestSerialization:
    def _catalog(self):
        return ProviderCatalog(
            {
                "my-alias": ProviderConfig(
                    base="fake-base",
                    base_url="https://alias.example/v1",
                    options={"marker": "catalog"},
                )
            }
        )

    def test_roundtrip_restores_and_runs_in_use(self):
        catalog = self._catalog()
        model = catalog.create_model("fake_type", "my-alias", "m1", temperature=0.3)
        assert model.ctor_catalog_id == id(catalog)
        serialized = model.serialize()
        state = serialized["state"]
        # Plain data only: no catalog / class / credential objects.
        msgspec.json.encode(state)
        assert "_provider_source" in state
        assert "ProviderCatalog" not in msgspec.json.encode(state).decode()
        restored = catalog.from_serialized("my-alias", "fake_type", state)
        assert restored.model_id == "m1"
        assert restored.marker == "catalog"
        assert restored.temperature == 0.3
        assert restored.runtime == id(catalog)

    def test_model_entrypoints_delegate_to_catalog(self, monkeypatch):
        monkeypatch.setenv("CHAT_ALIAS_API_KEY", "test-key")
        catalog = self._catalog()
        model = (
            Model.chat_completion(
                "my-alias/m1", provider_catalog=catalog, temperature=0.2
            )
            if False
            else catalog.create_model("fake_type", "my-alias", "m1")
        )
        # Use a real chat-type alias via the Model API with a builtin base.
        chat_catalog = ProviderCatalog({"chat-alias": ProviderConfig(base="nvidia")})
        chat = Model.chat_completion(
            "chat-alias/some-model", provider_catalog=chat_catalog
        )
        assert chat.provider == "chat-alias"
        assert chat.api_key_env == "CHAT_ALIAS_API_KEY"
        state = chat.serialize()["state"]
        restored = Model.from_serialized(
            "chat-alias", "chat_completion", state, provider_catalog=chat_catalog
        )
        assert restored.provider == "chat-alias"
        # Nested use() supplies the catalog for Model resolution too.
        with chat_catalog.use():
            nested = Model.chat_completion("chat-alias/other-model")
            assert nested.provider == "chat-alias"
        assert model.marker == "catalog"

    def test_changed_intermediate_fails_restore(self):
        catalog = self._catalog()
        model = catalog.create_model("fake_type", "my-alias", "m1")
        changed = ProviderCatalog(
            {
                "my-alias": ProviderConfig(
                    base="fake-base",
                    base_url="https://changed.example/v1",
                    options={"marker": "catalog"},
                )
            }
        )
        with pytest.raises(ValueError, match="does not match"):
            changed.from_serialized("my-alias", "fake_type", model.serialize()["state"])

    def test_missing_config_fails_restore(self):
        catalog = self._catalog()
        model = catalog.create_model("fake_type", "my-alias", "m1")
        state = model.serialize()["state"]
        with pytest.raises(ValueError, match="requires a"):
            ProviderCatalog().from_serialized("my-alias", "fake_type", state)
        with pytest.raises(ValueError, match="requires a trusted"):
            Model.from_serialized("my-alias", "fake_type", state)

    def test_tampered_source_never_imports_factory(self, tmp_path, monkeypatch):
        module = _factory_module(
            tmp_path,
            monkeypatch,
            "ext_factory_tamper",
            """
            from msgflux.models.base import BaseModel

            class TamperModel(BaseModel):
                model_type = "fake_type"
                provider = "tamper"

                def __init__(self, model_id):
                    self.model_id = model_id

                def _initialize(self):
                    pass

                def __call__(self):
                    return "tamper"
            """,
        )
        catalog = ProviderCatalog(
            {"tamper": ProviderConfig(factory=f"{module}:TamperModel")}
        )
        model = catalog.create_model("fake_type", "tamper", "m1")
        state = copy.deepcopy(model.serialize()["state"])
        state["_provider_source"]["factory"] = "nonexistent_module_xyz:Cls"
        sys.modules.pop("nonexistent_module_xyz", None)
        with pytest.raises(ValueError, match="does not match"):
            catalog.from_serialized("tamper", "fake_type", state)
        assert "nonexistent_module_xyz" not in sys.modules

    def test_missing_source_metadata_fails_for_configured(self):
        catalog = self._catalog()
        model = catalog.create_model("fake_type", "my-alias", "m1")
        state = copy.deepcopy(model.serialize()["state"])
        state.pop("_provider_source")
        with pytest.raises(ValueError, match="missing"):
            catalog.from_serialized("my-alias", "fake_type", state)


class TestScopedRegistration:
    def test_catalog_register_isolation_and_replace(self):
        catalog = ProviderCatalog()
        other = ProviderCatalog()

        class LocalA(BaseModel):
            model_type = "local_type"
            provider = "local"

            def _initialize(self):
                pass

            def __call__(self):
                return "a"

        class LocalB(BaseModel):
            model_type = "local_type"
            provider = "local"

            def _initialize(self):
                pass

            def __call__(self):
                return "b"

        assert catalog.register(LocalA) is LocalA
        assert catalog.register(LocalA) is LocalA
        assert catalog.get_model_class("local_type", "local") is LocalA
        with pytest.raises(ValueError, match="already registered"):
            catalog.register(LocalB)
        catalog.register(LocalB, replace=True)
        assert catalog.get_model_class("local_type", "local") is LocalB
        with pytest.raises(ValueError, match=r"not supported|not registered"):
            other.get_model_class("local_type", "local")


class TestBuiltinPreservation:
    def test_builtin_and_multimodal_registry_unchanged(self):
        assert Model._get_model_class("chat_completion", "nvidia").provider == "nvidia"
        embedder = Model._get_model_class("text_embedder", "openai")
        assert embedder.provider == "openai"
        assert embedder.model_type == "text_embedder"
        assert "chat-alias" not in str(Model.providers())
        # use() alone must not redirect builtin decorators to the catalog.
        catalog = ProviderCatalog()
        with catalog.use():
            assert get_current_provider_catalog() is catalog

            @register_model
            class StillGlobal(BaseModel):
                model_type = "still_global_type"
                provider = "still-global"

                def _initialize(self):
                    pass

                def __call__(self):
                    return "global"

        try:
            assert model_registry["still_global_type"]["still-global"] is StillGlobal
        finally:
            model_registry.pop("still_global_type", None)

    def test_codex_explicit_api_key_env_uses_bearer(self, monkeypatch):
        from msgflux.models.model_credentials import (
            APIKeyCredentialResolver,
            BearerTokenCredentialResolver,
        )

        monkeypatch.setenv("CODEX_ALIAS_KEY", "test-key-value")
        catalog = ProviderCatalog(
            {
                "codex-alias": ProviderConfig(
                    base="openai-codex", api_key_env="CODEX_ALIAS_KEY"
                )
            }
        )
        model = Model.chat_completion("codex-alias/gpt-5.6", provider_catalog=catalog)
        assert isinstance(model.credential_resolver, BearerTokenCredentialResolver)
        assert isinstance(model.credential_resolver, APIKeyCredentialResolver)
        assert model.provider_auth_method == "api_key"
        assert model._get_api_key() == "test-key-value"
        # Codex transport/codec are preserved.
        assert model.api_mode == "responses"
        assert model.reasoning_codec is not None

    def test_codex_oauth_alias_keeps_oauth_marker(self):
        from msgflux.models.providers.openai_codex import OpenAICodexChatCompletion

        catalog = ProviderCatalog(
            {"codex-oauth-alias": ProviderConfig(base="openai-codex")}
        )
        cls = catalog.get_model_class("chat_completion", "codex-oauth-alias")
        assert cls.provider_auth_method == "oauth"
        assert issubclass(cls, OpenAICodexChatCompletion)
        assert isinstance(
            cls.credential_resolver,
            OpenAICodexChatCompletion.credential_resolver.__class__,
        )


class TestOpenCodeAliasBackend:
    @pytest.mark.parametrize(
        "base,alias,builtin_env,builtin_url_env,custom_url,anthropic_url",
        [
            (
                "opencode-go",
                "my-go",
                "OPENCODE_GO_API_KEY",
                "OPENCODE_GO_BASE_URL",
                "https://alias.example/go/openai/v1",
                "https://alias.example/go/anthropic",
            ),
            (
                "opencode-zen",
                "my-zen",
                "OPENCODE_ZEN_API_KEY",
                "OPENCODE_ZEN_BASE_URL",
                "https://alias.example/openai/v1",
                "https://alias.example/anthropic",
            ),
        ],
    )
    @pytest.mark.parametrize(
        "api_mode", ["chat_completions", "responses", "anthropic_messages"]
    )
    def test_alias_modes_preserve_provider_env_and_restore(
        self,
        monkeypatch,
        base,
        alias,
        builtin_env,
        builtin_url_env,
        custom_url,
        anthropic_url,
        api_mode,
    ):
        from msgflux.models.model_credentials import BearerTokenCredentialResolver
        from msgflux.models.providers import opencode
        from msgflux.models.reasoning import AnthropicThinkingCodec

        alias_env = f"{alias.upper().replace('-', '_')}_API_KEY"
        monkeypatch.delenv(builtin_url_env, raising=False)
        monkeypatch.setenv(builtin_env, "builtin-key")
        monkeypatch.setenv(alias_env, "alias-key")
        catalog = ProviderCatalog(
            {
                alias: ProviderConfig(
                    base=base, base_url=custom_url, api_key_env=alias_env
                )
            }
        )
        model = catalog.create_model(
            "chat_completion", alias, "some-model", api_mode=api_mode
        )
        assert model.provider == alias
        assert model.model_type == "chat_completion"
        assert model.api_key_env == alias_env
        assert model._get_api_key() == "alias-key"
        assert isinstance(model.credential_resolver, BearerTokenCredentialResolver)
        assert model.api_mode == api_mode
        if api_mode == "anthropic_messages":
            original_backend = getattr(
                opencode,
                "_OpenCodeGoAnthropicMessages"
                if base == "opencode-go"
                else "_OpenCodeZenAnthropicMessages",
            )
            assert isinstance(model, original_backend)
            assert type(model) is not original_backend
            assert model._get_base_url() == anthropic_url
            assert "x-opencode-session" in model._native_headers()
            assert isinstance(model.reasoning_codec, AnthropicThinkingCodec)
        else:
            assert model.sampling_params["base_url"] == custom_url
        # Serialize/restore preserves alias backend identity.
        state = model.serialize()["state"]
        restored = catalog.from_serialized(alias, "chat_completion", state)
        assert restored.provider == alias
        assert restored.model_type == "chat_completion"
        assert restored.api_key_env == alias_env
        assert restored._get_api_key() == "alias-key"
        assert isinstance(restored.credential_resolver, BearerTokenCredentialResolver)
        assert restored.api_mode == api_mode
        if api_mode == "anthropic_messages":
            assert restored._get_base_url() == anthropic_url
            assert "x-opencode-session" in restored._native_headers()
            assert isinstance(restored.reasoning_codec, AnthropicThinkingCodec)

    def test_alias_default_env_and_builtin_unchanged(self, monkeypatch):
        monkeypatch.delenv("OPENCODE_ZEN_BASE_URL", raising=False)
        monkeypatch.setenv("OPENCODE_ZEN_API_KEY", "builtin-key")
        monkeypatch.setenv("ZEN_ALIAS_API_KEY", "zen-alias-key")
        catalog = ProviderCatalog({"zen-alias": ProviderConfig(base="opencode-zen")})
        model = catalog.create_model(
            "chat_completion", "zen-alias", "m", api_mode="anthropic_messages"
        )
        assert model.provider == "zen-alias"
        assert model.api_key_env == "ZEN_ALIAS_API_KEY"
        assert model._get_api_key() == "zen-alias-key"
        # Builtin provider still uses its own env/backend.
        from msgflux.models.providers.opencode import OpenCodeZenChatCompletion

        assert OpenCodeZenChatCompletion.api_key_env == "OPENCODE_ZEN_API_KEY"

    def test_factory_facade_override_anthropic(self, tmp_path, monkeypatch):
        module = _factory_module(
            tmp_path,
            monkeypatch,
            "ext_zen_factory",
            """
            from msgflux.models.providers.opencode import OpenCodeZenChatCompletion

            class MyZenFactory(OpenCodeZenChatCompletion):
                provider = "my-zen-factory"
                display_name = "My Zen Factory"
                api_key_env = "MY_ZEN_FACTORY_KEY"
                base_url = "https://factory.example/openai/v1"
            """,
        )
        monkeypatch.delenv("OPENCODE_ZEN_BASE_URL", raising=False)
        monkeypatch.setenv("OPENCODE_ZEN_API_KEY", "builtin-key")
        monkeypatch.setenv("OVERRIDE_ZEN_KEY", "override-key")
        catalog = ProviderCatalog(
            {
                "my-zen-factory": ProviderConfig(
                    factory=f"{module}:MyZenFactory",
                    base_url="https://override.example/openai/v1",
                    api_key_env="OVERRIDE_ZEN_KEY",
                )
            }
        )
        model = catalog.create_model(
            "chat_completion",
            "my-zen-factory",
            "m",
            api_mode="anthropic_messages",
        )
        assert model.provider == "my-zen-factory"
        assert model.api_key_env == "OVERRIDE_ZEN_KEY"
        assert model._get_api_key() == "override-key"
        assert model._get_base_url() == "https://override.example/anthropic"
        assert "x-opencode-session" in model._native_headers()
        restored = catalog.from_serialized(
            "my-zen-factory", "chat_completion", model.serialize()["state"]
        )
        assert restored.provider == "my-zen-factory"
        assert restored.api_key_env == "OVERRIDE_ZEN_KEY"
        assert restored._get_api_key() == "override-key"
        assert restored._get_base_url() == "https://override.example/anthropic"
        assert "x-opencode-session" in restored._native_headers()


class TestFactoryCodexApiKey:
    def test_factory_codex_api_env_create_and_restore(self, tmp_path, monkeypatch):
        module = _factory_module(
            tmp_path,
            monkeypatch,
            "ext_codex_factory",
            """
            from msgflux.models.providers.openai_codex import OpenAICodexChatCompletion

            class MyCodexFactory(OpenAICodexChatCompletion):
                provider = "my-codex-factory"
                display_name = "My Codex Factory"
                api_key_env = "MY_CODEX_FACTORY_KEY"
            """,
        )
        from msgflux.models.model_credentials import BearerTokenCredentialResolver
        from msgflux.models.providers.openai_codex import (
            CodexCredentialResolver,
            OpenAICodexChatCompletion,
        )

        monkeypatch.setenv("MY_CODEX_FACTORY_KEY", "codex-factory-key")
        catalog = ProviderCatalog(
            {"my-codex-factory": ProviderConfig(factory=f"{module}:MyCodexFactory")}
        )
        model = catalog.create_model("chat_completion", "my-codex-factory", "gpt-5.6")
        assert model.provider_auth_method == "api_key"
        assert isinstance(model.credential_resolver, BearerTokenCredentialResolver)
        assert not isinstance(model.credential_resolver, CodexCredentialResolver)
        assert model._get_api_key() == "codex-factory-key"
        assert model.api_mode == "responses"
        assert model.reasoning_codec is not None
        restored = catalog.from_serialized(
            "my-codex-factory", "chat_completion", model.serialize()["state"]
        )
        assert restored.provider_auth_method == "api_key"
        assert isinstance(restored.credential_resolver, BearerTokenCredentialResolver)
        assert restored._get_api_key() == "codex-factory-key"
        # Shared factory class must not be mutated.
        original = importlib.import_module(module).MyCodexFactory
        assert isinstance(original.credential_resolver, CodexCredentialResolver)
        assert getattr(original, "provider_auth_method", None) is None
        assert issubclass(original, OpenAICodexChatCompletion)

    def test_factory_codex_custom_resolver_preserved(self, tmp_path, monkeypatch):
        module = _factory_module(
            tmp_path,
            monkeypatch,
            "ext_codex_custom",
            """
            from msgflux.models.model_credentials import BearerTokenCredentialResolver
            from msgflux.models.providers.openai_codex import OpenAICodexChatCompletion

            class CustomResolver(BearerTokenCredentialResolver):
                pass

            class MyCodexCustom(OpenAICodexChatCompletion):
                provider = "my-codex-custom"
                api_key_env = "MY_CODEX_CUSTOM_KEY"
                credential_resolver = CustomResolver()
            """,
        )
        monkeypatch.setenv("MY_CODEX_CUSTOM_KEY", "custom-key")
        catalog = ProviderCatalog(
            {"my-codex-custom": ProviderConfig(factory=f"{module}:MyCodexCustom")}
        )
        model = catalog.create_model("chat_completion", "my-codex-custom", "gpt-5.6")
        custom_cls = importlib.import_module(module).CustomResolver
        assert isinstance(model.credential_resolver, custom_cls)
        assert model._get_api_key() == "custom-key"


class TestInstanceIdentityMismatch:
    def test_actual_instance_mismatch_cleanup(self, tmp_path, monkeypatch):
        module = _factory_module(
            tmp_path,
            monkeypatch,
            "ext_mismatch_new",
            """
            from msgflux.models.base import BaseModel

            CREATED = []

            class OtherModel(BaseModel):
                model_type = "fake_type"
                provider = "other-provider"

                def __init__(self, model_id):
                    self.model_id = model_id
                    self.closed = False

                def _initialize(self):
                    pass

                def close(self):
                    self.closed = True

                def __call__(self):
                    return "other"

            class MismatchModel(BaseModel):
                model_type = "fake_type"
                provider = "mismatch-provider"

                def __new__(cls, model_id, **kwargs):
                    inst = OtherModel(model_id)
                    CREATED.append(inst)
                    return inst

                def _initialize(self):
                    pass

                def __call__(self):
                    return "mismatch"
            """,
        )
        catalog = ProviderCatalog(
            {"mismatch-provider": ProviderConfig(factory=f"{module}:MismatchModel")}
        )
        with pytest.raises(ValueError, match="does not match"):
            catalog.create_model("fake_type", "mismatch-provider", "m1")
        created = importlib.import_module(module).CREATED
        assert len(created) == 1
        assert created[0].closed is True
