"""Model gateway catalog: parsing, inheritance, validation, and inspection."""

import asyncio
import sys
from types import SimpleNamespace

import msgspec
import pytest

from msgflux.coding.config import (
    CodingModelDeployment,
    coding_agent_options,
)
from msgflux.runtime.service.local.agent_inspection import (
    inspect_registered_agent,
    manifest_agent_configuration,
)
from msgflux.runtime.service.local.manifest import load_service_manifest
from msgflux.runtime.service.records import AgentConfiguration, AgentInspection

GATEWAY_TOML = """
[defaults]
agent = 'coding'

[models.fast]
model = 'openai-codex/fast-model'
description = 'Fast default model'

[models.smart]
model = 'other/smart-model'
description = 'Smart fallback model'
reasoning_effort = 'high'
account_name = 'work'

[agents.coding]
preset = 'coding'
models = ['fast', 'smart']
model_fallback = true
"""


def _write(tmp_path, content):
    path = tmp_path / "config.toml"
    path.write_text(content)
    return path


def test_gateway_parsing_first_alias_default_and_inheritance(tmp_path):
    manifest = load_service_manifest(_write(tmp_path, GATEWAY_TOML))
    options = manifest.coding_catalog.agents["coding"]

    assert [d.model_name for d in options.models] == ["fast", "smart"]
    # options.model is the first deployment's path for backwards compatibility.
    assert options.model == "openai-codex/fast-model"
    assert options.model_fallback is True
    fast, smart = options.models
    assert fast.description == "Fast default model"
    assert fast.reasoning_effort is None and fast.account_name is None
    assert smart.reasoning_effort == "high" and smart.account_name == "work"
    # Mixed providers: workspace expands per provider, unioned in order.
    assert options.tools_active_selectors == ("workspace",)
    assert options.tools_deferred_selectors == ()
    assert options.tools_active == (
        "bash",
        "read",
        "apply_patch",
        "edit",
        "write",
        "delete",
    )
    assert options.tools_deferred == ()


def test_neither_model_nor_models_uses_defaults_model(tmp_path):
    path = _write(
        tmp_path,
        "[defaults]\nagent='coding'\nmodel='provider/model-id'\n"
        "[agents.coding]\npreset='coding'\n",
    )
    options = load_service_manifest(path).coding_catalog.agents["coding"]
    assert options.model == "provider/model-id"
    assert options.models == () and options.model_fallback is False
    # Ordinary single-model behavior is unchanged.
    assert options.tools_active_selectors is None
    assert options.tools_deferred_selectors is None
    assert options.tools_active == ("bash", "read", "edit", "write", "delete")


def test_scalar_model_ignores_defaults_and_stays_single(tmp_path):
    path = _write(
        tmp_path,
        "[defaults]\nagent='coding'\nmodel='provider/default'\n"
        "[agents.coding]\npreset='coding'\nmodel='provider/explicit'\n",
    )
    options = load_service_manifest(path).coding_catalog.agents["coding"]
    assert options.model == "provider/explicit"
    assert options.models == ()


@pytest.mark.parametrize(
    "content, match",
    [
        # Explicit scalar model conflicts with agent models.
        (
            GATEWAY_TOML.replace(
                "models = ['fast', 'smart']",
                "model = 'provider/x'\nmodels = ['fast', 'smart']",
            ),
            "both model and models",
        ),
        # Unknown alias.
        (
            GATEWAY_TOML.replace("['fast', 'smart']", "['fast', 'missing']"),
            "unknown model",
        ),
        # Duplicate alias.
        (GATEWAY_TOML.replace("['fast', 'smart']", "['fast', 'fast']"), "duplicate"),
        # Empty alias list.
        (GATEWAY_TOML.replace("['fast', 'smart']", "[]"), "must not be empty"),
        # Fallback without models (explicit opt-in only).
        (
            "[defaults]\nagent='coding'\nmodel='provider/m'\n"
            "[agents.coding]\npreset='coding'\nmodel_fallback=true\n",
            "model_fallback requires models",
        ),
        # Missing required description.
        (
            "[defaults]\nagent='coding'\n[models.a]\nmodel='provider/m'\n"
            "[agents.coding]\npreset='coding'\nmodels=['a']\n",
            "description",
        ),
        # Empty description.
        (
            GATEWAY_TOML.replace(
                "description = 'Fast default model'", "description = '  '"
            ),
            "description must be non-empty",
        ),
        # Invalid account name.
        (
            GATEWAY_TOML.replace("account_name = 'work'", "account_name = 'bad name'"),
            "account_name is invalid",
        ),
        # Invalid model path.
        (
            GATEWAY_TOML.replace("'openai-codex/fast-model'", "'no-slash'"),
            "provider/model-id",
        ),
        # Custom factories reject pool fields.
        (
            "[defaults]\nagent='custom'\n"
            "[agents.custom]\nfactory='tests.test_local_service_runner:local_factory'\n"
            "models=['fast']\n",
            "cannot use a preset or coding options",
        ),
    ],
)
def test_manifest_rejects_invalid_gateway_config(tmp_path, content, match):
    with pytest.raises(ValueError, match=match):
        load_service_manifest(_write(tmp_path, content))


def test_coding_agent_options_direct_validation():
    deployment = CodingModelDeployment(
        model_name="a", model="provider/m", description="A model"
    )
    options = coding_agent_options("agent", models=(deployment,))
    assert options.model == "provider/m" and options.models == (deployment,)
    # Absent fallback defaults to False; explicit False without models is fine.
    assert options.model_fallback is False
    assert (
        coding_agent_options("agent", "provider/m", model_fallback=False).models == ()
    )

    with pytest.raises(ValueError, match="mutually exclusive"):
        coding_agent_options("agent", "provider/m", models=(deployment,))
    with pytest.raises(ValueError, match="must not be empty"):
        coding_agent_options("agent", models=())
    with pytest.raises(ValueError, match="model_fallback requires models"):
        coding_agent_options("agent", "provider/m", model_fallback=True)
    with pytest.raises(ValueError, match="duplicate model alias"):
        coding_agent_options("agent", models=(deployment, deployment))
    with pytest.raises(ValueError, match="description must be non-empty"):
        coding_agent_options(
            "agent",
            models=(
                CodingModelDeployment(
                    model_name="a", model="provider/m", description=" "
                ),
            ),
        )
    with pytest.raises(ValueError, match="account_name is invalid"):
        coding_agent_options(
            "agent",
            models=(
                CodingModelDeployment(
                    model_name="a",
                    model="provider/m",
                    description="A model",
                    account_name="bad name",
                ),
            ),
        )
    with pytest.raises(ValueError, match="reasoning_effort must be non-empty"):
        coding_agent_options(
            "agent",
            models=(
                CodingModelDeployment(
                    model_name="a",
                    model="provider/m",
                    description="A model",
                    reasoning_effort="",
                ),
            ),
        )


def test_selector_preservation_and_union_ordering():
    options = coding_agent_options(
        "agent",
        models=(
            CodingModelDeployment(
                model_name="one", model="openai-codex/m1", description="One"
            ),
            CodingModelDeployment(
                model_name="two", model="other/m2", description="Two"
            ),
        ),
        active=("workspace", "web_fetch"),
        deferred=("web_search",),
    )
    assert options.tools_active_selectors == ("workspace", "web_fetch")
    assert options.tools_deferred_selectors == ("web_search",)
    assert options.tools_active == (
        "bash",
        "read",
        "apply_patch",
        "web_fetch",
        "edit",
        "write",
        "delete",
    )
    assert options.tools_deferred == ("web_search",)
    # Resolution is idempotent on an already-resolved gateway options object.
    from msgflux.coding.config import resolve_coding_agent_options

    assert resolve_coding_agent_options(options) == options


def test_gateway_rejects_active_deferred_overlap():
    with pytest.raises(ValueError, match="cannot be both active and deferred"):
        coding_agent_options(
            "agent",
            models=(
                CodingModelDeployment(
                    model_name="a", model="provider/m", description="A model"
                ),
            ),
            active=("read",),
            deferred=("read",),
        )


def test_inspection_exposes_deployments_without_provider_construction(tmp_path):
    manifest = load_service_manifest(_write(tmp_path, GATEWAY_TOML))
    result = manifest_agent_configuration(manifest, "coding")

    assert result.model == "openai-codex/fast-model"
    assert result.model_fallback is True
    assert [d.model_name for d in result.models] == ["fast", "smart"]
    assert result.models[0].is_default is True
    assert result.models[1].is_default is False
    assert result.models[0].account_name is None  # inherits execution account
    assert result.models[1].account_name == "work"
    assert result.models[1].reasoning_effort == "high"
    assert result.models[1].description == "Smart fallback model"


def test_inspection_text_shows_default_and_account_inheritance(tmp_path, monkeypatch):
    manifest = load_service_manifest(_write(tmp_path, GATEWAY_TOML))
    configuration = manifest_agent_configuration(manifest, "coding")

    class _Client:
        async def inspect_agent(self, agent_id):
            return AgentInspection(
                agent_id=agent_id,
                is_default=True,
                account_name="default",
                configuration=configuration,
            )

    written = []
    monkeypatch.setattr(sys.stdout, "write", written.append)
    asyncio.run(
        inspect_registered_agent(
            _Client(), SimpleNamespace(agent_id="coding", json=False)
        )
    )
    text = "".join(written)
    assert "Model fallback: yes" in text
    assert "fast: openai-codex/fast-model (default)" in text
    assert "account=inherited execution account" in text
    assert "smart: other/smart-model" in text
    assert "account=work" in text


def test_old_configuration_payloads_get_safe_defaults():
    old = msgspec.convert({"model": "provider/m"}, type=AgentConfiguration, strict=True)
    assert old.models == () and old.model_fallback is False
    assert old.model == "provider/m"

    roundtripped = msgspec.convert(
        msgspec.to_builtins(old), type=AgentConfiguration, strict=True
    )
    assert roundtripped == old


@pytest.mark.parametrize("bad", [None, {}, "fast"])
def test_python_catalog_rejects_invalid_deployments_before_access(bad):
    with pytest.raises(TypeError, match="CodingModelDeployment"):
        coding_agent_options("worker", models=(bad,))
