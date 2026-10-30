from pathlib import Path

import pytest

from msgflux.coding.config import coding_agent_options
from msgflux.runtime.service.local.manifest import load_service_manifest

_CODING_DEFAULTS = """[defaults]
agent = "coding"
model = "provider/model-id"

[agents.coding]
preset = "coding"
"""


def test_coding_preset_is_neutral_and_keeps_generic_defaults(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text(_CODING_DEFAULTS)

    coding = load_service_manifest(config).coding_catalog.agents["coding"]

    assert coding == coding_agent_options(
        "coding", "provider/model-id", preset="coding"
    )
    assert coding.preset == "coding"
    # A neutral preset adds no role instructions or description.
    assert coding.description is None and coding.instructions is None
    assert coding.permissions == ("read", "write", "execute")
    assert coding.tools_active == ("bash", "read", "edit", "write", "delete")
    assert coding.tools_deferred == ()
    assert coding.subagents == ()


def test_explorer_defaults_and_explicit_overrides_share_the_python_contract(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text("""[defaults]
agent = "coding"
model = "provider/model-id"
[agents.coding]
preset = "coding"
subagents = ["explorer"]
[agents.explorer]
preset = "explorer"
""")
    explorer = load_service_manifest(config).coding_catalog.agents["explorer"]
    assert explorer == coding_agent_options(
        "explorer", "provider/model-id", preset="explorer"
    )
    assert explorer.permissions == ("read",)
    assert explorer.tools_active == (
        "read",
        "ls",
        "glob",
        "grep",
        "web_fetch",
        "web_search",
    )
    assert "trace dependencies" in explorer.description
    assert explorer.tools_deferred == ()
    assert explorer.preset == "explorer"
    customized = coding_agent_options(
        "reviewer",
        "other/model",
        preset="explorer",
        description="Review changes",
        active=(),
        permissions=(),
        prompt_file=Path("review.md"),
    )
    assert customized.permissions == () and customized.tools_active == ()
    assert customized.description == "Review changes"
    assert customized.prompt_file == Path("review.md")


def test_python_contract_accepts_unspecified_preset():
    options = coding_agent_options("plain", "provider/model-id")
    assert options.preset is None
    assert options.description is None and options.instructions is None
    assert options.permissions == ("read", "write", "execute")
    assert options.tools_active == ("bash", "read", "edit", "write", "delete")


@pytest.mark.parametrize(
    "agent_body, match",
    [
        ('preset = "unknown"\n', "Unknown coding preset 'unknown'"),
        ('preset = ""\n', "preset must be non-empty"),
        ('preset = "coding"\ndescription = ""\n', "description must be non-empty"),
    ],
)
def test_preset_manifest_rejects_invalid_defaults(tmp_path, agent_body, match):
    config = tmp_path / "config.toml"
    config.write_text(
        '[defaults]\nagent = "a"\nmodel = "provider/model-id"\n'
        f"[agents.a]\n{agent_body}"
    )
    with pytest.raises(ValueError, match=match):
        load_service_manifest(config)


@pytest.mark.parametrize(
    "body, match",
    [
        # A missing selector cannot be inferred from other fields.
        (
            '[defaults]\nagent="a"\nmodel="p/m"\n[agents.a]\nsubagents=[]\n',
            "exactly one preset or factory",
        ),
        # Legacy TOML selector is rejected instead of being rewritten.
        (
            '[defaults]\nagent="a"\nmodel="p/m"\n[agents.a]\nharness="coding"\n',
            "unknown field `harness`",
        ),
        # Exactly one selector per agent.
        (
            '[defaults]\nagent="a"\nmodel="p/m"\n[agents.a]\npreset="coding"\n'
            'factory="tests.test_local_service_runner:local_factory"\n',
            "exactly one preset or factory",
        ),
        # Custom factories keep their own construction.
        (
            '[defaults]\nagent="a"\n[agents.a]\n'
            'factory="tests.test_local_service_runner:local_factory"\n'
            'description="d"\n',
            "cannot use a preset or coding options",
        ),
        (
            '[defaults]\nagent="a"\n[agents.a]\n'
            'factory="tests.test_local_service_runner:local_factory"\n'
            "[agents.a.tools]\nactive=[]\n",
            "cannot use a preset or coding options",
        ),
        (
            '[defaults]\nagent="a"\n[agents.a]\n'
            'factory="tests.test_local_service_runner:local_factory"\n'
            'subagents=["a"]\n',
            "cannot use a preset or coding options",
        ),
        # Preset entries cannot select a custom factory as a subagent.
        (
            '[defaults]\nagent="a"\nmodel="p/m"\n[agents.a]\npreset="coding"\n'
            'subagents=["b"]\n[agents.b]\n'
            'factory="tests.test_local_service_runner:local_factory"\n',
            "must use a coding preset",
        ),
    ],
)
def test_manifest_requires_exactly_one_known_selector(tmp_path, body, match):
    config = tmp_path / "config.toml"
    config.write_text(body)
    with pytest.raises(ValueError, match=match):
        load_service_manifest(config)


def test_preset_tool_defaults_survive_partial_tool_table(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text("""[defaults]
agent="explorer"
model="p/m"
[agents.explorer]
preset="explorer"
[agents.explorer.tools]
deferred=[]
""")
    options = load_service_manifest(config).coding_catalog.agents["explorer"]
    assert options.tools_active == (
        "read",
        "ls",
        "glob",
        "grep",
        "web_fetch",
        "web_search",
    )
    assert options.tools_deferred == ()


@pytest.mark.parametrize(
    "model,tools",
    [
        ("openai/model", ("bash", "read", "apply_patch")),
        ("anthropic/model", ("bash", "read", "edit", "write", "delete")),
        ("opencode-go/openai/model", ("bash", "read", "apply_patch")),
    ],
)
def test_builder_defaults_keep_native_editing_without_redundant_search(model, tools):
    builder = coding_agent_options("implementer", model, preset="builder")
    assert builder.preset == "builder"
    assert builder.permissions == ("read", "write", "execute")
    assert builder.tools_active == tools
    assert not builder.tools_deferred
    assert not set(builder.tools_active) & {
        "web_fetch",
        "web_search",
        "ls",
        "glob",
        "grep",
    }
    assert "effective workspace permissions" in builder.instructions


def test_explicit_builder_tools_and_permissions_replace_preset_defaults():
    options = coding_agent_options(
        "custom-builder",
        "provider/model",
        preset="builder",
        active=("read",),
        permissions=("read",),
        deferred=(),
    )
    assert options.tools_active == ("read",)
    assert options.permissions == ("read",)
    assert options.preset == "builder"
