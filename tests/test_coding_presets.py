from pathlib import Path

import pytest

from msgflux.coding.config import coding_agent_options
from msgflux.runtime.service.local.manifest import load_service_manifest


def test_explorer_defaults_and_explicit_overrides_share_the_python_contract(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text("""[defaults]
agent = "coding"
model = "provider/model-id"
[agents.coding]
harness = "coding"
subagents = ["explorer"]
[agents.explorer]
harness = "coding"
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


@pytest.mark.parametrize("extra", ['preset="unknown"', 'description=""'])
def test_preset_manifest_rejects_invalid_defaults(tmp_path, extra):
    config = tmp_path / "config.toml"
    config.write_text(
        f'[defaults]\nagent="a"\nmodel="p/m"\n[agents.a]\nharness="coding"\n{extra}\n'
    )
    with pytest.raises(ValueError):
        load_service_manifest(config)


def test_preset_tool_defaults_survive_partial_tool_table(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text("""[defaults]
agent="explorer"
model="p/m"
[agents.explorer]
harness="coding"
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
