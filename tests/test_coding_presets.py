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
    assert explorer.tools_active == ("read", "ls", "glob", "grep")
    assert "trace dependencies" in explorer.description
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
deferred=["web_fetch"]
""")
    options = load_service_manifest(config).coding_catalog.agents["explorer"]
    assert options.tools_active == ("read", "ls", "glob", "grep")
    assert options.tools_deferred == ("web_fetch",)
