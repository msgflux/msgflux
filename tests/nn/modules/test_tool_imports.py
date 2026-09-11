import subprocess
import sys

from msgflux.nn.modules.tool import LocalTool, MCPTool, Tool, ToolLibrary
from msgflux.nn.modules.tool.implementations import (
    LocalTool as FragmentedLocalTool,
)
from msgflux.nn.modules.tool.implementations import MCPTool as FragmentedMCPTool
from msgflux.nn.modules.tool.implementations import Tool as FragmentedTool
from msgflux.nn.modules.tool.library import ToolLibrary as FragmentedToolLibrary
from msgflux.nn.modules.tool_runtime import (
    ToolCatalogEntry as LegacyToolCatalogEntry,
)
from msgflux.nn.modules.tool_runtime import ToolCatalogView as LegacyToolCatalogView
from msgflux.nn.modules.tool_runtime import ToolChoice as LegacyToolChoice
from msgflux.nn.modules.tool_runtime import ToolRef as LegacyToolRef
from msgflux.tools import ToolCatalogEntry, ToolCatalogView, ToolChoice, ToolRef


def test_fragmented_tool_package_preserves_public_class_identity():
    assert Tool is FragmentedTool
    assert LocalTool is FragmentedLocalTool
    assert MCPTool is FragmentedMCPTool
    assert ToolLibrary is FragmentedToolLibrary


def test_legacy_runtime_exports_resolve_to_canonical_catalog_contracts():
    assert LegacyToolCatalogEntry is ToolCatalogEntry
    assert LegacyToolCatalogView is ToolCatalogView
    assert LegacyToolChoice is ToolChoice
    assert LegacyToolRef is ToolRef


def test_model_provider_import_does_not_load_nn_tool_runtime():
    script = """
import sys
import msgflux.models.providers.openai

assert "msgflux.nn.modules.tool" not in sys.modules
assert not any(name.startswith("msgflux.nn.modules.tool.") for name in sys.modules)
"""
    subprocess.run([sys.executable, "-c", script], check=True)  # noqa: S603
