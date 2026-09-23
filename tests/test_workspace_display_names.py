from msgflux.tools.builtin import ApplyPatchTool, GlobTool, GrepTool, LsTool


def test_workspace_display_names():
    assert GlobTool.display_name == "Find"
    assert LsTool.display_name == "List"
    assert GrepTool.display_name == "Search"
    assert ApplyPatchTool.display_name == "ApplyPatch"
