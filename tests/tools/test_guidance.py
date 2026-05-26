from msgflux.core.dotdict import dotdict
from msgflux.tools.guidance import BUILTIN_TOOL_USAGE_GUIDANCE, apply_tool_guidance


def test_apply_tool_guidance_sets_builtin_guidance_on_function():
    def web_search(query: str) -> str:
        return query

    [tool] = apply_tool_guidance([web_search])

    assert tool is web_search
    assert tool.tool_config.usage_guidance == BUILTIN_TOOL_USAGE_GUIDANCE["web_search"]


def test_apply_tool_guidance_preserves_explicit_guidance():
    def web_fetch(url: str) -> str:
        return url

    web_fetch.tool_config = dotdict({"usage_guidance": "Custom guidance."})

    [tool] = apply_tool_guidance([web_fetch])

    assert tool.tool_config.usage_guidance == "Custom guidance."


def test_apply_tool_guidance_uses_tool_name_attribute():
    class SearchTool:
        name = "web_search"

    [tool] = apply_tool_guidance([SearchTool()])

    assert tool.tool_config.usage_guidance == BUILTIN_TOOL_USAGE_GUIDANCE["web_search"]


def test_apply_tool_guidance_accepts_custom_guidance_map():
    def custom_tool() -> str:
        return "ok"

    [tool] = apply_tool_guidance(
        [custom_tool],
        guidance={"custom_tool": "Use for custom work."},
    )

    assert tool.tool_config.usage_guidance == "Use for custom work."


def test_apply_tool_guidance_ignores_unknown_tools():
    def unknown_tool() -> str:
        return "ok"

    [tool] = apply_tool_guidance([unknown_tool])

    assert tool.tool_config == {}
