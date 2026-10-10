"""Tool-owned default usage guidance is opt-in and library-local."""

import msgflux as mf
import msgspec
from msgflux.chat_messages import ChatMessages

from msgflux.nn import ToolLibrary
from msgflux.tools.config import tool_config


def test_defaults_are_dormant_until_enabled_and_apply_to_later_tools():
    def first_lookup(query: str) -> str:
        """Look up the first item."""
        return query

    first_lookup.default_usage_guidance = "Use the first lookup for examples."

    def later_lookup(query: str) -> str:
        """Look up the next item."""
        return query

    later_lookup.default_usage_guidance = "Use the later lookup for updates."

    @tool_config(usage_guidance="Use this one only for approved queries.")
    def explicit_lookup(query: str) -> str:
        """Look up an approved item."""
        return query

    explicit_lookup.default_usage_guidance = "The explicit value must win."

    plain = ToolLibrary(name="plain", tools=[first_lookup])
    opted_in = ToolLibrary(name="opted-in", tools=[first_lookup, explicit_lookup])

    assert plain.get_tool_usage_guidance() == []
    opted_in.apply_default_usage_guidance()
    opted_in.add(later_lookup)

    assert opted_in.get_tool_usage_guidance() == [
        {
            "name": "first_lookup",
            "display_name": "first_lookup",
            "guidance": "Use the first lookup for examples.",
        },
        {
            "name": "explicit_lookup",
            "display_name": "explicit_lookup",
            "guidance": "Use this one only for approved queries.",
        },
        {
            "name": "later_lookup",
            "display_name": "later_lookup",
            "guidance": "Use the later lookup for updates.",
        },
    ]
    assert plain.get_tool_usage_guidance() == []
    assert first_lookup.default_usage_guidance == "Use the first lookup for examples."

    opted_in.remove("later_lookup")
    assert all(
        item["name"] != "later_lookup" for item in opted_in.get_tool_usage_guidance()
    )
    opted_in.add(later_lookup)
    assert any(
        item["name"] == "later_lookup" for item in opted_in.get_tool_usage_guidance()
    )


def test_explicit_guidance_and_empty_override_win_for_functions_and_instances():
    @tool_config(usage_guidance="")
    def silent_lookup(query: str) -> str:
        """Look up without adding guidance."""
        return query

    silent_lookup.default_usage_guidance = "This default must be suppressed."

    class InstanceLookup:
        """Look up using an instance-specific default."""

        name = "instance_lookup"
        default_usage_guidance = "Class default."

        def __call__(self, query: str) -> str:
            return query

    instance = InstanceLookup()
    instance.default_usage_guidance = "Instance default."

    library = ToolLibrary(name="overrides", tools=[silent_lookup, instance])
    library.apply_default_usage_guidance()

    assert library.get_tool_usage_guidance() == [
        {
            "name": "instance_lookup",
            "display_name": "instance_lookup",
            "guidance": "Instance default.",
        }
    ]
    assert silent_lookup.tool_config.usage_guidance == ""
    assert instance.default_usage_guidance == "Instance default."


def test_shared_compiled_definition_does_not_leak_between_libraries():
    def lookup(query: str) -> str:
        """Look up one item."""
        return query

    lookup.default_usage_guidance = "Use this lookup for one item."
    shared_definition = ToolLibrary.inspect_tool_definition(lookup)
    opted_in = ToolLibrary(name="enabled", tools=[shared_definition])
    plain = ToolLibrary(name="disabled", tools=[shared_definition])

    opted_in.apply_default_usage_guidance()

    assert opted_in.get_tool_usage_guidance()[0]["guidance"] == (
        "Use this lookup for one item."
    )
    assert plain.get_tool_usage_guidance() == []
    assert shared_definition.usage_guidance is None
    assert shared_definition.metadata["default_usage_guidance"] == (
        "Use this lookup for one item."
    )


def test_manual_definition_guidance_is_preserved_when_defaults_are_enabled():
    def lookup(query: str) -> str:
        """Look up one item."""
        return query

    lookup.default_usage_guidance = "Tool-owned default."
    compiled = ToolLibrary.inspect_tool_definition(lookup)
    metadata = dict(compiled.metadata)
    metadata.pop("declared_usage_guidance", None)
    manual = msgspec.structs.replace(
        compiled,
        usage_guidance="Explicit definition guidance.",
        metadata=metadata,
    )

    library = ToolLibrary(name="manual", tools=[manual])
    library.apply_default_usage_guidance()

    assert library.get_tool_usage_guidance()[0]["guidance"] == (
        "Explicit definition guidance."
    )


def test_alias_uses_default_attached_to_tool_declaration():
    @mf.tool_config(name_override="lookup_alias")
    def original_lookup(query: str) -> str:
        """Look up an item."""
        return query

    original_lookup.default_usage_guidance = "Use the aliased lookup."
    library = ToolLibrary(name="alias", tools=[original_lookup])
    library.apply_default_usage_guidance()

    assert library.get_tool_usage_guidance() == [
        {
            "name": "lookup_alias",
            "display_name": "lookup_alias",
            "guidance": "Use the aliased lookup.",
        }
    ]


def test_read_default_adds_vision_instructions_only_when_opted_in():
    from msgflux.tools.builtin import ReadFileTool

    text_reader = ReadFileTool()
    image_reader = ReadFileTool(supports_vision=True)
    plain_text = ToolLibrary(name="plain-text", tools=[text_reader])
    plain_image = ToolLibrary(name="plain-image", tools=[image_reader])
    assert plain_text.get_tool_usage_guidance() == []
    assert plain_image.get_tool_usage_guidance() == []

    plain_text.apply_default_usage_guidance()
    plain_image.apply_default_usage_guidance()
    text_guidance = plain_text.get_tool_usage_guidance()[0]["guidance"]
    image_guidance = plain_image.get_tool_usage_guidance()[0]["guidance"]

    assert "reducing limit" in text_guidance
    assert "single line" in text_guidance
    assert "user-role message" not in text_guidance
    assert "reducing limit" in image_guidance
    assert "single line" in image_guidance
    assert "user-role message" in image_guidance


def test_deferred_tool_gets_default_when_activated_after_opt_in():
    @mf.tool_config(defer_loading=True)
    def remote_lookup(query: str) -> str:
        """Look up remote information."""
        return query

    remote_lookup.default_usage_guidance = "Use for remote information."
    library = ToolLibrary(name="deferred", tools=[remote_lookup])
    library.apply_default_usage_guidance()
    thread = ChatMessages(thread_id="deferred-thread")

    library(
        [("search", "tool_search", {"select": ["remote_lookup"]})],
        messages=thread,
    )

    assert thread.get_loaded_tools(library.name) == {"remote_lookup"}
    assert [
        tool.name for tool in library.get_tool_catalog(thread).portable_tools()
    ] == ["remote_lookup"]
    assert library.get_tool_definition("remote_lookup").usage_guidance == (
        "Use for remote information."
    )
