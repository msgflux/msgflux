from copy import deepcopy
from types import SimpleNamespace

import pytest

from msgflux.chat_messages import ChatMessages
from msgflux.nn import ContextBinding
from msgflux.nn.modules.tool import ToolLibrary
from msgflux.tools.config import tool_config
from msgflux.tools.runtime import ToolIntent
from msgflux.tools.types import ToolBucket


def test_library_compiles_legacy_config_once_into_canonical_definition():
    @tool_config(
        allow_background=True,
        background_capabilities=("activity",),
        runtime_inputs=[
            "message",
            "messages",
            "handle",
            ContextBinding(
                source="vars",
                parameter="tenant",
                options={"key": "tenant"},
            ),
        ],
        defer_loading=True,
    )
    def inspect_inventory(sku: str) -> str:
        """Inspect one inventory item."""
        return sku

    library = ToolLibrary(name="warehouse", tools=[inspect_inventory])
    definition = library.get_tool_definition("inspect_inventory")

    assert definition.dispatch.name == "optional_background"
    assert definition.dispatch.options["capabilities"] == ("activity",)
    assert definition.feedback.name == "model"
    assert [binding.source for binding in definition.context.bindings] == [
        "message",
        "messages",
        "handle",
        "vars",
    ]
    assert definition.context.bindings[-1].parameter == "tenant"
    assert definition.context.bindings[-1].options == {"key": "tenant"}
    assert definition.loading.deferred
    assert definition.kind == "tool"
    assert "run_in_background" in definition.input_schema["properties"]
    assert definition.declaration["allow_background"] is True

    inspect_inventory.tool_config.background = True
    captured = library.captured_executors["inspect_inventory"]
    captured.tool_config.allow_background = False

    assert definition.dispatch.name == "optional_background"
    assert definition.declaration["allow_background"] is True


def test_deferred_loading_uses_compiled_definition_after_config_mutation():
    @tool_config(defer_loading=True)
    def lookup(query: str) -> str:
        """Look up one value."""
        return query

    library = ToolLibrary(name="search", tools=[lookup])
    library.captured_executors["lookup"].tool_config["defer_loading"] = False
    messages = ChatMessages(thread_id="thread_1")

    outcome = library.execute_intents(
        [ToolIntent(id="call_1", name="lookup", arguments={"query": "SKU-1"})],
        messages=messages,
    )[0]

    assert outcome.result == "SKU-1"
    assert messages.get_loaded_tools(library.name) == {"lookup"}


def test_library_has_no_parallel_tool_config_registry():
    def lookup(query: str) -> str:
        """Look up one value."""
        return query

    library = ToolLibrary(name="search", tools=[lookup])

    assert not hasattr(library, "tool_configs")
    assert library.get_tool_definition("lookup").declaration["tool_kind"] == "tool"


def test_library_accepts_a_precompiled_tool_definition():
    def lookup(query: str) -> str:
        """Look up one value."""
        return query

    definition = ToolLibrary.inspect_tool_definition(lookup)
    library = ToolLibrary(name="search", tools=[definition])

    assert library.get_tool_definition("lookup") is definition
    assert library.library["lookup"] is definition.executor


def test_bucket_routing_uses_registered_definition_after_executor_mutation():
    @tool_config(tool_kind="catalog")
    def lookup(query: str) -> str:
        """Look up one catalog value."""
        return query

    class CatalogBucket(ToolBucket):
        """Group catalog tools."""

        name = "catalog"
        capture = {"tool_kind": "catalog", "defer_loading": False}
        annotations = {"return": str}

        def __call__(self) -> str:
            return "catalog"

    library = ToolLibrary(name="search", tools=[lookup])
    library.library["lookup"].tool_config["tool_kind"] = "orders"

    library.add(CatalogBucket())

    assert "lookup" not in library.library
    assert library.bucket_has_tool("catalog", "lookup")
    assert library.get_tool_definition("lookup").kind == "catalog"


def test_presentation_helpers_use_definition_after_executor_mutation():
    @tool_config(
        display_name="Inventory Lookup",
        usage_guidance="Use for inventory questions.",
    )
    def lookup(query: str) -> str:
        """Look up one inventory value."""
        return query

    library = ToolLibrary(name="search", tools=[lookup])
    executor = library.library["lookup"]
    executor.display_name = "Changed"
    executor.usage_guidance = "Changed guidance."

    assert library.get_tool_display_names() == {"lookup": "Inventory Lookup"}
    assert library.get_tool_usage_guidance() == [
        {
            "name": "lookup",
            "display_name": "Inventory Lookup",
            "guidance": "Use for inventory questions.",
        }
    ]


def test_library_registry_indexes_public_and_captured_definitions():
    def calculate(value: int) -> int:
        """Calculate one value."""
        return value * 2

    @tool_config(defer_loading=True)
    def remote_lookup(query: str) -> str:
        """Look up one remote value."""
        return query

    library = ToolLibrary(name="runtime", tools=[calculate, remote_lookup])
    search_bucket = library.library["tool_search"].impl

    assert {definition.name for definition in library.registry.definitions()} == {
        "calculate",
        "remote_lookup",
        "tool_search",
    }
    assert library.registry.get("calculate").executor is library.library["calculate"]
    assert (
        library.registry.get("remote_lookup").executor
        is library.captured_executors["remote_lookup"]
    )
    assert search_bucket.tools["remote_lookup"] == library.get_tool_ref("remote_lookup")

    copied = deepcopy(library)
    copied_search = copied.library["tool_search"].impl

    assert copied.registry.get("calculate").executor is copied.library["calculate"]
    assert (
        copied.registry.get("remote_lookup").executor
        is copied.captured_executors["remote_lookup"]
    )
    assert copied_search.tools["remote_lookup"] == copied.get_tool_ref("remote_lookup")

    library.remove("remote_lookup")

    assert not library.registry.has("remote_lookup")
    assert not library.registry.has("tool_search")


def test_library_catalog_view_is_scoped_to_chat_messages_thread():
    @tool_config(defer_loading=True)
    def lookup(query: str) -> str:
        """Look up one value."""
        return query

    library = ToolLibrary(name="search", tools=[lookup])
    first = ChatMessages(thread_id="thread_a")
    second = ChatMessages(thread_id="thread_b")
    first.load_tools(library.name, ["lookup", "removed_tool"])

    first_view = library.get_tool_catalog_view(first)
    second_view = library.get_tool_catalog_view(second)

    assert first_view.thread_id == "thread_a"
    assert [entry.name for entry in first_view.visible_entries()] == ["lookup"]
    assert [entry.name for entry in second_view.visible_entries()] == ["tool_search"]
    assert [entry.name for entry in second_view.entries] == ["tool_search", "lookup"]


def test_library_catalog_view_requires_configured_thread():
    library = ToolLibrary(name="empty", tools=[])

    with pytest.raises(ValueError, match="configured thread id"):
        library.get_tool_catalog_view(ChatMessages())


def test_library_catalog_view_accepts_explicit_thread_without_chat_messages():
    def status() -> str:
        return "ok"

    library = ToolLibrary(name="runtime", tools=[status])

    view = library.get_tool_catalog_view(thread_id="thread_a")

    assert view.thread_id == "thread_a"


def test_feedback_flags_compile_to_one_feedback_axis():
    expected = {
        "return_direct": "direct",
        "handoff": "handoff",
        "call_as_response": "call_as_response",
    }
    for flag, feedback in expected.items():

        def implementation() -> str:
            """Return a value."""
            return "ok"

        configured = tool_config(**{flag: True})(implementation)
        definition = ToolLibrary.inspect_tool_definition(configured)
        assert definition.feedback.name == feedback


def test_mcp_tools_receive_canonical_definitions_without_losing_executor_type():
    tool_info = SimpleNamespace(
        name="lookup",
        description="Remote lookup.",
        inputSchema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
        },
    )

    class Client:
        async def call_tool(self, _name, _arguments):
            raise AssertionError("This compilation test must not execute the tool")

    remote = ToolLibrary.create_mcp_tool(
        name="lookup",
        mcp_client=Client(),
        mcp_tool_info=tool_info,
        namespace="search",
        config={"defer_loading": True},
    )
    library = ToolLibrary(name="remote", tools=[remote])

    definition = library.get_tool_definition("search__lookup")

    assert definition.executor is remote
    assert definition.input_schema == tool_info.inputSchema
    assert definition.loading.deferred


def test_model_catalog_projects_compiled_definition_without_rebuilding_schema():
    def lookup(query: str) -> str:
        """Look up a value."""
        return query

    library = ToolLibrary(name="search", tools=[lookup])
    tool = library.library["lookup"]

    def unexpected_schema_rebuild():
        raise AssertionError("Catalog generation must use the compiled definition")

    tool.get_json_schema = unexpected_schema_rebuild
    catalog = library.get_tool_catalog()

    assert len(catalog.tools) == 1
    assert catalog.tools[0].name == "lookup"
    assert catalog.tools[0].parameters["properties"]["query"] == {"type": "string"}
