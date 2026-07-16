"""Tests for msgflux.nn.modules.tool module."""

from copy import deepcopy
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import msgflux as mf
import pytest

from msgflux.core.dotdict import dotdict
from msgflux.nn.modules.agent import Agent
from msgflux.nn.modules.tool import (
    ToolCall,
    ToolResponses,
    Tool,
    LocalTool,
    MCPTool,
    ToolLibrary,
    _convert_module_to_nn_tool,
)
from msgflux.runtime.context import execution_context
from msgflux.tasks import InMemoryTaskStore, TaskActivityRecorder
from msgflux.tools import ToolBackground, ToolBucket, ToolLibraryOperator, ToolMetadata
from msgflux.tools.builtin.task_tool import TaskActivityTool, TaskMessageTool
from msgflux.tools.helpers import (
    build_call_parameters_for_response,
    should_copy_injected_messages,
)


def _activity_summaries(store: InMemoryTaskStore, task_id: str) -> list[str]:
    return [activity.summary for activity in store.list_activity(task_id)]


class TestToolCall:
    """Test suite for ToolCall dataclass."""

    def test_tool_call_initialization(self):
        """Test ToolCall basic initialization."""
        tool_call = ToolCall(id="call_123", name="test_tool")
        assert tool_call.id == "call_123"
        assert tool_call.name == "test_tool"
        assert tool_call.parameters == {}
        assert tool_call.result is None
        assert tool_call.error is None

    def test_tool_call_with_parameters(self):
        """Test ToolCall with parameters."""
        params = {"arg1": "value1", "arg2": 42}
        tool_call = ToolCall(id="call_456", name="my_tool", parameters=params)
        assert tool_call.parameters == params

    def test_tool_call_with_result(self):
        """Test ToolCall with result."""
        tool_call = ToolCall(id="call_789", name="calculator", result={"sum": 10})
        assert tool_call.result == {"sum": 10}

    def test_tool_call_with_error(self):
        """Test ToolCall with error."""
        tool_call = ToolCall(id="call_err", name="broken_tool", error="Tool failed")
        assert tool_call.error == "Tool failed"


class TestToolResponses:
    """Test suite for ToolResponses dataclass."""

    def test_tool_responses_initialization(self):
        """Test ToolResponses basic initialization."""
        responses = ToolResponses(return_directly=False)
        assert responses.return_directly is False
        assert responses.tool_calls == []

    def test_tool_responses_with_calls(self):
        """Test ToolResponses with tool calls."""
        call1 = ToolCall(id="call_1", name="tool1", result="result1")
        call2 = ToolCall(id="call_2", name="tool2", result="result2")
        responses = ToolResponses(return_directly=True, tool_calls=[call1, call2])

        assert responses.return_directly is True
        assert len(responses.tool_calls) == 2
        assert responses.tool_calls[0].id == "call_1"
        assert responses.tool_calls[1].id == "call_2"

    def test_tool_responses_to_dict(self):
        """Test ToolResponses to_dict conversion."""
        call = ToolCall(id="call_x", name="toolx", parameters={"key": "val"})
        responses = ToolResponses(return_directly=False, tool_calls=[call])
        result_dict = responses.to_dict()

        assert isinstance(result_dict, dict)
        assert result_dict["return_directly"] is False
        assert len(result_dict["tool_calls"]) == 1
        assert result_dict["tool_calls"][0]["id"] == "call_x"

    def test_tool_responses_to_json(self):
        """Test ToolResponses to_json conversion."""
        responses = ToolResponses(return_directly=True)
        result_json = responses.to_json()

        assert isinstance(result_json, bytes)

    def test_tool_responses_to_dict_accepts_nested_dotdict(self):
        """Test ToolResponses.to_dict handles nested dotdict payloads."""
        call = ToolCall(
            id="call_dotdict",
            name="report",
            result=dotdict(
                {
                    "participants_data": [
                        {"name": "Alice", "company": "OpenAI"},
                        {"name": "Bob", "company": "Msgflux"},
                    ]
                }
            ),
        )
        responses = ToolResponses(return_directly=False, tool_calls=[call])

        result_dict = responses.to_dict()

        assert result_dict["tool_calls"][0]["result"] == {
            "participants_data": [
                {"name": "Alice", "company": "OpenAI"},
                {"name": "Bob", "company": "Msgflux"},
            ]
        }

    def test_tool_responses_get_by_id(self):
        """Test ToolResponses get_by_id method."""
        call1 = ToolCall(id="call_abc", name="tool1")
        call2 = ToolCall(id="call_def", name="tool2")
        responses = ToolResponses(return_directly=False, tool_calls=[call1, call2])

        found = responses.get_by_id("call_abc")
        assert found is not None
        assert found.name == "tool1"

        not_found = responses.get_by_id("call_xyz")
        assert not_found is None

    def test_tool_responses_get_by_name(self):
        """Test ToolResponses get_by_name method."""
        call1 = ToolCall(id="call_1", name="calculator")
        call2 = ToolCall(id="call_2", name="search")
        responses = ToolResponses(return_directly=False, tool_calls=[call1, call2])

        found = responses.get_by_name("search")
        assert found is not None
        assert found.id == "call_2"

        not_found = responses.get_by_name("unknown")
        assert not_found is None


class TestTool:
    """Test suite for Tool base class."""

    def test_tool_inheritance(self):
        """Test that Tool inherits from Module."""
        from msgflux.nn.modules.module import Module

        assert issubclass(Tool, Module)

    def test_tool_get_json_schema(self):
        """Test Tool get_json_schema method."""

        class SimpleTool(Tool):
            """A simple tool for testing."""

            def forward(self, x: int) -> int:
                """Add one to x.

                Args:
                    x: The input number.

                Returns:
                    The input number plus one.
                """
                return x + 1

        tool = SimpleTool()
        schema = tool.get_json_schema()

        assert isinstance(schema, dict)
        assert "type" in schema
        assert schema["type"] == "function"


class TestLocalTool:
    """Test suite for LocalTool."""

    def test_local_tool_initialization(self):
        """Test LocalTool basic initialization."""

        def my_func(x: int) -> int:
            """Test function."""
            return x + 1

        tool = LocalTool(
            name="my_tool",
            description="A test tool",
            annotations={"x": int},
            tool_config={},
            impl=my_func,
        )

        assert tool.name == "my_tool"
        assert tool.description == "A test tool"

    def test_local_tool_uses_python_default_when_null_is_given(self):
        """Test null transport values are omitted when the callable has a default."""

        def my_func(query: str, limit: int = 5) -> int:
            """Test function."""
            return limit

        tool = _convert_module_to_nn_tool(my_func)

        result = tool(query="hello", limit=None)

        assert result == 5

    def test_local_tool_keeps_none_without_python_default(self):
        """Test null values are preserved when the callable requires the param."""

        def my_func(query: Optional[str]) -> Optional[str]:
            """Test function."""
            return query

        tool = _convert_module_to_nn_tool(my_func)

        result = tool(query=None)

        assert result is None
        assert tool.impl == my_func

    def test_local_tool_forward_sync_function(self):
        """Test LocalTool forward with sync function."""

        def add_numbers(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        tool = LocalTool(
            name="add",
            description="Add numbers",
            annotations={"a": int, "b": int},
            tool_config={},
            impl=add_numbers,
        )

        result = tool(a=5, b=3)
        assert result == 8

    def test_local_tool_forward_async_function(self):
        """Test LocalTool forward with async function."""

        async def async_multiply(x: int, y: int) -> int:
            """Multiply two numbers."""
            return x * y

        tool = LocalTool(
            name="multiply",
            description="Multiply numbers",
            annotations={"x": int, "y": int},
            tool_config={},
            impl=async_multiply,
        )

        result = tool(x=4, y=5)
        assert result == 20

    @pytest.mark.asyncio
    async def test_local_tool_aforward_async_function(self):
        """Test LocalTool aforward with async function."""

        async def async_subtract(a: int, b: int) -> int:
            """Subtract b from a."""
            return a - b

        tool = LocalTool(
            name="subtract",
            description="Subtract numbers",
            annotations={"a": int, "b": int},
            tool_config={},
            impl=async_subtract,
        )

        result = await tool.aforward(a=10, b=3)
        assert result == 7

    @pytest.mark.asyncio
    async def test_local_tool_aforward_sync_function(self):
        """Test LocalTool aforward with sync function (runs in executor)."""

        def divide(a: int, b: int) -> float:
            """Divide a by b."""
            return a / b

        tool = LocalTool(
            name="divide",
            description="Divide numbers",
            annotations={"a": int, "b": int},
            tool_config={},
            impl=divide,
        )

        result = await tool.aforward(a=10, b=2)
        assert result == 5.0

    @pytest.mark.asyncio
    async def test_local_tool_aforward_with_acall(self):
        """Test LocalTool aforward with object that has acall method."""

        class CustomCallable:
            async def acall(self, x: int) -> int:
                return x * 2

        obj = CustomCallable()
        tool = LocalTool(
            name="custom",
            description="Custom callable",
            annotations={"x": int},
            tool_config={},
            impl=obj,
        )

        result = await tool.aforward(x=5)
        assert result == 10


class TestConvertModuleToNNTool:
    """Test suite for _convert_module_to_nn_tool function."""

    def test_convert_function_to_tool(self):
        """Test converting a function to Tool."""

        def calculator(a: int, b: int) -> int:
            """Add two numbers together."""
            return a + b

        tool = _convert_module_to_nn_tool(calculator)

        assert isinstance(tool, LocalTool)
        assert tool.name == "calculator"
        assert "Add two numbers" in tool.description
        assert "a" in tool.annotations
        assert "b" in tool.annotations

    def test_convert_async_function_to_tool(self):
        """Test converting an async function to Tool."""

        async def async_calc(x: int) -> int:
            """Double the number."""
            return x * 2

        tool = _convert_module_to_nn_tool(async_calc)

        assert isinstance(tool, LocalTool)
        assert tool.name == "async_calc"

    def test_convert_class_to_tool(self):
        """Test converting a class to Tool."""

        class MyTool:
            """A custom tool."""

            def __call__(self, value: str) -> str:
                """Process a value."""
                return value.upper()

        tool = _convert_module_to_nn_tool(MyTool)

        assert isinstance(tool, LocalTool)
        assert tool.name == "MyTool"

    def test_convert_class_instance_to_tool(self):
        """Test converting a class instance to Tool."""

        class Counter:
            """A counter tool."""

            name = "counter"

            def __init__(self):
                self.count = 0

            def __call__(self, increment: int) -> int:
                """Increment the counter."""
                self.count += increment
                return self.count

        instance = Counter()
        tool = _convert_module_to_nn_tool(instance)

        assert isinstance(tool, LocalTool)
        assert tool.name == "counter"

    def test_convert_with_name_override(self):
        """Test converting with name override."""

        def my_func(x: int) -> int:
            """Test function."""
            return x

        my_func.tool_config = {"name_overridden": "custom_name"}
        tool = _convert_module_to_nn_tool(my_func)

        assert tool.name == "custom_name"

    def test_convert_with_handoff_config(self):
        """Test converting with handoff configuration."""

        def transfer_tool() -> None:
            """Transfer to another agent."""
            pass

        transfer_tool.tool_config = {"handoff": True}
        tool = _convert_module_to_nn_tool(transfer_tool)

        assert tool.name.startswith("transfer_to_")
        assert tool.annotations == {}

    def test_convert_with_disable_input_config(self):
        """Test converting with disable_input configuration."""

        def background_tool(task: str) -> str:
            """Run with runtime-only context."""
            return task

        background_tool.tool_config = {"disable_input": True}
        tool = _convert_module_to_nn_tool(background_tool)

        assert tool.name == "background_tool"
        assert tool.annotations == {}

    def test_convert_with_spawn_config(self):
        """Test converting with spawn configuration."""

        def dispatched_task(data: str) -> None:
            """Dispatch task without return."""
            pass

        dispatched_task.tool_config = {"spawn": True}
        tool = _convert_module_to_nn_tool(dispatched_task)

        assert "not generate a return" in tool.description.lower()

    def test_convert_function_with_no_params(self):
        """Test converting function with no parameters."""

        def no_params() -> int:
            """Return a constant."""
            return 42

        tool = _convert_module_to_nn_tool(no_params)
        assert isinstance(tool, LocalTool)
        assert "return" in tool.annotations

    def test_convert_class_missing_docstring(self):
        """Test that class missing docstring raises error."""

        class NoDoc:
            def __call__(self, x: int):
                return x

        with pytest.raises(NotImplementedError, match="docstring"):
            _convert_module_to_nn_tool(NoDoc)

    def test_convert_class_not_callable(self):
        """Test that class without __call__ raises error."""

        class NotCallable:
            """Has doc but not callable."""

            pass

        # This will raise AttributeError when trying to access __call__
        with pytest.raises(AttributeError):
            _convert_module_to_nn_tool(NotCallable)

    def test_convert_class_missing_annotations(self):
        """Test that class with __call__ but missing annotations raises error."""

        class NoAnnotations:
            """Has doc and __call__ but no annotations."""

            def __call__(self, x):
                """Does something."""
                return x

        # This should succeed - annotations are optional if there are no params
        tool = _convert_module_to_nn_tool(NoAnnotations)
        assert tool is not None


class TestToolLibrary:
    """Test suite for ToolLibrary."""

    def test_tool_library_initialization(self):
        """Test ToolLibrary basic initialization."""

        def tool1(x: int) -> int:
            """Tool 1."""
            return x

        def tool2(y: str) -> str:
            """Tool 2."""
            return y

        library = ToolLibrary(name="my_lib", tools=[tool1, tool2])

        assert library.name == "my_lib_tool_library"
        assert "tool1" in library.library
        assert "tool2" in library.library

    def test_tool_library_runtime_helpers_are_lazy(self):
        """Test runtime helper objects are created only when needed."""

        def tool1(x: int) -> int:
            """Tool 1."""
            return x

        library = ToolLibrary(name="my_lib", tools=[tool1])

        assert library._handle is None
        assert library._background_dispatcher is None
        assert library._task_store is None
        assert library._agent_inbox is None

        library.get_tool_json_schemas()

        assert library._handle is None
        assert library._background_dispatcher is None

    def test_tool_library_add_tool(self):
        """Test adding a tool to library."""

        def new_tool(z: float) -> float:
            """New tool."""
            return z * 2

        library = ToolLibrary(name="lib", tools=[])
        library.add(new_tool)

        assert "new_tool" in library.library

    def test_build_call_parameters_for_response_omits_runtime_values(self):
        parameters = build_call_parameters_for_response(
            {
                "query": "hello",
                "vars": {"tenant": "acme"},
                "messages": [{"role": "user", "content": "hello"}],
                "handle": object(),
                "tool_call_id": "call_1",
                "run_in_background": True,
            }
        )

        assert parameters == {"query": "hello"}
        assert build_call_parameters_for_response(None) is None

    def test_tool_library_skips_background_validation_for_regular_metadata(self):
        def regular_tool() -> str:
            return "ok"

        metadata = ToolMetadata(
            name="regular_tool",
            description="A regular tool.",
            annotations={"return": str},
            tool_config={"tool_kind": "tool"},
            impl=regular_tool,
        )

        with patch.object(
            ToolBackground, "validate_background_capabilities"
        ) as validate:
            library = ToolLibrary(name="test_library", tools=[metadata])

        assert "regular_tool" in library.library
        validate.assert_not_called()

    def test_tool_library_defaults_all_registered_tools_to_not_on_demand(self):
        def plain_tool() -> str:
            """A plain callable tool."""
            return "ok"

        metadata = ToolMetadata(
            name="metadata_tool",
            description="A regular tool.",
            annotations={"return": str},
            tool_config={},
            impl=lambda: "ok",
        )
        library = ToolLibrary(name="lib", tools=[plain_tool, metadata])

        assert metadata.tool_config["on_demand"] is False
        assert library.tool_configs["plain_tool"]["on_demand"] is False
        assert library.tool_configs["metadata_tool"]["on_demand"] is False

    def test_tool_library_add_duplicate_raises_error(self):
        """Test that adding duplicate tool raises error."""

        def my_tool(x: int) -> int:
            """My tool."""
            return x

        library = ToolLibrary(name="lib", tools=[my_tool])

        with pytest.raises(ValueError, match="already in tool library"):
            library.add(my_tool)

    def test_background_task_tool_conflict_raises_error(self):
        """Test that background task tools cannot overwrite user tools."""

        def task_status(task_id: str) -> str:
            """User-defined task status."""
            return task_id

        @mf.tool_config(background=True)
        def background_tool() -> str:
            """Run in the background."""
            return "ok"

        with pytest.raises(
            ValueError,
            match="background task tool `task_status` conflicts",
        ):
            ToolLibrary(name="lib", tools=[task_status, background_tool])

    def test_tool_library_add_already_tool_instance(self):
        """Test adding Tool instance directly."""

        def my_func(x: int) -> int:
            """Test."""
            return x

        tool = _convert_module_to_nn_tool(my_func)
        library = ToolLibrary(name="lib", tools=[])
        library.add(tool)

        assert "my_func" in library.library

    def test_tool_library_rejects_non_mapping_tool_params(self):
        """Test that tool call params must be mappings."""

        def my_tool(x: int) -> int:
            """My tool."""
            return x

        library = ToolLibrary(name="lib", tools=[my_tool])

        with pytest.raises(TypeError, match="parameters must be a mapping"):
            library([("call_1", "my_tool", "not-a-mapping")])

    def test_tool_library_remove_tool(self):
        """Test removing a tool from library."""

        def tool_to_remove(x: int) -> int:
            """Tool."""
            return x

        library = ToolLibrary(name="lib", tools=[tool_to_remove])
        library.remove("tool_to_remove")

        assert "tool_to_remove" not in library.library

    def test_tool_library_with_config(self):
        """Test ToolLibrary stores tool configs."""

        def my_tool(x: int) -> int:
            """Tool."""
            return x

        my_tool.tool_config = {"return_direct": True}
        library = ToolLibrary(name="lib", tools=[my_tool])

        assert "my_tool" in library.tool_configs
        assert library.tool_configs["my_tool"]["return_direct"] is True

    def test_tool_library_remove_nonexistent_raises_error(self):
        """Test that removing non-existent tool raises error."""
        library = ToolLibrary(name="lib", tools=[])

        with pytest.raises(ValueError, match="not in tool library"):
            library.remove("nonexistent")

    def test_tool_library_clear(self):
        """Test clearing library."""

        def tool1(x: int) -> int:
            """Tool 1."""
            return x

        def tool2(y: int) -> int:
            """Tool 2."""
            return y

        library = ToolLibrary(name="lib", tools=[tool1, tool2])
        assert len(library.library) == 2

        library.clear()

        assert len(library.library) == 0

    def test_tool_library_get_tools(self):
        """Test getting all tools."""

        def tool1(x: int) -> int:
            """Tool 1."""
            return x

        def tool2(y: int) -> int:
            """Tool 2."""
            return y

        library = ToolLibrary(name="lib", tools=[tool1, tool2])
        tools = list(library.get_tools())

        assert len(tools) == 2

    def test_tool_library_get_tool_names(self):
        """Test getting tool names."""

        def tool1(x: int) -> int:
            """Tool 1."""
            return x

        def tool2(y: int) -> int:
            """Tool 2."""
            return y

        library = ToolLibrary(name="lib", tools=[tool1, tool2])
        names = library.get_tool_names()

        assert "tool1" in names
        assert "tool2" in names

    def test_exposed_false_hides_schema_without_unregistering_tool(self):
        def internal_lookup(query: str) -> str:
            """Run an internal lookup."""
            return f"internal:{query}"

        metadata = ToolMetadata(
            name="internal_lookup",
            description="Run an internal lookup.",
            annotations={"query": str, "return": str},
            tool_config={"exposed": False},
            impl=internal_lookup,
        )
        library = ToolLibrary(name="lib", tools=[metadata])

        assert "internal_lookup" in library.library
        assert library.get_tool_names() == []
        assert library.get_tool_json_schemas() == []
        assert library.get_tool_annotations() == {}
        assert library.get_tool_display_names() == {}
        assert library.execute("internal_lookup", {"query": "cache"}) == (
            "internal:cache"
        )

    def test_tools_default_to_exposed_when_configuration_omits_it(self):
        def lookup(query: str) -> str:
            """Run a lookup."""
            return query

        library = ToolLibrary(name="lib", tools=[lookup])

        assert library.tool_configs["lookup"]["exposed"] is True
        assert library.get_tool_names() == ["lookup"]

    def test_exposed_must_be_boolean(self):
        def invalid() -> None:
            """Declare invalid exposure."""

        metadata = ToolMetadata(
            name="invalid",
            description="Declare invalid exposure.",
            annotations={"return": None},
            tool_config={"exposed": "no"},
            impl=invalid,
        )
        with pytest.raises(TypeError, match="`exposed` must be a bool"):
            ToolLibrary(name="lib", tools=[metadata])

    def test_tool_library_get_tool_display_names(self):
        """Test getting human-readable tool display names."""

        def plain_tool(x: int) -> int:
            """Plain tool."""
            return x

        def configured_tool(x: int) -> int:
            """Configured tool."""
            return x

        configured_tool.tool_config = dotdict({"display_name": "Configured Tool"})
        library = ToolLibrary(name="lib", tools=[plain_tool, configured_tool])

        display_names = library.get_tool_display_names()

        assert display_names["plain_tool"] == "plain_tool"
        assert display_names["configured_tool"] == "Configured Tool"

    def test_tool_library_get_tool_usage_guidance(self):
        """Test getting tool usage guidance metadata."""

        def search_orders(order_id: str) -> str:
            """Search orders."""
            return order_id

        search_orders.tool_config = dotdict(
            {
                "display_name": "Order Search",
                "usage_guidance": "Use when the user asks about an order.",
            }
        )
        library = ToolLibrary(name="lib", tools=[search_orders])

        guidance = library.get_tool_usage_guidance()

        assert guidance == [
            {
                "name": "search_orders",
                "display_name": "Order Search",
                "guidance": "Use when the user asks about an order.",
            }
        ]

    def test_tool_library_filters_tool_usage_guidance(self):
        """Test usage guidance can be filtered by exposed tool names."""

        def search_orders(order_id: str) -> str:
            """Search orders."""
            return order_id

        def cancel_order(order_id: str) -> str:
            """Cancel orders."""
            return order_id

        search_orders.tool_config = dotdict(
            {"usage_guidance": "Use for order status questions."}
        )
        cancel_order.tool_config = dotdict(
            {"usage_guidance": "Use for order cancellation requests."}
        )
        library = ToolLibrary(name="lib", tools=[search_orders, cancel_order])

        guidance = library.get_tool_usage_guidance(tool_names={"search_orders"})

        assert guidance == [
            {
                "name": "search_orders",
                "display_name": "search_orders",
                "guidance": "Use for order status questions.",
            }
        ]

    def test_tool_library_get_tool_json_schemas(self):
        """Test getting tool JSON schemas."""

        def calculator(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        library = ToolLibrary(name="lib", tools=[calculator])
        schemas = library.get_tool_json_schemas()

        assert len(schemas) == 1
        assert isinstance(schemas[0], dict)

    def test_tool_library_hides_on_demand_tools_from_schemas(self):
        """Test that on-demand tools are hidden until loaded."""

        @mf.tool_config(on_demand=True)
        def remote_lookup(query: str) -> str:
            """Look up external information."""
            return query

        library = ToolLibrary(name="lib", tools=[remote_lookup])

        names = library.get_tool_names()
        schemas = library.get_tool_json_schemas()

        assert "remote_lookup" in names
        assert "search_tools" in names
        assert [schema["function"]["name"] for schema in schemas] == ["search_tools"]
        parameters = schemas[0]["function"]["parameters"]
        properties = parameters["properties"]
        assert properties == {"query": {"type": "string"}}
        assert parameters["required"] == ["query"]
        assert isinstance(library.library["search_tools"].impl, ToolLibraryOperator)
        assert isinstance(library.library["search_tools"].impl, ToolBucket)
        assert library.library["search_tools"].tool_config["handle"] == {
            "tools": ["activate"]
        }
        assert library.library["search_tools"].tool_config["tool_kind"] == "bucket"

    def test_search_tools_is_not_captured_by_search_tools_bucket(self):
        """Test search_tools stays registered even if a search bucket is added."""

        @mf.tool_config(on_demand=True)
        def remote_lookup(query: str) -> str:
            """Look up external information."""
            return query

        class SearchBucket(ToolBucket):
            name = "search_bucket"
            capture = {"tool_kind": "bucket", "on_demand": False}
            description = "Capture search tools."
            annotations = {"query": str, "return": str}

            def __call__(self, query: str) -> str:
                return query

        library = ToolLibrary(name="lib", tools=[remote_lookup])
        library.add(SearchBucket())

        assert "search_tools" in library.library
        assert library.library["search_bucket"].impl.tools == {}

    def test_tool_bucket_captures_multiple_tool_kinds(self):
        @mf.tool_config(tool_kind="catalog")
        def find_product(query: str) -> str:
            """Find a product."""
            return query

        @mf.tool_config(tool_kind="catalog")
        def list_products() -> str:
            """List products."""
            return "products"

        @mf.tool_config(tool_kind="orders")
        def get_order(order_id: str) -> str:
            """Get an order."""
            return order_id

        class CommerceBucket(ToolBucket):
            """Group commerce tools."""

            name = "commerce"
            capture = {"tool_kind": "catalog|orders", "on_demand": False}
            annotations = {"return": str}

            def __call__(self) -> str:
                return "commerce"

        bucket = CommerceBucket()
        library = ToolLibrary(
            name="lib",
            tools=[find_product, list_products, get_order, bucket],
        )

        assert set(library.library) == {
            "find_product",
            "list_products",
            "get_order",
            "commerce",
        }
        assert library.get_tool_names() == ["commerce"]
        assert library.tool_configs["find_product"]["exposed"] is False
        assert set(bucket.tools) == {"find_product", "list_products", "get_order"}
        assert library.tool_configs["commerce"]["tool_kind"] == "bucket"
        assert bucket.tools["find_product"].tool_config["tool_kind"] == "catalog"

        with pytest.raises(ValueError, match="Duplicate tool name `find_product`"):
            library.add(find_product)

    def test_tool_bucket_captures_background_tool_kinds(self):
        @mf.tool_config(tool_kind="background")
        def background_job() -> str:
            """Run a background job."""
            return "background"

        @mf.tool_config(tool_kind="allow_background")
        def optional_background_job() -> str:
            """Run an optional background job."""
            return "optional"

        class BackgroundBucket(ToolBucket):
            """Group background-capable tools."""

            name = "background_jobs"
            capture = {
                "tool_kind": "background|allow_background",
                "on_demand": False,
            }
            annotations = {"return": str}

            def __call__(self) -> str:
                return "background"

        bucket = BackgroundBucket()
        library = ToolLibrary(
            name="lib",
            tools=[background_job, optional_background_job, bucket],
        )

        assert set(library.library) == {
            "background_job",
            "optional_background_job",
            "background_jobs",
        }
        assert library.get_tool_names() == ["background_jobs"]
        assert set(bucket.tools) == {"background_job", "optional_background_job"}

    def test_tool_bucket_rejects_overlapping_capture_rules(self):
        class FirstBucket(ToolBucket):
            """Capture catalog tools."""

            name = "first"
            capture = {"tool_kind": "catalog|orders", "on_demand": False}

            def __call__(self) -> str:
                return "first"

        class SecondBucket(ToolBucket):
            """Capture order tools."""

            name = "second"
            capture = {"tool_kind": "orders|billing", "on_demand": False}

            def __call__(self) -> str:
                return "second"

        with pytest.raises(ValueError, match=r"capture.*overlaps"):
            ToolLibrary(name="lib", tools=[FirstBucket(), SecondBucket()])

    def test_tool_bucket_rejects_empty_capture_tool_kind_segment(self):
        class InvalidBucket(ToolBucket):
            """Invalid bucket."""

            name = "invalid"
            capture = {"tool_kind": "catalog||orders"}

            def __call__(self) -> str:
                return "invalid"

        with pytest.raises(ValueError, match="cannot be empty"):
            ToolLibrary(name="lib", tools=[InvalidBucket()])

    def test_tool_bucket_capture_policy_restricts_declared_handle_access(self):
        class RestrictedBucket(ToolBucket):
            """Capture tools with restricted handles."""

            name = "restricted"
            capture = {
                "tool_kind": "operation",
                "policy": {"handle": {"tools": ["list"]}},
            }

            def __call__(self) -> str:
                return "restricted"

        @mf.tool_config(tool_kind="operation", handle={"tools": ["list"]})
        def allowed(handle: mf.Hidden) -> str:
            """List tools."""
            return ",".join(handle.tools.list())

        @mf.tool_config(tool_kind="operation", handle={"tools": ["remove"]})
        def denied(handle: mf.Hidden) -> str:
            """Remove tools."""
            return handle.tools.remove("allowed")

        library = ToolLibrary(name="lib", tools=[RestrictedBucket(), allowed])

        assert "allowed" in library.library["restricted"].impl.tools
        with pytest.raises(ValueError, match=r"tools\.remove.*capture policy"):
            library.add(denied)
        assert "denied" not in library.get_tool_names()

    def test_tool_bucket_without_capture_policy_trusts_declared_handle_access(self):
        class TrustedBucket(ToolBucket):
            """Capture trusted tools."""

            name = "trusted"
            capture = {"tool_kind": "operation"}

            def __call__(self) -> str:
                return "trusted"

        @mf.tool_config(tool_kind="operation", handle={"tools": ["remove"]})
        def trusted_operation(handle: mf.Hidden) -> str:
            """Use trusted handle access."""
            return "ok"

        library = ToolLibrary(
            name="lib",
            tools=[TrustedBucket(), trusted_operation],
        )

        assert "policy" not in library.library["trusted"].impl.capture
        assert "trusted_operation" in library.library["trusted"].impl.tools

    def test_tool_bucket_rejects_unknown_capture_policy(self):
        class InvalidPolicyBucket(ToolBucket):
            """Declare an unknown capture policy."""

            name = "invalid_policy"
            capture = {
                "tool_kind": "operation",
                "policy": {"unknown": {}},
            }

            def __call__(self) -> str:
                return "invalid"

        with pytest.raises(ValueError, match="Supported policies: handle"):
            ToolLibrary(name="lib", tools=[InvalidPolicyBucket()])

    def test_tool_bucket_captures_generic_configuration(self):
        class PreviewBucket(ToolBucket):
            """Group preview tools."""

            name = "preview"
            capture = {"preview": True}
            annotations = {"return": str}

            def __call__(self) -> str:
                return "preview"

        def render_preview() -> str:
            """Render a preview."""
            return "preview"

        metadata = ToolMetadata(
            name="render_preview",
            description="Render a preview.",
            annotations={"return": str},
            tool_config={"tool_kind": "tool", "preview": True},
            impl=render_preview,
        )
        bucket = PreviewBucket()
        library = ToolLibrary(name="lib", tools=[metadata, bucket])

        assert set(library.library) == {"render_preview", "preview"}
        assert library.get_tool_names() == ["preview"]
        assert set(bucket.tools) == {"render_preview"}

    def test_tool_bucket_captures_by_name_and_capabilities(self):
        @mf.tool_config(capabilities=["python_callable", "filesystem_read"])
        def inspect_file(path: str) -> str:
            """Inspect a file."""
            return path

        @mf.tool_config(capabilities=["filesystem_read"])
        def read_file(path: str) -> str:
            """Read a file."""
            return path

        class InterpreterBucket(ToolBucket):
            """Expose tools available to an interpreter."""

            name = "interpreter"
            capture = {
                "source": "tool",
                "match": {
                    "any": [
                        {"name": ["read_file"]},
                        {"capabilities": {"all": ["python_callable"]}},
                    ]
                },
            }
            annotations = {"return": str}

            def __call__(self) -> str:
                return "ready"

        bucket = InterpreterBucket()
        library = ToolLibrary(
            name="lib",
            tools=[bucket, inspect_file, read_file],
        )

        assert set(bucket.tools) == {"inspect_file", "read_file"}
        assert bucket.tools["inspect_file"].tool_config["capabilities"] == (
            "python_callable",
            "filesystem_read",
        )
        assert set(library.library) == {"interpreter", "inspect_file", "read_file"}
        assert library.get_tool_names() == ["interpreter"]

    def test_tool_bucket_rejects_capture_cycle(self):
        class FirstBucket(ToolBucket):
            """Capture the second bucket."""

            name = "first"
            capture = {"source": "bucket", "name": "second"}

            def __call__(self) -> str:
                return "first"

        class SecondBucket(ToolBucket):
            """Capture the first bucket."""

            name = "second"
            capture = {"source": "bucket", "name": "first"}

            def __call__(self) -> str:
                return "second"

        library = ToolLibrary(name="lib", tools=[FirstBucket()])

        with pytest.raises(ValueError, match="capture cycle"):
            library.add(SecondBucket())

        assert list(library.library) == ["first"]

    def test_nested_bucket_parent_refresh_failure_restores_child_roots(self):
        def leaf() -> str:
            """Return a leaf value."""
            return "leaf"

        class ChildBucket(ToolBucket):
            """Capture the leaf tool."""

            name = "child"
            capture = {"name": "leaf"}

            def __call__(self) -> str:
                return "child"

        class RejectingParent(ToolBucket):
            """Reject a populated child during refresh."""

            name = "parent"
            capture = {"source": "bucket", "name": "child"}

            def refresh(self):
                if self.tools:
                    raise ValueError("parent rejected child")

            def __call__(self) -> str:
                return "parent"

        library = ToolLibrary(name="lib", tools=[RejectingParent(), leaf])

        with pytest.raises(ValueError, match="parent rejected child"):
            library.add(ChildBucket())

        assert list(library.library) == ["parent", "leaf"]
        assert library.library["parent"].impl.tools == {}

    def test_nested_bucket_ancestor_refresh_failure_rolls_back_late_child(self):
        def late_leaf() -> str:
            """Return a late leaf value."""
            return "late"

        class InnerBucket(ToolBucket):
            """Capture and describe the late leaf."""

            name = "inner"
            capture = {"name": "late_leaf"}
            description = "Inner tools: none."

            def refresh(self):
                names = ", ".join(self.tools) or "none"
                self.description = f"Inner tools: {names}."

            def __call__(self) -> str:
                return "inner"

        class OuterBucket(ToolBucket):
            """Reject an inner presentation containing the late leaf."""

            name = "outer"
            capture = {"source": "bucket", "name": "inner"}

            def refresh(self):
                inner = self.tools.get("inner")
                if inner is not None and "late_leaf" in inner.description:
                    raise ValueError("outer rejected late leaf")

            def __call__(self) -> str:
                return "outer"

        inner = InnerBucket()
        library = ToolLibrary(name="lib", tools=[OuterBucket(), inner])

        with pytest.raises(ValueError, match="outer rejected late leaf"):
            library.add(late_leaf)

        assert inner.tools == {}
        assert set(library.library) == {"outer", "inner"}
        assert library.tool_configs["inner"]["exposed"] is False
        assert library.get_tool_names() == ["outer"]
        inner_description = library.library["outer"].impl.tools["inner"].description
        assert "late_leaf" not in inner_description

    def test_background_reconciliation_restores_removed_captured_task_tools(self):
        class OptionalControls(ToolBucket):
            """Capture optional background controls."""

            name = "optional_controls"
            capture = {
                "tool_kind": "background_activity|background_message",
            }

            def refresh(self):
                if not self.tools:
                    raise ValueError("optional controls cannot become empty")

            def __call__(self) -> str:
                return "controls"

        @mf.tool_config(background=True)
        def background_job() -> str:
            """Run a background job."""
            return "done"

        bucket = OptionalControls()
        library = ToolLibrary(
            name="lib",
            tools=[TaskActivityTool(), TaskMessageTool(), bucket],
        )
        activity_wrapper = library.library["task_activity"]
        message_wrapper = library.library["task_message"]

        with pytest.raises(ValueError, match="cannot become empty"):
            library.add(background_job)

        assert list(library.library) == [
            "task_activity",
            "task_message",
            "optional_controls",
        ]
        assert library.tool_owners == {
            "task_activity": "optional_controls",
            "task_message": "optional_controls",
        }
        assert library.library["task_activity"] is activity_wrapper
        assert library.library["task_message"] is message_wrapper
        assert library.tool_configs["task_activity"]["exposed"] is False
        assert library.tool_configs["task_message"]["exposed"] is False
        assert library.get_tool_names() == ["optional_controls"]
        assert set(bucket.tools) == {"task_activity", "task_message"}

    def test_bucket_metadata_items_load_one_snapshot(self):
        def leaf() -> str:
            """Return a leaf."""
            return "leaf"

        class Bucket(ToolBucket):
            """Capture the leaf."""

            name = "bucket"
            capture = {"name": "leaf"}

            def __call__(self) -> str:
                return "bucket"

        bucket = Bucket()
        library = ToolLibrary(name="lib", tools=[bucket, leaf])
        loader = Mock(wraps=bucket._tools_view._loader)
        bucket._tools_view._loader = loader

        assert [name for name, _ in bucket.tools.items()] == ["leaf"]
        loader.assert_called_once_with()

    def test_deepcopied_bucket_uses_copied_registry_view(self):
        @mf.tool_config(tool_kind="leaf")
        def first() -> str:
            """Return the first leaf."""
            return "first"

        @mf.tool_config(tool_kind="leaf")
        def second() -> str:
            """Return the second leaf."""
            return "second"

        class Bucket(ToolBucket):
            """Capture leaf tools."""

            name = "bucket"
            capture = {"tool_kind": "leaf"}

            def __call__(self) -> str:
                return "bucket"

        original = ToolLibrary(name="original", tools=[Bucket(), first])
        copied = deepcopy(original)

        copied.add(second)

        assert set(original.library["bucket"].impl.tools) == {"first"}
        assert set(copied.library["bucket"].impl.tools) == {"first", "second"}

    def test_nested_staged_bucket_rollback_restores_complete_subtree(self):
        def leaf() -> str:
            """Return a staged leaf."""
            return "leaf"

        class InnerBucket(ToolBucket):
            """Capture the staged leaf."""

            name = "inner"
            capture = {"name": "leaf"}
            annotations = {"return": str}

            def __call__(self) -> str:
                return "inner"

        class MiddleBucket(ToolBucket):
            """Capture the staged inner bucket."""

            name = "middle"
            capture = {"source": "bucket", "name": "inner"}
            annotations = {"return": str}

            def __call__(self) -> str:
                return "middle"

        class RejectingParent(ToolBucket):
            """Reject the populated middle bucket."""

            name = "parent"
            capture = {"source": "bucket", "name": "middle"}
            annotations = {"return": str}

            def refresh(self):
                if self.tools:
                    raise ValueError("parent rejected middle")

            def __call__(self) -> str:
                return "parent"

        inner = InnerBucket()
        inner.add(
            ToolMetadata(
                name="leaf",
                description="Return a staged leaf.",
                annotations={"return": str},
                tool_config={},
                impl=leaf,
            )
        )
        middle = MiddleBucket()
        middle.add(
            ToolMetadata(
                name="inner",
                description="Capture the staged leaf.",
                annotations={"return": str},
                tool_config={"tool_kind": "bucket"},
                impl=inner,
            )
        )
        library = ToolLibrary(name="lib", tools=[RejectingParent()])

        with pytest.raises(ValueError, match="parent rejected middle"):
            library.add(middle)

        assert list(library.library) == ["parent"]
        assert library.tool_owners == {}
        assert list(middle.tools) == ["inner"]
        assert list(inner.tools) == ["leaf"]

    def test_tool_library_operator_injects_handle_by_default(self):
        """Test operator tools inherit handle injection."""

        class RuntimeEchoTool(ToolLibraryOperator):
            name = "runtime_echo"
            tool_kind = "diagnostic"
            description = "List the current tool names."
            annotations = {"handle": mf.Hidden, "return": str}

            def __call__(self, handle):
                return ",".join(handle.tools.list())

        library = ToolLibrary(name="lib", tools=[RuntimeEchoTool()])
        schema = next(
            item
            for item in library.get_tool_json_schemas()
            if item["function"]["name"] == "runtime_echo"
        )
        result = library([("call_1", "runtime_echo", {})])

        assert "handle" not in schema["function"]["parameters"].get("properties", {})
        assert result.tool_calls[0].result == "runtime_echo"
        assert library.library["runtime_echo"].tool_config["handle"] == {
            "tools": ["list"]
        }
        assert library.library["runtime_echo"].tool_config["tool_kind"] == "diagnostic"

    def test_public_tools_parameter_is_preserved_in_response(self):
        def echo_tools(tools: str) -> str:
            """Echo a public tools value."""
            return tools

        library = ToolLibrary(name="lib", tools=[echo_tools])

        result = library([("call_1", "echo_tools", {"tools": "public"})])

        assert result.tool_calls[0].parameters == {"tools": "public"}
        assert result.tool_calls[0].result == "public"

    def test_search_tools_captures_on_demand_operator_tools(self):
        """Test on-demand operators use the same ToolSearch bucket."""

        class DeferredOperator(ToolLibraryOperator):
            name = "deferred_operator"
            description = "List the currently registered tools."
            annotations = {"handle": mf.Hidden, "return": list[str]}
            tool_config = {
                "handle": {"tools": ["list"]},
                "on_demand": True,
            }

            def __call__(self, handle) -> list[str]:
                return handle.tools.list()

        library = ToolLibrary(name="lib", tools=[DeferredOperator()])

        assert [
            schema["function"]["name"] for schema in library.get_tool_json_schemas()
        ] == ["search_tools"]

        library([("call_1", "search_tools", {"query": "deferred_operator"})])
        response = library([("call_2", "deferred_operator", {})])

        assert "deferred_operator" in response.tool_calls[0].result

    def test_search_tools_does_not_duplicate_schema_guidance(self):
        """Tool search syntax lives in its schema description only."""

        @mf.tool_config(on_demand=True)
        def remote_lookup(query: str) -> str:
            """Look up external information."""
            return query

        library = ToolLibrary(name="lib", tools=[remote_lookup])

        guidance = library.get_tool_usage_guidance()

        assert guidance == []

    def test_search_tools_returns_matching_on_demand_tools_without_loading(self):
        """Test that keyword search describes matches without exposing them."""

        @mf.tool_config(on_demand=True)
        def remote_lookup(query: str) -> str:
            """Look up external information."""
            return query

        library = ToolLibrary(name="lib", tools=[remote_lookup])

        result = (
            library(
                [
                    (
                        "call_1",
                        "search_tools",
                        {"query": "remote lookup"},
                    )
                ]
            )
            .tool_calls[0]
            .result
        )
        schemas = library.get_tool_json_schemas()
        schema_names = [schema["function"]["name"] for schema in schemas]

        assert result == "remote_lookup: Look up external information."
        assert "search_tools" in schema_names
        assert "remote_lookup" not in schema_names

    def test_search_tools_exact_name_loads_and_regex_limit_searches(self):
        """Exact names load; regex and :K keep search compact."""

        @mf.tool_config(on_demand=True)
        def read_cloud_file(path: str) -> str:
            """Read a cloud file."""
            return path

        @mf.tool_config(on_demand=True)
        def read_legacy_file(path: str) -> str:
            """Read a legacy file."""
            return path

        library = ToolLibrary(
            name="lib",
            tools=[read_cloud_file, read_legacy_file],
        )

        regex_result = (
            library([("call_1", "search_tools", {"query": "/read_.*_file/:1"})])
            .tool_calls[0]
            .result
        )
        loaded_result = (
            library([("call_2", "search_tools", {"query": "read_cloud_file"})])
            .tool_calls[0]
            .result
        )

        assert regex_result == "read_cloud_file: Read a cloud file."
        assert loaded_result == "loaded=read_cloud_file"

    @pytest.mark.asyncio
    async def test_search_tools_loads_two_tools_in_one_turn(self):
        def echo(value: str) -> str:
            """Echo a value."""
            return value

        @mf.tool_config(on_demand=True)
        def first_lookup(query: str) -> str:
            """Run the first lookup."""
            return query

        @mf.tool_config(on_demand=True)
        def second_lookup(query: str) -> str:
            """Run the second lookup."""
            return query

        calls = [
            ("call_0", "echo", {"value": "ready"}),
            ("call_1", "search_tools", {"query": "first_lookup"}),
            ("call_2", "search_tools", {"query": "second_lookup"}),
        ]

        sync_library = ToolLibrary(
            name="sync",
            tools=[echo, first_lookup, second_lookup],
        )
        async_library = ToolLibrary(
            name="async",
            tools=[echo, first_lookup, second_lookup],
        )

        sync_response = sync_library(calls)
        async_response = await async_library.acall(calls)

        for library, response in (
            (sync_library, sync_response),
            (async_library, async_response),
        ):
            assert [call.result for call in response.tool_calls] == [
                "ready",
                "loaded=first_lookup",
                "loaded=second_lookup",
            ]
            assert [
                schema["function"]["name"]
                for schema in library.get_tool_json_schemas()
            ] == ["echo", "first_lookup", "second_lookup"]

    def test_search_tools_rejects_unsafe_regex_and_invalid_limit(self):
        @mf.tool_config(on_demand=True)
        def remote_lookup(query: str) -> str:
            """Look up external information."""
            return query

        library = ToolLibrary(name="lib", tools=[remote_lookup])

        unsafe = library(
            [("call_1", "search_tools", {"query": "/(a+)+$/"})]
        ).tool_calls[0]
        invalid_limit = library(
            [("call_2", "search_tools", {"query": "remote:0"})]
        ).tool_calls[0]

        assert "quantified groups" in unsafe.error
        assert "between 1 and 20" in invalid_limit.error

    def test_search_tools_rejects_duplicate_canonical_name_before_activation(self):
        """An on-demand tool cannot shadow a tool captured elsewhere."""

        class CatalogBucket(ToolBucket):
            name = "catalog"
            capture = {"tool_kind": "catalog", "on_demand": False}
            description = "Catalog operations."
            annotations = {"query": str, "return": str}

            def __call__(self, query: str) -> str:
                return query

        bucket = CatalogBucket()
        bucket.add(
            ToolMetadata(
                name="lookup",
                description="Existing lookup.",
                annotations={},
                tool_config={"tool_kind": "catalog", "on_demand": False},
                impl=lambda query: query,
            )
        )

        @mf.tool_config(
            on_demand=True,
            tool_kind="catalog",
            name_override="lookup",
        )
        def delayed_lookup(query: str) -> str:
            """Look up a catalog item later."""
            return query

        with pytest.raises(ValueError, match="Duplicate tool name `lookup`"):
            ToolLibrary(name="lib", tools=[bucket, delayed_lookup])

        assert set(bucket.tools) == {"lookup"}
        assert bucket.tools["lookup"].tool_config["on_demand"] is False

    def test_search_tools_is_removed_when_last_on_demand_tool_is_removed(self):
        """Test runtime tool cleanup when on-demand tools disappear."""

        @mf.tool_config(on_demand=True)
        def remote_lookup(query: str) -> str:
            """Look up external information."""
            return query

        library = ToolLibrary(name="lib", tools=[remote_lookup])

        assert "search_tools" in library.get_tool_names()

        library.remove("remote_lookup")

        assert "search_tools" not in library.get_tool_names()

    def test_search_tools_activates_populated_bucket_without_replacing_nodes(self):
        def leaf() -> str:
            """Return a deferred leaf."""
            return "leaf"

        class DeferredBucket(ToolBucket):
            """Expose one deferred bucket."""

            name = "deferred_bucket"
            capture = {"name": "leaf", "on_demand": False}
            annotations = {"return": str}

            def __call__(self) -> str:
                return "ready"

        bucket = DeferredBucket()
        bucket.add(
            ToolMetadata(
                name="leaf",
                description="Return a deferred leaf.",
                annotations={"return": str},
                tool_config={"on_demand": False},
                impl=leaf,
            )
        )
        deferred = mf.tool_config(on_demand=True)(bucket)
        library = ToolLibrary(name="lib", tools=[deferred])
        bucket_wrapper = library.library["deferred_bucket"]
        leaf_wrapper = library.library["leaf"]

        assert library.tool_owners == {
            "leaf": "deferred_bucket",
            "deferred_bucket": "search_tools",
        }
        assert [
            schema["function"]["name"]
            for schema in library.get_tool_json_schemas()
        ] == ["search_tools"]

        response = library(
            [("call_1", "search_tools", {"query": "deferred_bucket"})]
        )

        assert response.tool_calls[0].result == "loaded=deferred_bucket"
        assert library.tool_owners == {"leaf": "deferred_bucket"}
        assert library.library["deferred_bucket"] is bucket_wrapper
        assert library.library["leaf"] is leaf_wrapper
        assert [
            schema["function"]["name"]
            for schema in library.get_tool_json_schemas()
        ] == ["deferred_bucket"]

    def test_search_tools_activation_failure_restores_original_capture(self):
        @mf.tool_config(
            on_demand=True,
            tool_kind="catalog",
        )
        def deferred_lookup(query: str) -> str:
            """Run a deferred lookup."""
            return query

        class RejectingCatalog(ToolBucket):
            """Reject promoted catalog tools."""

            name = "catalog"
            capture = {"tool_kind": "catalog", "on_demand": False}
            annotations = {"return": str}

            def refresh(self):
                if self.tools:
                    raise ValueError("catalog rejected promotion")

            def __call__(self) -> str:
                return "catalog"

        library = ToolLibrary(
            name="lib",
            tools=[RejectingCatalog(), deferred_lookup],
        )
        wrapper = library.library["deferred_lookup"]

        response = library(
            [("call_1", "search_tools", {"query": "deferred_lookup"})]
        )

        assert "catalog rejected promotion" in response.tool_calls[0].error
        assert library.library["deferred_lookup"] is wrapper
        assert library.tool_owners["deferred_lookup"] == "search_tools"
        assert library.tool_configs["deferred_lookup"]["on_demand"] is True
        assert library.tool_configs["deferred_lookup"]["exposed"] is False
        assert set(library.library["search_tools"].impl.tools) == {
            "deferred_lookup"
        }

    @pytest.mark.asyncio
    async def test_model_calls_to_captured_tools_return_not_found(self):
        class Bucket(ToolBucket):
            """Capture a hidden leaf."""

            name = "bucket"
            capture = {"name": "hidden_leaf"}
            annotations = {"return": str}

            def __call__(self) -> str:
                return "bucket"

        def hidden_leaf() -> str:
            """Return a hidden value."""
            return "hidden"

        library = ToolLibrary(name="lib", tools=[Bucket(), hidden_leaf])

        sync_response = library([("sync", "hidden_leaf", {})])
        async_response = await library.aforward(
            [("async", "hidden_leaf", {})]
        )

        assert sync_response.tool_calls[0].error == (
            "Error: Tool `hidden_leaf` not found."
        )
        assert async_response.tool_calls[0].error == (
            "Error: Tool `hidden_leaf` not found."
        )

    def test_search_tools_cannot_be_removed_while_on_demand_tools_remain(self):
        """Test Tool Search retains its captured on-demand tools."""

        @mf.tool_config(on_demand=True)
        def remote_lookup(query: str) -> str:
            """Look up external information."""
            return query

        library = ToolLibrary(name="lib", tools=[remote_lookup])

        with pytest.raises(ValueError, match="still captures tools"):
            library.remove("search_tools")

        assert "search_tools" in library.library

    def test_injected_handle_can_add_on_demand_tool(self):
        """Test that injected handle can register on-demand tools."""

        @mf.tool_config(on_demand=True)
        def remote_lookup(query: str) -> str:
            """Look up external information."""
            return query

        @mf.tool_config(handle={"tools": ["list", "register"]})
        def enable_remote_lookup(
            handle: mf.Hidden,
        ) -> list[str]:
            """Register an on-demand tool."""
            handle.tools.register(remote_lookup)
            return handle.tools.list()

        library = ToolLibrary(name="lib", tools=[enable_remote_lookup])

        add_result = (
            library([("call_1", "enable_remote_lookup", {})]).tool_calls[0].result
        )
        schema_names = [
            schema["function"]["name"] for schema in library.get_tool_json_schemas()
        ]

        assert "remote_lookup" in add_result
        assert "search_tools" in add_result
        assert "search_tools" in schema_names
        assert "remote_lookup" not in schema_names

    def test_injected_handle_add_returns_normalized_tool_name(self):
        """Test that ToolHandle.tools.register returns the registered name."""

        @mf.tool_config(name_override="remote_lookup")
        def lookup(query: str) -> str:
            """Look up external information."""
            return query

        @mf.tool_config(handle={"tools": ["register"]})
        def enable_lookup(handle: mf.Hidden) -> str:
            """Register a tool."""
            return handle.tools.register(lookup)

        library = ToolLibrary(name="lib", tools=[enable_lookup])

        result = library([("call_1", "enable_lookup", {})]).tool_calls[0].result

        assert result == "remote_lookup"
        assert "remote_lookup" in library.get_tool_names()

    def test_tool_library_forward_basic(self):
        """Test ToolLibrary forward execution."""

        def add(a: int, b: int) -> int:
            """Add numbers."""
            return a + b

        library = ToolLibrary(name="lib", tools=[add])
        tool_callings = [("call_1", "add", {"a": 5, "b": 3})]

        result = library(tool_callings)

        assert isinstance(result, ToolResponses)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].result == 8

    def test_tool_library_forward_tool_not_found(self):
        """Test ToolLibrary forward with non-existent tool."""
        library = ToolLibrary(name="lib", tools=[])
        tool_callings = [("call_1", "nonexistent", {})]

        result = library(tool_callings)

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].error is not None
        assert "not found" in result.tool_calls[0].error

    def test_tool_library_forward_multiple_tools(self):
        """Test ToolLibrary forward with multiple tools."""

        def add(a: int, b: int) -> int:
            """Add."""
            return a + b

        def multiply(x: int, y: int) -> int:
            """Multiply."""
            return x * y

        library = ToolLibrary(name="lib", tools=[add, multiply])
        tool_callings = [
            ("call_1", "add", {"a": 2, "b": 3}),
            ("call_2", "multiply", {"x": 4, "y": 5}),
        ]

        result = library(tool_callings)

        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].result == 5
        assert result.tool_calls[1].result == 20

    @pytest.mark.asyncio
    async def test_tool_library_aforward_basic(self):
        """Test ToolLibrary async forward execution."""

        async def async_add(a: int, b: int) -> int:
            """Add numbers."""
            return a + b

        library = ToolLibrary(name="lib", tools=[async_add])
        tool_callings = [("call_1", "async_add", {"a": 10, "b": 20})]

        result = await library.aforward(tool_callings)

        assert isinstance(result, ToolResponses)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].result == 30

    @pytest.mark.asyncio
    async def test_tool_library_aforward_activity_recorder_uses_response_parameters(
        self,
    ):
        """Test async activity recording excludes hidden parameters."""
        task_store = InMemoryTaskStore()
        task = task_store.create(
            "worker",
            task_id="task_async_activity_sanitized",
            metadata={"task_kind": "agent"},
        )

        async def hidden_tool(name: str, secret: mf.Hidden[str] = "safe") -> str:
            """Hide a parameter from model-facing responses."""
            return f"{name}:{secret}"

        library = ToolLibrary(name="lib", tools=[hidden_tool])

        with execution_context(
            task_activity_recorder=TaskActivityRecorder(task.task_id, task_store)
        ):
            result = await library.aforward(
                [
                    (
                        "call_1",
                        "hidden_tool",
                        {"name": "lookup", "secret": "model"},
                    )
                ]
            )

        assert result.tool_calls[0].result == "lookup:safe"
        assert result.tool_calls[0].parameters == {"name": "lookup"}
        assert _activity_summaries(task_store, task.task_id) == [
            "Task queued.",
            "hidden_tool({'name': 'lookup'})",
        ]

    def test_tool_library_with_inject_vars_list(self):
        """Test ToolLibrary with inject_vars as list."""

        def tool_with_vars(a: int, injected: str) -> str:
            """Tool that uses injected var."""
            return f"{a}-{injected}"

        tool_with_vars.tool_config = {"inject_vars": ["injected"]}
        library = ToolLibrary(name="lib", tools=[tool_with_vars])

        tool_callings = [("call_1", "tool_with_vars", {"a": 5})]
        vars_dict = {"injected": "test"}

        result = library(tool_callings, vars=vars_dict)

        assert result.tool_calls[0].result == "5-test"

    def test_tool_library_with_inject_vars_true(self):
        """Test ToolLibrary with inject_vars=True (injects all vars)."""

        def tool_all_vars(x: int, vars: dict) -> int:
            """Tool that receives all vars."""
            return x + vars.get("extra", 0)

        tool_all_vars.tool_config = {"inject_vars": True}
        library = ToolLibrary(name="lib", tools=[tool_all_vars])

        tool_callings = [("call_1", "tool_all_vars", {"x": 10})]
        vars_dict = {"extra": 5}

        result = library(tool_callings, vars=vars_dict)

        assert result.tool_calls[0].result == 15

    def test_tool_library_inject_vars_missing_raises_error(self):
        """Test that missing injected var raises error."""

        def tool_needs_var(a: int, required: str) -> str:
            """Tool needs var."""
            return f"{a}-{required}"

        tool_needs_var.tool_config = {"inject_vars": ["required"]}
        library = ToolLibrary(name="lib", tools=[tool_needs_var])

        tool_callings = [("call_1", "tool_needs_var", {"a": 5})]

        with pytest.raises(ValueError, match="requires the injected parameter"):
            library(tool_callings, vars={})

    def test_tool_library_activity_recorder_uses_response_parameters(self):
        """Test activity recording excludes hidden and runtime parameters."""
        task_store = InMemoryTaskStore()
        task = task_store.create(
            "worker",
            task_id="task_activity_sanitized",
            metadata={"task_kind": "agent"},
        )

        @mf.tool_config(allow_background=True)
        def hidden_tool(name: str, secret: mf.Hidden[str] = "safe") -> str:
            """Hide a parameter from model-facing responses."""
            return f"{name}:{secret}"

        library = ToolLibrary(name="lib", tools=[hidden_tool])

        with execution_context(
            task_activity_recorder=TaskActivityRecorder(task.task_id, task_store)
        ):
            result = library(
                [
                    (
                        "call_1",
                        "hidden_tool",
                        {
                            "name": "lookup",
                            "secret": "model",
                            "run_in_background": False,
                        },
                    )
                ]
            )

        assert result.tool_calls[0].result == "lookup:safe"
        assert result.tool_calls[0].parameters == {"name": "lookup"}
        assert _activity_summaries(task_store, task.task_id) == [
            "Task queued.",
            "hidden_tool({'name': 'lookup'})",
        ]

    def test_tool_library_activity_recorder_waits_for_prepared_params(self):
        """Test invalid prepared parameters do not create a tool-call activity."""
        task_store = InMemoryTaskStore()
        task = task_store.create(
            "worker",
            task_id="task_activity_validation_error",
            metadata={"task_kind": "agent"},
        )

        @mf.tool_config(inject_vars=["required"])
        def tool_needs_var(a: int, required: str) -> str:
            """Tool needs var."""
            return f"{a}-{required}"

        library = ToolLibrary(name="lib", tools=[tool_needs_var])

        with pytest.raises(ValueError, match="requires the injected parameter"):
            with execution_context(
                task_activity_recorder=TaskActivityRecorder(task.task_id, task_store)
            ):
                library([("call_1", "tool_needs_var", {"a": 5})], vars={})

        assert _activity_summaries(task_store, task.task_id) == ["Task queued."]

    def test_tool_library_with_return_direct(self):
        """Test ToolLibrary with return_direct config."""

        def quick_tool(x: int) -> int:
            """Quick tool."""
            return x * 2

        quick_tool.tool_config = {"return_direct": True}
        library = ToolLibrary(name="lib", tools=[quick_tool])

        tool_callings = [("call_1", "quick_tool", {"x": 5})]
        result = library(tool_callings)

        assert result.return_directly is True

    def test_tool_library_with_call_as_response(self):
        """Test ToolLibrary with call_as_response config."""

        def response_tool(x: int) -> int:
            """Response tool."""
            return x

        response_tool.tool_config = {"call_as_response": True}
        library = ToolLibrary(name="lib", tools=[response_tool])

        tool_callings = [("call_1", "response_tool", {"x": 10})]
        result = library(tool_callings)

        assert result.return_directly is True
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].result is None  # Not executed, just returned

    def test_tool_library_with_inject_messages(self):
        """Test ToolLibrary with inject_messages config."""

        def stateful_tool(x: int, messages: dict) -> str:
            """Tool that uses model state."""
            return f"{x}-{messages.get('key', 'none')}"

        stateful_tool.tool_config = {"inject_messages": True}
        library = ToolLibrary(name="lib", tools=[stateful_tool])

        tool_callings = [("call_1", "stateful_tool", {"x": 5})]
        messages = {"key": "value"}

        result = library(tool_callings, messages=messages)

        assert result.tool_calls[0].result == "5-value"

    def test_tool_library_with_inject_messages_preserves_shared_messages(self):
        """Non-agent tools should receive the original messages reference."""

        def stateful_tool(messages: list) -> str:
            """Tool that mutates the shared conversation messages."""
            messages.append({"role": "tool", "content": "mutated"})
            messages[0]["content"] = "changed"
            return str(len(messages))

        stateful_tool.tool_config = {"inject_messages": True}
        library = ToolLibrary(name="lib", tools=[stateful_tool])

        original_messages = [{"role": "user", "content": "hello"}]
        tool_callings = [("call_1", "stateful_tool", {})]

        result = library(tool_callings, messages=original_messages)

        assert result.tool_calls[0].result == "2"
        assert original_messages == [
            {"role": "user", "content": "changed"},
            {"role": "tool", "content": "mutated"},
        ]

    def test_tool_library_with_agent_inject_messages_copies_per_subagent(self):
        """Forced agent isolation should copy messages per tool call."""

        def stateful_tool(messages: list) -> str:
            messages.append({"role": "tool", "content": "mutated"})
            messages[0]["content"] = "changed"
            return str(len(messages))

        stateful_tool.tool_config = {"inject_messages": True}
        library = ToolLibrary(name="lib", tools=[stateful_tool])

        original_messages = [{"role": "user", "content": "hello"}]
        tool_callings = [("call_1", "stateful_tool", {})]

        with patch(
            "msgflux.nn.modules.tool.should_copy_injected_messages",
            return_value=True,
        ):
            result = library(tool_callings, messages=original_messages)

        assert result.tool_calls[0].result == "2"
        assert original_messages == [{"role": "user", "content": "hello"}]

    def test_should_copy_injected_messages_for_wrapped_agent(self):
        """Wrapped Agent tools should opt into isolated message copies."""

        mock_model = Mock()
        mock_model.model_type = "chat_completion"
        agent = Agent(name="child_agent", model=mock_model)
        agent.tool_config = {"inject_messages": True, "disable_input": True}

        local_tool = _convert_module_to_nn_tool(agent)

        assert should_copy_injected_messages(local_tool, local_tool.tool_config) is True

    def test_tool_library_with_inject_message(self):
        """Test ToolLibrary with inject_message config."""

        def stateful_tool(x: int, message: dict) -> str:
            """Tool that uses the original message envelope."""
            return f"{x}-{message.get('key', 'none')}"

        stateful_tool.tool_config = {"inject_message": True}
        library = ToolLibrary(name="lib", tools=[stateful_tool])

        tool_callings = [("call_1", "stateful_tool", {"x": 5})]
        message = {"key": "value"}

        result = library(tool_callings, message=message)

        assert result.tool_calls[0].result == "5-value"

    def test_tool_library_with_handle_access(self):
        """Test ToolLibrary with exact handle access."""

        @mf.tool_config(handle={"tools": ["list"]})
        def runtime_tool(handle: mf.Hidden) -> str:
            """Tool that uses the runtime handle."""
            return ",".join(handle.tools.list())

        library = ToolLibrary(name="lib", tools=[runtime_tool])
        schema = next(
            item
            for item in library.get_tool_json_schemas()
            if item["function"]["name"] == "runtime_tool"
        )

        result = library([("call_1", "runtime_tool", {})])

        assert "handle" not in schema["function"]["parameters"].get("properties", {})
        assert "runtime_tool" in result.tool_calls[0].result

    def test_tool_library_tool_library_parameter_is_not_injected(self):
        """Test tool_library is a normal parameter, not a runtime alias."""

        def echo_tool_library(tool_library: str) -> str:
            """Echo the provided value."""
            return tool_library

        library = ToolLibrary(name="lib", tools=[echo_tool_library])
        schemas = library.get_tool_json_schemas()
        props = schemas[0]["function"]["parameters"].get("properties", {})
        result = library(
            [("call_1", "echo_tool_library", {"tool_library": "explicit"})]
        )

        assert "tool_library" in props
        assert result.tool_calls[0].parameters == {"tool_library": "explicit"}
        assert result.tool_calls[0].result == "explicit"

    def test_tool_library_with_disable_input_ignores_model_params(self):
        """Test ToolLibrary ignores model-supplied params when input is disabled."""

        def stateful_tool(messages: dict) -> str:
            """Tool that relies only on injected runtime state."""
            return messages.get("key", "none")

        stateful_tool.tool_config = {
            "disable_input": True,
            "inject_messages": True,
        }
        library = ToolLibrary(name="lib", tools=[stateful_tool])

        tool_callings = [("call_1", "stateful_tool", {"x": 5})]
        result = library(tool_callings, messages={"key": "value"})

        assert result.tool_calls[0].result == "value"
        assert "x" not in result.tool_calls[0].parameters

    @pytest.mark.asyncio
    async def test_tool_library_aforward_tool_not_found(self):
        """Test async ToolLibrary forward with non-existent tool."""
        library = ToolLibrary(name="lib", tools=[])
        tool_callings = [("call_1", "nonexistent", {})]

        result = await library.aforward(tool_callings)

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].error is not None

    @pytest.mark.asyncio
    async def test_tool_library_aforward_inject_vars(self):
        """Test async ToolLibrary with inject_vars."""

        async def async_tool(a: int, injected: str) -> str:
            """Async tool with vars."""
            return f"{a}-{injected}"

        async_tool.tool_config = {"inject_vars": ["injected"]}
        library = ToolLibrary(name="lib", tools=[async_tool])

        tool_callings = [("call_1", "async_tool", {"a": 3})]
        result = await library.aforward(tool_callings, vars={"injected": "test"})

        assert result.tool_calls[0].result == "3-test"

    def test_tool_library_empty_tool_callings(self):
        """Test ToolLibrary with empty tool callings."""

        def dummy(x: int) -> int:
            """Dummy."""
            return x

        library = ToolLibrary(name="lib", tools=[dummy])
        result = library([])

        assert result.return_directly is False
        assert len(result.tool_calls) == 0

    def test_tool_library_get_mcp_tool_names(self):
        """Test getting MCP tool names."""

        def local_tool(x: int) -> int:
            """Local."""
            return x

        library = ToolLibrary(name="lib", tools=[local_tool])
        mcp_names = library.get_mcp_tool_names()

        assert isinstance(mcp_names, list)
        assert len(mcp_names) == 0  # No MCP tools

    @pytest.mark.asyncio
    async def test_tool_library_aforward_inject_vars_missing_key(self):
        """Test async ToolLibrary inject_vars with missing key raises error."""

        async def async_tool(a: int, required_var: str) -> str:
            """Async tool requiring specific var."""
            return f"{a}-{required_var}"

        async_tool.tool_config = {"inject_vars": ["required_var"]}
        library = ToolLibrary(name="lib", tools=[async_tool])

        tool_callings = [("call_1", "async_tool", {"a": 3})]

        with pytest.raises(ValueError, match="requires the injected parameter"):
            await library.aforward(tool_callings, vars={"other_var": "test"})

    @pytest.mark.asyncio
    async def test_tool_library_aforward_inject_vars_true_mode(self):
        """Test async ToolLibrary with inject_vars=True."""

        async def async_tool(a: int, vars: dict) -> str:
            """Async tool with vars dict."""
            return f"{a}-{vars['key']}"

        async_tool.tool_config = {"inject_vars": True}
        library = ToolLibrary(name="lib", tools=[async_tool])

        tool_callings = [("call_1", "async_tool", {"a": 5})]
        result = await library.aforward(tool_callings, vars={"key": "value"})

        assert "5-value" in result.tool_calls[0].result

    @pytest.mark.asyncio
    async def test_tool_library_aforward_spawn(self):
        """Test async ToolLibrary spawn execution."""

        async def async_tool(x: int) -> int:
            """Spawn async tool."""
            return x * 2

        async_tool.tool_config = {"spawn": True}
        library = ToolLibrary(name="lib", tools=[async_tool])

        tool_callings = [("call_1", "async_tool", {"x": 10})]
        result = await library.aforward(tool_callings)

        assert result.return_directly is False
        assert "dispatched" in result.tool_calls[0].result.lower()

    @pytest.mark.asyncio
    async def test_tool_library_aforward_call_as_response(self):
        """Test async ToolLibrary call_as_response."""

        async def async_tool(x: int) -> int:
            """Tool with call as response."""
            return x * 3

        async_tool.tool_config = {"call_as_response": True}
        library = ToolLibrary(name="lib", tools=[async_tool])

        tool_callings = [("call_1", "async_tool", {"x": 7})]
        result = await library.aforward(tool_callings)

        assert result.return_directly is True
        assert result.tool_calls[0].result is None  # Not executed yet

    @pytest.mark.asyncio
    async def test_tool_library_aforward_inject_messages(self):
        """Test async ToolLibrary inject_messages."""

        async def async_tool(x: int, messages: dict) -> str:
            """Tool with model state."""
            return f"{x}-{messages['key']}"

        async_tool.tool_config = {"inject_messages": True}
        library = ToolLibrary(name="lib", tools=[async_tool])

        tool_callings = [("call_1", "async_tool", {"x": 8})]
        result = await library.aforward(tool_callings, messages={"key": "state_value"})

        assert "8-state_value" in result.tool_calls[0].result

    @pytest.mark.asyncio
    async def test_tool_library_aforward_inject_messages_preserves_shared_messages(
        self,
    ):
        """Async non-agent tools should receive the original messages reference."""

        async def async_tool(messages: list) -> str:
            """Tool that mutates the shared conversation messages."""
            messages.append({"role": "tool", "content": "mutated"})
            messages[0]["content"] = "changed"
            return str(len(messages))

        async_tool.tool_config = {"inject_messages": True}
        library = ToolLibrary(name="lib", tools=[async_tool])

        original_messages = [{"role": "user", "content": "hello"}]
        tool_callings = [("call_1", "async_tool", {})]

        result = await library.aforward(tool_callings, messages=original_messages)

        assert result.tool_calls[0].result == "2"
        assert original_messages == [
            {"role": "user", "content": "changed"},
            {"role": "tool", "content": "mutated"},
        ]

    @pytest.mark.asyncio
    async def test_tool_library_aforward_agent_inject_messages_copies_per_subagent(
        self,
    ):
        """Async forced agent isolation should copy messages per tool call."""

        async def async_tool(messages: list) -> str:
            messages.append({"role": "tool", "content": "mutated"})
            messages[0]["content"] = "changed"
            return str(len(messages))

        async_tool.tool_config = {"inject_messages": True}
        library = ToolLibrary(name="lib", tools=[async_tool])

        original_messages = [{"role": "user", "content": "hello"}]
        tool_callings = [("call_1", "async_tool", {})]

        with patch(
            "msgflux.nn.modules.tool.should_copy_injected_messages",
            return_value=True,
        ):
            result = await library.aforward(tool_callings, messages=original_messages)

        assert result.tool_calls[0].result == "2"
        assert original_messages == [{"role": "user", "content": "hello"}]

    @pytest.mark.asyncio
    async def test_tool_library_aforward_inject_message(self):
        """Test async ToolLibrary inject_message."""

        async def async_tool(x: int, message: dict) -> str:
            """Tool with original message envelope."""
            return f"{x}-{message['key']}"

        async_tool.tool_config = {"inject_message": True}
        library = ToolLibrary(name="lib", tools=[async_tool])

        tool_callings = [("call_1", "async_tool", {"x": 8})]
        result = await library.aforward(tool_callings, message={"key": "state_value"})

        assert "8-state_value" in result.tool_calls[0].result

    @pytest.mark.asyncio
    async def test_tool_library_aforward_handle_access(self):
        """Test async ToolLibrary handle access."""

        async def async_tool(handle: mf.Hidden) -> str:
            """Tool with runtime handle."""
            return ",".join(handle.tools.list())

        async_tool.tool_config = {"handle": {"tools": ["list"]}}
        library = ToolLibrary(name="lib", tools=[async_tool])

        tool_callings = [("call_1", "async_tool", {})]
        result = await library.aforward(tool_callings)

        assert "async_tool" in result.tool_calls[0].result

    @pytest.mark.asyncio
    async def test_tool_library_aforward_disable_input_ignores_model_params(self):
        """Test async ToolLibrary ignores model params when input is disabled."""

        async def async_tool(messages: dict) -> str:
            """Tool that relies only on injected runtime state."""
            return messages["key"]

        async_tool.tool_config = {
            "disable_input": True,
            "inject_messages": True,
        }
        library = ToolLibrary(name="lib", tools=[async_tool])

        tool_callings = [("call_1", "async_tool", {"x": 8})]
        result = await library.aforward(tool_callings, messages={"key": "state_value"})

        assert result.tool_calls[0].result == "state_value"
        assert "x" not in result.tool_calls[0].parameters

    def test_tool_library_forward_spawn(self):
        """Test ToolLibrary spawn execution in sync mode."""

        def sync_tool(x: int) -> int:
            """Spawn sync tool."""
            return x * 4

        sync_tool.tool_config = {"spawn": True}
        library = ToolLibrary(name="lib", tools=[sync_tool])

        tool_callings = [("call_1", "sync_tool", {"x": 5})]
        result = library(tool_callings)

        assert result.return_directly is False
        assert "dispatched" in result.tool_calls[0].result.lower()

    def test_tool_library_mcp_initialization_stdio(self):
        """Test ToolLibrary MCP initialization with stdio transport."""
        mcp_servers = [
            {
                "name": "test_server",
                "transport": "stdio",
                "command": "test_cmd",
                "args": ["--arg"],
                "timeout": 30.0,
            }
        ]

        with (
            patch("msgflux.nn.modules.tool.MCPClient") as mock_mcp_client_class,
            patch("msgflux.nn.modules.tool.F.wait_for") as mock_wait_for,
        ):
            mock_client = Mock()
            mock_tool_info = Mock()
            mock_tool_info.name = "test_tool"
            mock_tool_info.description = "Test"

            # Set up the wait_for calls for connect and list_tools
            mock_wait_for.side_effect = [None, [mock_tool_info]]

            mock_mcp_client_class.from_stdio.return_value = mock_client

            library = ToolLibrary(name="lib", tools=[], mcp_servers=mcp_servers)

            assert "test_server" in library.mcp_clients
            assert "test_server__test_tool" in library.library

    def test_tool_library_mcp_initialization_http(self):
        """Test ToolLibrary MCP initialization with http transport."""
        mcp_servers = [
            {
                "name": "http_server",
                "transport": "http",
                "base_url": "http://localhost:8000",
                "timeout": 30.0,
            }
        ]

        with (
            patch("msgflux.nn.modules.tool.MCPClient") as mock_mcp_client_class,
            patch("msgflux.nn.modules.tool.F.wait_for") as mock_wait_for,
        ):
            mock_client = Mock()
            mock_tool_info = Mock()
            mock_tool_info.name = "http_tool"
            mock_tool_info.description = "HTTP Test"

            # Set up the wait_for calls
            mock_wait_for.side_effect = [None, [mock_tool_info]]

            mock_mcp_client_class.from_http.return_value = mock_client

            library = ToolLibrary(name="lib", tools=[], mcp_servers=mcp_servers)

            assert "http_server" in library.mcp_clients
            assert "http_server__http_tool" in library.library

    def test_tool_library_mcp_initialization_invalid_transport(self):
        """Test ToolLibrary MCP initialization with invalid transport."""
        mcp_servers = [
            {
                "name": "bad_server",
                "transport": "invalid",
            }
        ]

        with patch("msgflux.nn.modules.tool.MCPClient"):
            with pytest.raises(ValueError, match="Unknown transport type"):
                ToolLibrary(name="lib", tools=[], mcp_servers=mcp_servers)

    def test_tool_library_mcp_initialization_missing_name(self):
        """Test ToolLibrary MCP initialization without name."""
        mcp_servers = [
            {
                "transport": "stdio",
                "command": "test",
            }
        ]

        with pytest.raises(ValueError, match="must include 'name' field"):
            ToolLibrary(name="lib", tools=[], mcp_servers=mcp_servers)

    def test_tool_library_mcp_initialization_with_filters(self):
        """Test ToolLibrary MCP initialization with include/exclude filters."""
        mcp_servers = [
            {
                "name": "filtered_server",
                "transport": "stdio",
                "command": "test",
                "include_tools": ["tool1"],
                "exclude_tools": ["tool2"],
            }
        ]

        with patch("msgflux.nn.modules.tool.MCPClient") as mock_mcp_client_class:
            mock_client = Mock()
            mock_tool1 = Mock()
            mock_tool1.name = "tool1"
            mock_tool1.description = "Tool 1"

            mock_client.connect = AsyncMock()
            mock_client.list_tools = AsyncMock(return_value=[mock_tool1])
            mock_mcp_client_class.from_stdio.return_value = mock_client

            with patch("msgflux.nn.modules.tool.filter_tools") as mock_filter:
                mock_filter.return_value = [mock_tool1]

                library = ToolLibrary(name="lib", tools=[], mcp_servers=mcp_servers)

                mock_filter.assert_called_once()
                assert "filtered_server__tool1" in library.library

    def test_tool_library_mcp_initialization_connection_error(self):
        """Test ToolLibrary MCP initialization handles connection errors gracefully."""
        mcp_servers = [
            {
                "name": "failing_server",
                "transport": "stdio",
                "command": "fail",
            }
        ]

        with patch("msgflux.nn.modules.tool.MCPClient") as mock_mcp_client_class:
            mock_client = Mock()
            mock_client.connect = AsyncMock(side_effect=Exception("Connection failed"))
            mock_mcp_client_class.from_stdio.return_value = mock_client

            # Should not raise, but log error
            library = ToolLibrary(name="lib", tools=[], mcp_servers=mcp_servers)

            # Server should not be added to mcp_clients
            assert "failing_server" not in library.mcp_clients


class TestMCPTool:
    """Test suite for MCPTool (requires mocking)."""

    def test_mcp_tool_initialization(self):
        """Test MCPTool basic initialization."""
        mock_client = Mock()
        mock_info = Mock()
        mock_info.description = "Test MCP tool"

        tool = MCPTool(
            name="read_file",
            mcp_client=mock_client,
            mcp_tool_info=mock_info,
            namespace="filesystem",
        )

        assert tool.name == "filesystem__read_file"
        assert tool.description == "Test MCP tool"
        assert tool._namespace == "filesystem"
        assert tool._mcp_tool_name == "read_file"

    def test_mcp_tool_get_json_schema(self):
        """Test MCPTool get_json_schema."""
        mock_client = Mock()
        mock_info = Mock()
        mock_info.name = "test_tool"
        mock_info.description = "Test tool"
        mock_info.inputSchema = {
            "type": "object",
            "properties": {"arg": {"type": "string"}},
        }

        tool = MCPTool(
            name="test_tool",
            mcp_client=mock_client,
            mcp_tool_info=mock_info,
            namespace="test",
        )

        schema = tool.get_json_schema()

        assert isinstance(schema, dict)
        assert schema["type"] == "function"

    def test_mcp_tool_with_config(self):
        """Test MCPTool with configuration."""
        mock_client = Mock()
        mock_info = Mock()
        mock_info.description = "Test tool with config"

        tool = MCPTool(
            name="tool",
            mcp_client=mock_client,
            mcp_tool_info=mock_info,
            namespace="test",
            config={"timeout": 30},
        )

        assert tool.tool_config["timeout"] == 30

    def test_mcp_tool_display_name_and_usage_guidance_are_not_duplicated(self):
        """Test MCP metadata is read from library tools without duplicate entries."""
        mock_client = Mock()
        mock_info = Mock()
        mock_info.name = "search"
        mock_info.description = "Search docs"
        mock_info.inputSchema = {"type": "object", "properties": {}}

        tool = MCPTool(
            name="search",
            mcp_client=mock_client,
            mcp_tool_info=mock_info,
            namespace="docs",
            config={
                "display_name": "Docs Search",
                "usage_guidance": "Use for documentation questions.",
            },
        )
        library = ToolLibrary(name="lib", tools=[])
        library.library[tool.name] = tool
        library.mcp_clients["docs"] = {
            "client": mock_client,
            "tools": [mock_info],
            "tool_config": {
                "search": {
                    "display_name": "Docs Search",
                    "usage_guidance": "Use for documentation questions.",
                }
            },
        }

        assert library.get_tool_display_names() == {"docs__search": "Docs Search"}
        assert library.get_tool_usage_guidance() == [
            {
                "name": "docs__search",
                "display_name": "Docs Search",
                "guidance": "Use for documentation questions.",
            }
        ]

    def test_mcp_tool_display_name_falls_back_to_registered_name(self):
        """Test MCP display_name falls back to the registered tool name."""
        mock_client = Mock()
        mock_info = Mock()
        mock_info.name = "edit"
        mock_info.description = "Edit docs"
        mock_info.inputSchema = {"type": "object", "properties": {}}

        tool = MCPTool(
            name="edit",
            mcp_client=mock_client,
            mcp_tool_info=mock_info,
            namespace="docs",
            config={"display_name": None},
        )
        library = ToolLibrary(name="lib", tools=[])
        library.library[tool.name] = tool

        assert tool.display_name == "docs__edit"
        assert library.get_tool_display_names() == {"docs__edit": "docs__edit"}

    def test_mcp_tool_forward_success(self):
        """Test MCPTool forward execution with success."""
        with (
            patch("msgflux.nn.modules.tool.F.wait_for") as mock_wait_for,
            patch("msgflux.nn.modules.tool.extract_tool_result_text") as mock_extract,
        ):
            mock_client = Mock()
            mock_info = Mock()
            mock_info.description = "Test tool"

            # Mock successful result
            mock_result = Mock()
            mock_result.isError = False

            mock_wait_for.return_value = mock_result
            mock_extract.return_value = "Success result"

            tool = MCPTool(
                name="test",
                mcp_client=mock_client,
                mcp_tool_info=mock_info,
                namespace="ns",
            )

            result = tool(arg="value")
            assert result == "Success result"
            mock_wait_for.assert_called_once()

    def test_mcp_tool_forward_error(self):
        """Test MCPTool forward execution with error."""
        with (
            patch("msgflux.nn.modules.tool.F.wait_for") as mock_wait_for,
            patch("msgflux.nn.modules.tool.extract_tool_result_text") as mock_extract,
        ):
            mock_client = Mock()
            mock_info = Mock()
            mock_info.description = "Test tool"

            # Mock error result
            mock_result = Mock()
            mock_result.isError = True

            mock_wait_for.return_value = mock_result
            mock_extract.return_value = "Error message"

            tool = MCPTool(
                name="test",
                mcp_client=mock_client,
                mcp_tool_info=mock_info,
                namespace="ns",
            )

            with pytest.raises(RuntimeError, match="MCP tool error"):
                tool(arg="value")

    @pytest.mark.asyncio
    async def test_mcp_tool_aforward_success(self):
        """Test MCPTool aforward execution with success."""
        with patch("msgflux.nn.modules.tool.extract_tool_result_text") as mock_extract:
            mock_client = Mock()
            mock_info = Mock()
            mock_info.description = "Test tool"

            # Mock successful result
            mock_result = Mock()
            mock_result.isError = False

            mock_client.call_tool = AsyncMock(return_value=mock_result)
            mock_extract.return_value = "Async success"

            tool = MCPTool(
                name="test",
                mcp_client=mock_client,
                mcp_tool_info=mock_info,
                namespace="ns",
            )

            result = await tool.acall(arg="value")
            assert result == "Async success"

    @pytest.mark.asyncio
    async def test_mcp_tool_aforward_error(self):
        """Test MCPTool aforward execution with error."""
        with patch("msgflux.nn.modules.tool.extract_tool_result_text") as mock_extract:
            mock_client = Mock()
            mock_info = Mock()
            mock_info.description = "Test tool"

            # Mock error result
            mock_result = Mock()
            mock_result.isError = True

            mock_client.call_tool = AsyncMock(return_value=mock_result)
            mock_extract.return_value = "Async error"

            tool = MCPTool(
                name="test",
                mcp_client=mock_client,
                mcp_tool_info=mock_info,
                namespace="ns",
            )

            with pytest.raises(RuntimeError, match="MCP tool error"):
                await tool.acall(arg="value")
