"""Tests for msgflux.nn.modules.tool module."""

import msgflux as mf
import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Optional

from msgflux.core.dotdict import dotdict
from msgflux.chat_messages import ChatMessages
from msgflux.nn import ContextBinding
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
from msgflux.tools import ToolBackground, ToolBucket, ToolLibraryOperator
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
        """Test converting with detached configuration."""

        def dispatched_task(data: str) -> None:
            """Dispatch task without return."""
            pass

        dispatched_task.tool_config = {"detached": True}
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

    def test_tool_library_rejects_removed_spawn_option(self):
        def legacy_task() -> str:
            """Run a detached task."""
            return "done"

        legacy_task.tool_config = {"spawn": True}

        with pytest.raises(ValueError, match="removed; use `detached`"):
            ToolLibrary(name="lib", tools=[legacy_task])

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

    def test_tool_library_skips_background_validation_for_regular_tool(self):
        def regular_tool() -> str:
            """Return a regular value."""
            return "ok"

        with patch.object(
            ToolBackground, "validate_background_capabilities"
        ) as validate:
            library = ToolLibrary(name="test_library", tools=[regular_tool])

        assert "regular_tool" in library.library
        validate.assert_not_called()

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

    def test_tool_library_compiles_config_into_definition(self):
        """Test ToolLibrary stores declarations in canonical definitions."""

        def my_tool(x: int) -> int:
            """Tool."""
            return x

        my_tool.tool_config = {"return_direct": True}
        library = ToolLibrary(name="lib", tools=[my_tool])

        definition = library.get_tool_definition("my_tool")
        assert definition.declaration["return_direct"] is True
        assert definition.feedback.name == "direct"

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

    def test_tool_library_hides_deferred_tools_from_schemas(self):
        """Test that deferred tools are hidden until loaded."""

        @mf.tool_config(defer_loading=True)
        def remote_lookup(query: str) -> str:
            """Look up external information."""
            return query

        library = ToolLibrary(name="lib", tools=[remote_lookup])

        names = library.get_tool_names()
        schemas = library.get_tool_json_schemas()

        assert "remote_lookup" in names
        assert "tool_search" in names
        assert [schema["function"]["name"] for schema in schemas] == ["tool_search"]
        parameters = schemas[0]["function"]["parameters"]
        properties = parameters["properties"]
        assert properties["query"]["description"] == "Keywords used to find tools."
        assert properties["select"]["description"] == (
            "Exact tool names to load for the current thread."
        )
        assert parameters["required"] == [
            "query",
            "select",
            "description",
            "max_results",
        ]
        assert {"type": "null"} in properties["query"]["anyOf"]
        assert {"type": "null"} in properties["max_results"]["anyOf"]
        assert isinstance(library.library["tool_search"].impl, ToolLibraryOperator)
        assert isinstance(library.library["tool_search"].impl, ToolBucket)
        assert [
            binding.source
            for binding in library.get_tool_definition("tool_search").context.bindings
        ] == ["handle", "messages"]
        assert library.library["tool_search"].tool_config["tool_kind"] == "bucket"

    def test_tool_search_is_not_captured_by_tool_search_bucket(self):
        """Test tool_search stays registered even if a search bucket is added."""

        @mf.tool_config(defer_loading=True)
        def remote_lookup(query: str) -> str:
            """Look up external information."""
            return query

        class SearchBucket(ToolBucket):
            name = "search_bucket"
            capture = {"tool_kind": "bucket", "defer_loading": False}
            description = "Capture search tools."
            annotations = {"query": str, "return": str}

            def __call__(self, query: str) -> str:
                return query

        library = ToolLibrary(name="lib", tools=[remote_lookup])
        library.add(SearchBucket())

        assert "tool_search" in library.library
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
            capture = {"tool_kind": "catalog|orders", "defer_loading": False}
            annotations = {"return": str}

            def __call__(self) -> str:
                return "commerce"

        bucket = CommerceBucket()
        library = ToolLibrary(
            name="lib",
            tools=[find_product, list_products, get_order, bucket],
        )

        assert list(library.library) == ["commerce"]
        assert set(bucket.tools) == {"find_product", "list_products", "get_order"}
        assert library.get_tool_definition("commerce").kind == "bucket"
        assert library.get_tool_definition("find_product").kind == "catalog"

        with pytest.raises(ValueError, match="Duplicate tool name `find_product`"):
            library.add(find_product)

    def test_tool_bucket_handle_executes_captured_local_tool(self):
        @mf.tool_config(tool_kind="catalog")
        def find_product(query: str) -> str:
            """Find a product."""
            return f"match:{query}"

        class CatalogBucket(ToolBucket, ToolLibraryOperator):
            """Dispatch catalog operations."""

            name = "catalog"
            capture = {"tool_kind": "catalog", "defer_loading": False}
            annotations = {
                "name": str,
                "query": str,
                "handle": mf.Hidden,
                "return": str,
            }

            def __call__(self, name: str, query: str, *, handle) -> str:
                return handle(name, query=query)

        library = ToolLibrary(name="lib", tools=[CatalogBucket(), find_product])
        activity_recorder = Mock()

        with execution_context(task_activity_recorder=activity_recorder):
            response = library(
                [
                    (
                        "call_1",
                        "catalog",
                        {"name": "find_product", "query": "SKU-1"},
                    )
                ]
            )

        assert response.tool_calls[0].result == "match:SKU-1"
        activity_recorder.tool_call.assert_called_once_with(
            "find_product",
            {"query": "SKU-1"},
        )

    @pytest.mark.asyncio
    async def test_tool_bucket_handle_async_executes_captured_tool(self):
        async def find_product(query: str) -> str:
            """Find a product."""
            return f"match:{query}"

        find_product.tool_config = dotdict({"tool_kind": "catalog"})

        class CatalogBucket(ToolBucket, ToolLibraryOperator):
            """Dispatch catalog operations."""

            name = "catalog"
            capture = {"tool_kind": "catalog", "defer_loading": False}
            annotations = {
                "name": str,
                "query": str,
                "handle": mf.Hidden,
                "return": str,
            }

            async def acall(self, name: str, query: str, *, handle) -> str:
                return await handle.acall(name, query=query)

            def __call__(self, name: str, query: str, *, handle) -> str:
                return handle(name, query=query)

        library = ToolLibrary(name="lib", tools=[CatalogBucket(), find_product])

        response = await library.acall(
            [("call_1", "catalog", {"name": "find_product", "query": "SKU-1"})]
        )

        assert response.tool_calls[0].result == "match:SKU-1"

    def test_tool_bucket_handle_preserves_non_agent_context_identity(self):
        seen = {}

        @mf.tool_config(
            tool_kind="catalog",
            runtime_inputs=["messages", "vars"],
        )
        def inspect_context(messages: list, vars: dict) -> str:
            """Inspect the exact context objects received by a captured tool."""
            seen["messages"] = messages
            seen["vars"] = vars
            return "ok"

        class CatalogBucket(ToolBucket, ToolLibraryOperator):
            """Dispatch catalog operations."""

            name = "catalog"
            capture = {"tool_kind": "catalog", "defer_loading": False}
            annotations = {
                "name": str,
                "handle": mf.Hidden,
                "return": str,
            }

            def __call__(self, name: str, *, handle) -> str:
                return handle(name)

        library = ToolLibrary(name="lib", tools=[CatalogBucket(), inspect_context])
        messages = []
        runtime_vars = {}

        response = library(
            [("call_1", "catalog", {"name": "inspect_context"})],
            messages=messages,
            vars=runtime_vars,
        )

        assert response.tool_calls[0].result == "ok"
        assert seen["messages"] is messages
        assert seen["vars"] is runtime_vars

    @pytest.mark.parametrize(
        "option",
        [
            "background",
            "allow_background",
            "detached",
            "call_as_response",
            "return_direct",
            "handoff",
        ],
    )
    def test_executable_tool_bucket_rejects_captured_model_loop_options(
        self,
        option,
    ):
        def catalog_tool() -> str:
            """Run a catalog operation."""
            return "ok"

        catalog_tool.tool_config = {
            "tool_kind": "catalog",
            option: True,
        }

        class CatalogBucket(ToolBucket, ToolLibraryOperator):
            """Dispatch catalog operations."""

            name = "catalog"
            capture = {"tool_kind": "catalog", "defer_loading": False}
            annotations = {"return": str}

            def __call__(self) -> str:
                return "ok"

        with pytest.raises(
            ValueError,
            match="Configure that behavior on the public bucket",
        ):
            ToolLibrary(name="lib", tools=[CatalogBucket(), catalog_tool])

    def test_tool_search_keeps_deferred_tool_loop_options_as_catalog_metadata(self):
        @mf.tool_config(defer_loading=True, return_direct=True)
        def deferred_lookup() -> str:
            """Run a deferred lookup."""
            return "ok"

        library = ToolLibrary(name="lib", tools=[deferred_lookup])

        assert library.get_tool_names() == ["tool_search", "deferred_lookup"]

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
                "defer_loading": False,
            }
            annotations = {"return": str}

            def __call__(self) -> str:
                return "background"

        bucket = BackgroundBucket()
        library = ToolLibrary(
            name="lib",
            tools=[background_job, optional_background_job, bucket],
        )

        assert list(library.library) == ["background_jobs"]
        assert set(bucket.tools) == {"background_job", "optional_background_job"}

    def test_tool_bucket_rejects_overlapping_capture_rules(self):
        class FirstBucket(ToolBucket):
            """Capture catalog tools."""

            name = "first"
            capture = {"tool_kind": "catalog|orders", "defer_loading": False}

            def __call__(self) -> str:
                return "first"

        class SecondBucket(ToolBucket):
            """Capture order tools."""

            name = "second"
            capture = {"tool_kind": "orders|billing", "defer_loading": False}

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

        render_preview.tool_config = {"preview": True}
        bucket = PreviewBucket()
        library = ToolLibrary(name="lib", tools=[render_preview, bucket])

        assert list(library.library) == ["preview"]
        assert set(bucket.tools) == {"render_preview"}

    def test_tool_library_operator_injects_handle_by_default(self):
        """Test operator tools inherit handle injection."""

        class RuntimeEchoTool(ToolLibraryOperator):
            name = "runtime_echo"
            tool_kind = "diagnostic"
            description = "List the current tool names."
            annotations = {"handle": mf.Hidden, "return": str}

            def __call__(self, handle):
                return ",".join(handle.list_tools())

        library = ToolLibrary(name="lib", tools=[RuntimeEchoTool()])
        schema = next(
            item
            for item in library.get_tool_json_schemas()
            if item["function"]["name"] == "runtime_echo"
        )
        result = library([("call_1", "runtime_echo", {})])

        assert "handle" not in schema["function"]["parameters"].get("properties", {})
        assert result.tool_calls[0].result == "runtime_echo"
        assert [
            binding.source
            for binding in library.get_tool_definition("runtime_echo").context.bindings
        ] == ["handle"]
        assert library.library["runtime_echo"].tool_config["tool_kind"] == "diagnostic"

    def test_tool_search_captures_deferred_operator_tools(self):
        """Test deferred operators use the same ToolSearch bucket."""

        class DeferredOperator(ToolLibraryOperator):
            name = "deferred_operator"
            description = "List the currently registered tools."
            annotations = {"handle": mf.Hidden, "return": list[str]}
            tool_config = {"inject_handle": True, "defer_loading": True}

            def __call__(self, handle) -> list[str]:
                return handle.list_tools()

        library = ToolLibrary(name="lib", tools=[DeferredOperator()])

        assert [
            schema["function"]["name"] for schema in library.get_tool_json_schemas()
        ] == ["tool_search"]

        library([("call_1", "tool_search", {"select": ["deferred_operator"]})])
        response = library([("call_2", "deferred_operator", {})])

        assert "deferred_operator" in response.tool_calls[0].result

    def test_tool_search_has_default_usage_guidance(self):
        """Test tool_search exposes default guidance when deferred tools exist."""

        @mf.tool_config(defer_loading=True)
        def remote_lookup(query: str) -> str:
            """Look up external information."""
            return query

        library = ToolLibrary(name="lib", tools=[remote_lookup])

        guidance = library.get_tool_usage_guidance()

        assert guidance == [
            {
                "name": "tool_search",
                "display_name": "Tool Search",
                "guidance": (
                    "Search first; activate an exact match with `select` before "
                    "calling it."
                ),
            }
        ]

    def test_tool_search_returns_matching_deferred_tools_without_loading(self):
        """Test that keyword search describes matches without exposing them."""

        @mf.tool_config(defer_loading=True)
        def remote_lookup(query: str) -> str:
            """Look up external information."""
            return query

        library = ToolLibrary(name="lib", tools=[remote_lookup])

        result = (
            library(
                [
                    (
                        "call_1",
                        "tool_search",
                        {"query": "remote lookup", "description": True},
                    )
                ]
            )
            .tool_calls[0]
            .result
        )
        schemas = library.get_tool_json_schemas()
        schema_names = [schema["function"]["name"] for schema in schemas]

        assert result["matches"] == ["remote_lookup"]
        assert result["loaded"] == []
        assert result["descriptions"][0]["name"] == "remote_lookup"
        assert "already_loaded" not in result
        assert "tool_search" in schema_names
        assert "remote_lookup" not in schema_names

    def test_tool_search_select_supports_explicit_and_legacy_names(self):
        """Test that tool_search supports explicit selection and legacy syntax."""

        @mf.tool_config(defer_loading=True)
        def read_cloud_file(path: str) -> str:
            """Read a cloud file."""
            return path

        @mf.tool_config(defer_loading=True)
        def read_legacy_file(path: str) -> str:
            """Read a legacy file."""
            return path

        library = ToolLibrary(
            name="lib",
            tools=[read_cloud_file, read_legacy_file],
        )

        explicit_result = (
            library([("call_1", "tool_search", {"select": ["read_cloud_file"]})])
            .tool_calls[0]
            .result
        )
        legacy_result = (
            library([("call_2", "tool_search", {"query": "select:read_legacy_file"})])
            .tool_calls[0]
            .result
        )

        assert explicit_result["matches"] == ["read_cloud_file"]
        assert explicit_result["loaded"] == ["read_cloud_file"]
        assert legacy_result["matches"] == ["read_legacy_file"]
        assert legacy_result["loaded"] == ["read_legacy_file"]

    def test_tool_search_loading_is_isolated_by_thread(self):
        """Loading a deferred tool must not change another thread's catalog."""

        @mf.tool_config(defer_loading=True)
        def lookup(query: str) -> str:
            """Look up a catalog item."""
            return query

        library = ToolLibrary(name="lib", tools=[lookup])
        first = ChatMessages(thread_id="thread_a")
        second = ChatMessages(thread_id="thread_b")

        response = library(
            [("call_1", "tool_search", {"select": ["lookup"]})],
            messages=first,
        )

        assert response.tool_calls[0].result["loaded"] == ["lookup"]
        assert first.get_loaded_tools(library.name) == {"lookup"}
        assert second.get_loaded_tools(library.name) == set()
        assert [
            tool.name for tool in library.get_tool_catalog(first).portable_tools()
        ] == ["lookup"]
        assert [
            tool.name for tool in library.get_tool_catalog(second).portable_tools()
        ] == ["tool_search"]
        assert "lookup" not in library.library

    def test_tool_search_is_removed_when_last_deferred_tool_is_removed(self):
        """Test runtime tool cleanup when deferred tools disappear."""

        @mf.tool_config(defer_loading=True)
        def remote_lookup(query: str) -> str:
            """Look up external information."""
            return query

        library = ToolLibrary(name="lib", tools=[remote_lookup])

        assert "tool_search" in library.get_tool_names()

        library.remove("remote_lookup")

        assert "tool_search" not in library.get_tool_names()

    def test_tool_search_cannot_be_removed_while_deferred_tools_remain(self):
        """Test Tool Search retains its captured deferred tools."""

        @mf.tool_config(defer_loading=True)
        def remote_lookup(query: str) -> str:
            """Look up external information."""
            return query

        library = ToolLibrary(name="lib", tools=[remote_lookup])

        with pytest.raises(ValueError, match="still captures tools"):
            library.remove("tool_search")

        assert "tool_search" in library.library

    def test_injected_handle_can_add_deferred_tool(self):
        """Test that an injected handle can register deferred tools."""

        @mf.tool_config(defer_loading=True)
        def remote_lookup(query: str) -> str:
            """Look up external information."""
            return query

        @mf.tool_config(runtime_inputs=["handle"])
        def enable_remote_lookup(
            handle: mf.Hidden,
        ) -> list[str]:
            """Register a deferred tool."""
            handle.add(remote_lookup)
            return handle.list_tools()

        library = ToolLibrary(name="lib", tools=[enable_remote_lookup])

        add_result = (
            library([("call_1", "enable_remote_lookup", {})]).tool_calls[0].result
        )
        schema_names = [
            schema["function"]["name"] for schema in library.get_tool_json_schemas()
        ]

        assert "remote_lookup" in add_result
        assert "tool_search" in add_result
        assert "tool_search" in schema_names
        assert "remote_lookup" not in schema_names

    def test_injected_handle_add_returns_normalized_tool_name(self):
        """Test that ToolLibraryHandle.add returns the registered tool name."""

        @mf.tool_config(name_override="remote_lookup")
        def lookup(query: str) -> str:
            """Look up external information."""
            return query

        @mf.tool_config(runtime_inputs=["handle"])
        def enable_lookup(handle: mf.Hidden) -> str:
            """Register a tool."""
            return handle.add(lookup)

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

        @mf.tool_config(
            runtime_inputs=[
                ContextBinding(
                    source="vars",
                    parameter="required",
                    options={"key": "required"},
                )
            ]
        )
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

        stateful_tool.tool_config = {
            "inject_messages": True,
            "tool_kind": "agent",
        }
        library = ToolLibrary(name="lib", tools=[stateful_tool])

        original_messages = [{"role": "user", "content": "hello"}]
        tool_callings = [("call_1", "stateful_tool", {})]

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

    def test_tool_library_with_inject_handle(self):
        """Test ToolLibrary with inject_handle config."""

        @mf.tool_config(runtime_inputs=["handle"])
        def runtime_tool(handle: mf.Hidden) -> str:
            """Tool that uses the runtime handle."""
            return ",".join(handle.list_tools())

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
        """Test async ToolLibrary detached execution."""

        async def async_tool(x: int) -> int:
            """Run an async tool in detached mode."""
            return x * 2

        async_tool.tool_config = {"detached": True}
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

        async_tool.tool_config = {
            "inject_messages": True,
            "tool_kind": "agent",
        }
        library = ToolLibrary(name="lib", tools=[async_tool])

        original_messages = [{"role": "user", "content": "hello"}]
        tool_callings = [("call_1", "async_tool", {})]

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
    async def test_tool_library_aforward_inject_handle(self):
        """Test async ToolLibrary inject_handle."""

        async def async_tool(handle: mf.Hidden) -> str:
            """Tool with runtime handle."""
            return ",".join(handle.list_tools())

        async_tool.tool_config = {"inject_handle": True}
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
        """Test ToolLibrary detached execution in sync mode."""

        def sync_tool(x: int) -> int:
            """Run a sync tool in detached mode."""
            return x * 4

        sync_tool.tool_config = {"detached": True}
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
            patch(
                "msgflux.nn.extensions.tool_library.MCPClient"
            ) as mock_mcp_client_class,
            patch("msgflux.nn.extensions.tool_library.F.wait_for") as mock_wait_for,
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
            patch(
                "msgflux.nn.extensions.tool_library.MCPClient"
            ) as mock_mcp_client_class,
            patch("msgflux.nn.extensions.tool_library.F.wait_for") as mock_wait_for,
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

        with patch("msgflux.nn.extensions.tool_library.MCPClient"):
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

        with patch(
            "msgflux.nn.extensions.tool_library.MCPClient"
        ) as mock_mcp_client_class:
            mock_client = Mock()
            mock_tool1 = Mock()
            mock_tool1.name = "tool1"
            mock_tool1.description = "Tool 1"

            mock_client.connect = AsyncMock()
            mock_client.list_tools = AsyncMock(return_value=[mock_tool1])
            mock_mcp_client_class.from_stdio.return_value = mock_client

            with patch(
                "msgflux.nn.extensions.tool_library.filter_tools"
            ) as mock_filter:
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

        with patch(
            "msgflux.nn.extensions.tool_library.MCPClient"
        ) as mock_mcp_client_class:
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

    def test_tool_bucket_handle_preserves_mcp_proxy_execution(self):
        class MockClient:
            def __init__(self):
                self.calls = []

            async def call_tool(self, name, arguments):
                self.calls.append((name, arguments))
                return mock_result

        mock_result = Mock()
        mock_result.isError = False
        mock_client = MockClient()
        mock_info = Mock(
            name="search",
            description="Search the remote catalog.",
            inputSchema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
            },
        )
        remote_tool = MCPTool(
            name="search",
            mcp_client=mock_client,
            mcp_tool_info=mock_info,
            namespace="catalog",
            config={"tool_kind": "catalog"},
        )

        class CatalogBucket(ToolBucket, ToolLibraryOperator):
            """Dispatch remote catalog operations."""

            name = "catalog"
            capture = {"tool_kind": "catalog"}
            annotations = {
                "name": str,
                "query": str,
                "handle": mf.Hidden,
                "return": str,
            }

            def __call__(self, name: str, query: str, *, handle) -> str:
                return handle(name, query=query)

        library = ToolLibrary(name="lib", tools=[CatalogBucket(), remote_tool])

        with patch(
            "msgflux.nn.modules.tool.implementations.extract_tool_result_text",
            return_value="remote match",
        ):
            response = library(
                [
                    (
                        "call_1",
                        "catalog",
                        {"name": "catalog__search", "query": "SKU-1"},
                    )
                ]
            )

        assert response.tool_calls[0].result == "remote match"
        assert mock_client.calls == [("search", {"query": "SKU-1"})]

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
        library = ToolLibrary(name="lib", tools=[tool])
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
        library = ToolLibrary(name="lib", tools=[tool])

        assert tool.display_name == "docs__edit"
        assert library.get_tool_display_names() == {"docs__edit": "docs__edit"}

    def test_mcp_tool_forward_success(self):
        """Test MCPTool forward execution with success."""
        with (
            patch(
                "msgflux.nn.modules.tool.implementations.F.wait_for"
            ) as mock_wait_for,
            patch(
                "msgflux.nn.modules.tool.implementations.extract_tool_result_text"
            ) as mock_extract,
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
            patch(
                "msgflux.nn.modules.tool.implementations.F.wait_for"
            ) as mock_wait_for,
            patch(
                "msgflux.nn.modules.tool.implementations.extract_tool_result_text"
            ) as mock_extract,
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
        with patch(
            "msgflux.nn.modules.tool.implementations.extract_tool_result_text"
        ) as mock_extract:
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
        with patch(
            "msgflux.nn.modules.tool.implementations.extract_tool_result_text"
        ) as mock_extract:
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
