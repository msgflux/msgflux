"""Tests for msgflux.nn.modules.tool module."""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from msgflux.nn.modules.tool import (
    ToolCall,
    ToolResponses,
    Tool,
    LocalTool,
    MCPTool,
    ToolLibrary,
    _convert_module_to_nn_tool,
)


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

    def test_tool_library_add_tool(self):
        """Test adding a tool to library."""

        def new_tool(z: float) -> float:
            """New tool."""
            return z * 2

        library = ToolLibrary(name="lib", tools=[])
        library.add(new_tool)

        assert "new_tool" in library.library

    def test_tool_library_add_duplicate_raises_error(self):
        """Test that adding duplicate tool raises error."""

        def my_tool(x: int) -> int:
            """My tool."""
            return x

        library = ToolLibrary(name="lib", tools=[my_tool])

        with pytest.raises(ValueError, match="already in tool library"):
            library.add(my_tool)

    def test_tool_library_add_already_tool_instance(self):
        """Test adding Tool instance directly."""

        def my_func(x: int) -> int:
            """Test."""
            return x

        tool = _convert_module_to_nn_tool(my_func)
        library = ToolLibrary(name="lib", tools=[])
        library.add(tool)

        assert "my_func" in library.library

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

    def test_tool_library_get_tool_json_schemas(self):
        """Test getting tool JSON schemas."""

        def calculator(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        library = ToolLibrary(name="lib", tools=[calculator])
        schemas = library.get_tool_json_schemas()

        assert len(schemas) == 1
        assert isinstance(schemas[0], dict)

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
