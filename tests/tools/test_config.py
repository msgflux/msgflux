"""Unit tests for msgflux.tools.config module."""

import pytest

from msgflux.core.dotdict import dotdict
from msgflux.nn import ContextBinding
from msgflux.nn.modules.tool import ToolLibrary
from msgflux.tools.config import decorate_function, decorate_instance, tool_config


class TestToolConfig:
    """Test suite for tool_config decorator."""

    def test_tool_config_on_function(self):
        """Test tool_config decorator on a regular function."""

        @tool_config(return_direct=True, detached=False)
        def sample_function(x: int) -> int:
            return x * 2

        assert hasattr(sample_function, "tool_config")
        assert sample_function.tool_config.return_direct is True
        assert sample_function.tool_config.detached is False
        assert sample_function(5) == 10

    def test_tool_config_on_method(self):
        """Test tool_config decorator on a method."""

        class SampleClass:
            @tool_config(return_direct=True)
            def sample_method(self, x: int) -> int:
                return x * 2

        instance = SampleClass()
        assert hasattr(instance.sample_method, "tool_config")
        assert instance.sample_method.tool_config.return_direct is True
        assert instance.sample_method(5) == 10

    def test_tool_config_on_class(self):
        """Test tool_config decorator on a class."""

        @tool_config(return_direct=True)
        class SampleCallable:
            def __call__(self, x: int) -> int:
                return x * 2

        # Decorated class is still a class (creates subclass)
        assert isinstance(SampleCallable, type)
        # Instances have tool_config
        instance = SampleCallable()
        assert hasattr(instance, "tool_config")
        assert instance.tool_config.return_direct is True
        assert instance(5) == 10

    def test_tool_config_default_values(self):
        """Test that default values are set correctly."""

        @tool_config()
        def sample_function():
            pass

        config = sample_function.tool_config
        assert config.return_direct is False
        assert config.detached is False
        assert config.background is False
        assert config.handoff is False
        assert config.call_as_response is False
        assert config.disable_input is False
        assert config.defer_loading is False
        assert config.runtime_inputs.bindings == ()
        assert config.description is None
        assert config.display_name is None
        assert config.usage_guidance is None
        assert config.feedback is None

    def test_tool_config_accepts_custom_feedback(self):
        @tool_config(feedback="approval")
        def sample_function():
            pass

        assert sample_function.tool_config.feedback.name == "approval"

    @pytest.mark.parametrize(
        "legacy",
        [
            {"return_direct": True},
            {"handoff": True},
            {"call_as_response": True},
        ],
    )
    def test_custom_feedback_rejects_legacy_feedback_aliases(self, legacy):
        with pytest.raises(ValueError, match="cannot be combined"):

            @tool_config(feedback="approval", **legacy)
            def sample_function():
                pass

    def test_tool_config_display_name_and_usage_guidance(self):
        """Test display_name and usage_guidance metadata."""

        @tool_config(
            display_name="Customer Lookup",
            usage_guidance="Use when you need customer profile data.",
        )
        def sample_function():
            pass

        assert sample_function.tool_config.display_name == "Customer Lookup"
        assert (
            sample_function.tool_config.usage_guidance
            == "Use when you need customer profile data."
        )

    def test_tool_config_description_overrides_docstring(self):
        @tool_config(description="Search the product catalog by SKU.")
        def search_products(query: str) -> str:
            return query

        library = ToolLibrary(name="lib", tools=[search_products])

        assert library.library["search_products"].description == (
            "Search the product catalog by SKU."
        )

    def test_tool_config_call_as_response_sets_return_direct(self):
        """Test that call_as_response=True automatically sets return_direct=True."""

        @tool_config(call_as_response=True, return_direct=False)
        def sample_function():
            pass

        assert sample_function.tool_config.call_as_response is True
        assert sample_function.tool_config.return_direct is True

    def test_tool_config_handoff_sets_return_direct_and_messages_input(self):
        """Test that handoff=True adds the messages runtime input."""

        @tool_config(handoff=True)
        def sample_function():
            pass

        assert sample_function.tool_config.handoff is True
        assert sample_function.tool_config.return_direct is True
        assert [
            binding.source
            for binding in sample_function.tool_config.runtime_inputs.bindings
        ] == ["messages"]

    def test_tool_config_spawn_incompatible_with_return_direct(self):
        """Test that detached=True is incompatible with return_direct=True."""
        with pytest.raises(ValueError, match="`detached=True` is not compatible"):

            @tool_config(detached=True, return_direct=True)
            def sample_function():
                pass

    def test_tool_config_spawn_incompatible_with_call_as_response(self):
        """Test that detached=True is incompatible with call_as_response=True."""
        with pytest.raises(ValueError, match="`detached=True` is not compatible"):

            @tool_config(detached=True, call_as_response=True)
            def sample_function():
                pass

    def test_tool_config_background_incompatible_with_spawn(self):
        """Test that background=True is incompatible with detached=True."""
        with pytest.raises(ValueError, match="`background=True` is not compatible"):

            @tool_config(background=True, detached=True)
            def sample_function():
                pass

    def test_runtime_inputs_incompatible_with_call_as_response(self):
        with pytest.raises(ValueError, match="`runtime_inputs` is not compatible"):

            @tool_config(runtime_inputs=["vars"], call_as_response=True)
            def sample_function():
                pass

    def test_runtime_inputs_accept_context_bindings(self):
        binding = ContextBinding(
            source="vars",
            parameter="var1",
            options={"key": "var1"},
        )

        @tool_config(runtime_inputs=[binding])
        def sample_function():
            pass

        assert sample_function.tool_config.runtime_inputs.bindings == (binding,)

    def test_runtime_inputs_accept_source_names(self):

        @tool_config(runtime_inputs=["vars", "handle"])
        def sample_function():
            pass

        assert [
            binding.source
            for binding in sample_function.tool_config.runtime_inputs.bindings
        ] == ["vars", "handle"]

    def test_tool_config_defer_loading_true(self):
        """Test that defer_loading=True is stored correctly."""

        @tool_config(defer_loading=True)
        def sample_function():
            pass

        assert sample_function.tool_config.defer_loading is True

    def test_tool_config_name_override(self):
        """Test that name_override changes the function name."""

        @tool_config(name_override="custom_name")
        def original_name():
            pass

        assert original_name.tool_config.name_overridden == "custom_name"

    def test_tool_config_preserves_function_behavior(self):
        """Test that decorator preserves original function behavior."""

        @tool_config(return_direct=True)
        def add(a: int, b: int) -> int:
            return a + b

        assert add(2, 3) == 5
        assert add(10, 20) == 30

    def test_tool_config_preserves_function_metadata(self):
        """Test that decorator preserves function metadata."""

        @tool_config()
        def documented_function(x: int) -> int:
            """This is a documented function."""
            return x

        assert documented_function.__doc__ == "This is a documented function."


class TestDecorateFunction:
    """Test suite for decorate_function helper."""

    def test_decorate_function_adds_tool_config(self):
        """Test that decorate_function adds tool_config to function."""

        def sample_function():
            return "result"

        config = {"tool_config": dotdict({"return_direct": True, "detached": False})}

        decorated = decorate_function(sample_function, config)
        assert hasattr(decorated, "tool_config")
        assert decorated.tool_config.return_direct is True
        assert decorated() == "result"

    def test_decorate_function_preserves_functionality(self):
        """Test that decorate_function preserves original functionality."""

        def multiply(x: int, y: int) -> int:
            return x * y

        config = {"tool_config": dotdict({"return_direct": False, "detached": False})}

        decorated = decorate_function(multiply, config)
        assert decorated(3, 4) == 12
        assert decorated(5, 6) == 30


class TestDecorateInstance:
    """Test suite for decorate_instance helper."""

    def test_decorate_instance_adds_tool_config(self):
        """Test that decorate_instance adds tool_config to instance."""

        class SampleCallable:
            def __call__(self):
                return "result"

        instance = SampleCallable()
        config = {"tool_config": dotdict({"return_direct": True, "detached": False})}

        decorated = decorate_instance(instance, config)
        assert hasattr(decorated, "tool_config")
        assert decorated.tool_config.return_direct is True
        assert decorated() == "result"

    def test_decorate_instance_preserves_functionality(self):
        """Test that decorate_instance preserves original functionality."""

        class Multiplier:
            def __call__(self, x: int, y: int) -> int:
                return x * y

        instance = Multiplier()
        config = {"tool_config": dotdict({"return_direct": False, "detached": False})}

        decorated = decorate_instance(instance, config)
        assert decorated(3, 4) == 12
        assert decorated(5, 6) == 30


class TestToolConfigCombinations:
    """Test various combinations of tool_config parameters."""

    def test_return_direct_true(self):
        """Test return_direct=True configuration."""

        @tool_config(return_direct=True)
        def sample():
            pass

        assert sample.tool_config.return_direct is True

    def test_spawn_true(self):
        """Test detached=True configuration."""

        @tool_config(detached=True)
        def sample():
            pass

        assert sample.tool_config.detached is True
        assert sample.tool_config.return_direct is False

    def test_messages_runtime_input(self):

        @tool_config(runtime_inputs=["messages"])
        def sample():
            pass

        assert sample.tool_config.runtime_inputs.bindings[0].source == "messages"

    def test_message_runtime_input(self):

        @tool_config(runtime_inputs=["message"])
        def sample():
            pass

        assert sample.tool_config.runtime_inputs.bindings[0].source == "message"

    def test_handle_runtime_input(self):

        @tool_config(runtime_inputs=["handle"])
        def sample():
            pass

        assert sample.tool_config.runtime_inputs.bindings[0].source == "handle"

    def test_disable_input_true(self):
        """Test disable_input=True configuration."""

        @tool_config(disable_input=True)
        def sample():
            pass

        assert sample.tool_config.disable_input is True

    def test_multiple_parameters(self):
        """Test multiple parameters set simultaneously."""

        @tool_config(
            return_direct=True,
            disable_input=True,
            runtime_inputs=["vars", "messages", "handle"],
        )
        def sample():
            pass

        config = sample.tool_config
        assert config.return_direct is True
        assert config.disable_input is True
        assert [binding.source for binding in config.runtime_inputs.bindings] == [
            "vars",
            "messages",
            "handle",
        ]

    def test_all_false_parameters(self):
        """Test all parameters set to False."""

        @tool_config(
            return_direct=False,
            detached=False,
            handoff=False,
            call_as_response=False,
            disable_input=False,
            defer_loading=False,
        )
        def sample():
            pass

        config = sample.tool_config
        assert config.return_direct is False
        assert config.detached is False
        assert config.handoff is False
        assert config.call_as_response is False
        assert config.disable_input is False
        assert config.defer_loading is False
        assert config.runtime_inputs.bindings == ()


class TestToolConfigEdgeCases:
    """Test edge cases and special scenarios."""

    def test_nested_decorators(self):
        """Test that tool_config works with nested decorators."""

        def other_decorator(func):
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            return wrapper

        @other_decorator
        @tool_config(return_direct=True)
        def sample():
            return "result"

        # The tool_config should be on the inner function
        # but wrapped by outer decorator
        assert sample() == "result"

    def test_lambda_function(self):
        """Test tool_config on lambda functions."""
        decorated_lambda = tool_config(return_direct=True)(lambda x: x * 2)
        assert decorated_lambda.tool_config.return_direct is True
        assert decorated_lambda(5) == 10

    def test_class_with_init_parameters(self):
        """Test tool_config on class that requires init parameters."""

        @tool_config(return_direct=True)
        class ParameterizedCallable:
            def __init__(self):
                self.multiplier = 2

            def __call__(self, x: int) -> int:
                return x * self.multiplier

        # Decorated class is still a class (creates subclass)
        assert isinstance(ParameterizedCallable, type)
        # Instances have tool_config
        instance = ParameterizedCallable()
        assert hasattr(instance, "tool_config")
        assert instance.tool_config.return_direct is True
        assert instance(5) == 10

    def test_empty_decorator(self):
        """Test tool_config with no parameters."""

        @tool_config()
        def sample():
            return "result"

        assert hasattr(sample, "tool_config")
        assert sample() == "result"

    def test_function_with_defaults(self):
        """Test tool_config on function with default arguments."""

        @tool_config(return_direct=True)
        def sample(x: int = 10, y: int = 20) -> int:
            return x + y

        assert sample() == 30
        assert sample(5) == 25
        assert sample(5, 10) == 15

    def test_function_with_kwargs(self):
        """Test tool_config on function with **kwargs."""

        @tool_config(runtime_inputs=["vars"])
        def sample(**kwargs):
            return kwargs

        result = sample(a=1, b=2)
        assert result == {"a": 1, "b": 2}
        assert sample.tool_config.runtime_inputs.bindings[0].source == "vars"

    def test_function_with_args_and_kwargs(self):
        """Test tool_config on function with *args and **kwargs."""

        @tool_config(return_direct=True)
        def sample(*args, **kwargs):
            return args, kwargs

        args, kwargs = sample(1, 2, 3, a=4, b=5)
        assert args == (1, 2, 3)
        assert kwargs == {"a": 4, "b": 5}

    def test_custom_dispatch_is_preserved(self):
        @tool_config(dispatch="queue")
        def sample() -> str:
            return "ok"

        assert sample.tool_config.dispatch == "queue"

    @pytest.mark.parametrize("option", ["background", "allow_background", "detached"])
    def test_custom_dispatch_rejects_convenience_dispatch_options(self, option):
        with pytest.raises(ValueError, match="cannot be combined"):

            @tool_config(dispatch="queue", **{option: True})
            def sample() -> str:
                return "ok"
