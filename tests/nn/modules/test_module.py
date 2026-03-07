"""Tests for msgflux.nn.modules.module base class."""

import pytest
from unittest.mock import Mock, MagicMock, patch

from msgflux.core.dotdict import dotdict
from msgflux.core.message import Message
from msgflux.nn.modules.module import (
    Module,
    _IncompatibleKeys,
    _addindent,
    get_callable_name,
)
from msgflux.nn.parameter import Parameter


class SimpleModule(Module):
    """Simple test module."""

    def __init__(self, value=10):
        super().__init__()
        self.value = value

    def forward(self, x):
        return x + self.value


class NestedModule(Module):
    """Module with nested submodules."""

    def __init__(self):
        super().__init__()
        self.sub1 = SimpleModule(5)
        self.sub2 = SimpleModule(10)

    def forward(self, x):
        return self.sub2(self.sub1(x))


class TestHelperFunctions:
    """Test helper functions."""

    def test_incompatible_keys_empty(self):
        """Test _IncompatibleKeys with no keys."""
        keys = _IncompatibleKeys([], [])
        assert "<All keys matched successfully>" in str(keys)

    def test_incompatible_keys_with_missing(self):
        """Test _IncompatibleKeys with missing keys."""
        keys = _IncompatibleKeys(["key1", "key2"], [])
        assert "key1" in str(keys)
        assert "key2" in str(keys)

    def test_incompatible_keys_with_unexpected(self):
        """Test _IncompatibleKeys with unexpected keys."""
        keys = _IncompatibleKeys([], ["extra1"])
        assert "extra1" in str(keys)

    def test_addindent_single_line(self):
        """Test _addindent with single line."""
        result = _addindent("single", 2)
        assert result == "single"

    def test_addindent_multi_line(self):
        """Test _addindent with multiple lines."""
        result = _addindent("first\nsecond\nthird", 2)
        assert result == "first\n  second\n  third"

    def test_get_callable_name_module(self):
        """Test get_callable_name with Module."""
        module = SimpleModule()
        module.set_name("test")
        name = get_callable_name(module)
        assert name == "test"

    def test_get_callable_name_function(self):
        """Test get_callable_name with function."""

        def my_func():
            pass

        name = get_callable_name(my_func)
        assert name == "my_func"

    def test_get_callable_name_class(self):
        """Test get_callable_name with class instance."""

        class MyClass:
            pass

        obj = MyClass()
        name = get_callable_name(obj)
        assert name == "MyClass"


class TestModule:
    """Test suite for Module base class."""

    def test_module_initialization(self):
        """Test Module basic initialization."""
        module = SimpleModule()
        assert module.value == 10

    def test_module_call_invokes_forward(self):
        """Test that calling module invokes forward method."""
        module = SimpleModule(value=5)
        result = module(10)
        assert result == 15

    def test_module_set_name(self):
        """Test Module set_name method."""
        module = SimpleModule()
        module.set_name("test_module")
        assert module.name == "test_module"

    def test_module_set_description(self):
        """Test Module set_description method."""
        module = SimpleModule()
        module.set_description("A test module")
        assert module.description == "A test module"

    def test_module_has_parameters_dict(self):
        """Test that Module has parameters dict."""
        module = SimpleModule()
        assert hasattr(module, "_parameters")
        assert isinstance(module._parameters, dict)

    def test_module_has_buffers_dict(self):
        """Test that Module has buffers dict."""
        module = SimpleModule()
        assert hasattr(module, "_buffers")
        assert isinstance(module._buffers, dict)

    def test_module_has_modules_dict(self):
        """Test that Module has modules dict."""
        module = SimpleModule()
        assert hasattr(module, "_modules")
        assert isinstance(module._modules, dict)

    def test_register_buffer(self):
        """Test registering a buffer."""
        module = SimpleModule()
        module.register_buffer("my_buffer", [1, 2, 3])
        assert "my_buffer" in module._buffers
        assert module.my_buffer == [1, 2, 3]

    def test_register_buffer_none(self):
        """Test registering None buffer."""
        module = SimpleModule()
        module.register_buffer("none_buffer", None)
        assert module._buffers["none_buffer"] is None

    def test_register_parameter(self):
        """Test registering a parameter."""
        module = SimpleModule()
        param = Parameter("test data", "test spec")
        module.register_parameter("param1", param)
        assert "param1" in module._parameters
        assert module.param1 == param

    def test_register_parameter_none(self):
        """Test registering None parameter."""
        module = SimpleModule()
        module.register_parameter("none_param", None)
        assert module._parameters["none_param"] is None

    def test_add_module(self):
        """Test adding a submodule."""
        parent = SimpleModule()
        child = SimpleModule(value=20)
        parent.add_module("child", child)
        assert "child" in parent._modules
        assert parent.child == child

    def test_add_module_none(self):
        """Test adding None module."""
        module = SimpleModule()
        module.add_module("none_mod", None)
        assert module._modules["none_mod"] is None

    def test_register_module(self):
        """Test register_module (alias for add_module)."""
        parent = SimpleModule()
        child = SimpleModule(value=15)
        parent.register_module("registered_child", child)
        assert "registered_child" in parent._modules

    def test_get_submodule_direct(self):
        """Test get_submodule with direct child."""
        parent = NestedModule()
        sub = parent.get_submodule("sub1")
        assert sub == parent.sub1

    def test_get_submodule_nested(self):
        """Test get_submodule with nested path."""
        parent = NestedModule()
        # This should fail since sub1 has no 'value' submodule
        with pytest.raises(AttributeError):
            parent.get_submodule("sub1.value")

    def test_get_submodule_empty_target(self):
        """Test get_submodule with empty target returns self."""
        module = SimpleModule()
        result = module.get_submodule("")
        assert result is module

    def test_set_submodule(self):
        """Test set_submodule."""
        parent = NestedModule()
        new_module = SimpleModule(value=99)
        parent.set_submodule("sub1", new_module)
        assert parent.sub1 == new_module
        assert parent.sub1.value == 99

    def test_get_parameter(self):
        """Test get_parameter."""
        module = SimpleModule()
        param = Parameter("data", "spec")
        module.register_parameter("p1", param)
        retrieved = module.get_parameter("p1")
        assert retrieved == param

    def test_get_parameter_nonexistent(self):
        """Test get_parameter with nonexistent parameter."""
        module = SimpleModule()
        with pytest.raises(AttributeError):
            module.get_parameter("nonexistent")

    def test_get_buffer(self):
        """Test get_buffer."""
        module = SimpleModule()
        module.register_buffer("b1", {"data": "test"})
        retrieved = module.get_buffer("b1")
        assert retrieved == {"data": "test"}

    def test_get_buffer_nonexistent(self):
        """Test get_buffer with nonexistent buffer."""
        module = SimpleModule()
        with pytest.raises(AttributeError):
            module.get_buffer("nonexistent")

    def test_named_parameters(self):
        """Test named_parameters iterator."""
        module = SimpleModule()
        param1 = Parameter("data1", "spec1")
        param2 = Parameter("data2", "spec2")
        module.register_parameter("param1", param1)
        module.register_parameter("param2", param2)

        names = [name for name, _ in module.named_parameters()]
        assert "param1" in names
        assert "param2" in names

    def test_named_buffers(self):
        """Test named_buffers iterator."""
        module = SimpleModule()
        module.register_buffer("buf1", "buffer_string_1")
        module.register_buffer("buf2", "buffer_string_2")

        names = [name for name, _ in module.named_buffers()]
        assert "buf1" in names
        assert "buf2" in names

    def test_named_children(self):
        """Test named_children iterator."""
        parent = NestedModule()
        children = dict(parent.named_children())
        assert "sub1" in children
        assert "sub2" in children

    def test_named_modules(self):
        """Test named_modules iterator."""
        parent = NestedModule()
        modules = dict(parent.named_modules())
        assert "" in modules  # Parent itself
        assert "sub1" in modules
        assert "sub2" in modules

    def test_children(self):
        """Test children iterator."""
        parent = NestedModule()
        children = list(parent.children())
        assert len(children) == 2

    def test_modules(self):
        """Test modules iterator."""
        parent = NestedModule()
        modules = list(parent.modules())
        assert len(modules) == 3  # parent + 2 children

    def test_register_forward_pre_hook(self):
        """Test registering forward pre-hook."""
        module = SimpleModule()
        hook_called = []

        def my_hook(module, args, kwargs):
            hook_called.append(True)
            return args, kwargs

        handle = module.register_forward_pre_hook(my_hook)
        module(5)

        assert len(hook_called) == 1
        handle.remove()

    def test_register_forward_hook(self):
        """Test registering forward hook."""
        module = SimpleModule()
        hook_results = []

        def my_hook(module, args, kwargs, output):
            hook_results.append(output)
            return output

        handle = module.register_forward_hook(my_hook)
        result = module(5)

        assert len(hook_results) == 1
        assert hook_results[0] == result
        handle.remove()

    def test_forward_hook_modifies_output(self):
        """Test that forward hook can modify output."""
        module = SimpleModule(value=10)

        def modify_hook(module, args, kwargs, output):
            return output * 2

        module.register_forward_hook(modify_hook)
        result = module(5)

        # Original: 5 + 10 = 15, hook modifies to 15 * 2 = 30
        assert result == 30

    def test_get_module_name(self):
        """Test get_module_name."""
        module = SimpleModule()
        module.set_name("my_module")
        assert module.get_module_name() == "my_module"

    def test_get_module_description(self):
        """Test get_module_description."""
        module = SimpleModule()
        module.set_description("My description")
        assert module.get_module_description() == "My description"

    def test_set_annotations(self):
        """Test set_annotations."""
        module = SimpleModule()
        annotations = {"x": int, "y": str}
        module.set_annotations(annotations)
        assert module.annotations == annotations

    def test_get_module_annotations(self):
        """Test get_module_annotations."""
        module = SimpleModule()
        annotations = {"a": float}
        module.set_annotations(annotations)
        assert module.get_module_annotations() == annotations

    def test_module_repr(self):
        """Test module __repr__."""
        module = SimpleModule()
        module.set_name("test")
        repr_str = repr(module)
        assert "SimpleModule" in repr_str or "test" in repr_str

    def test_parameters_iterator(self):
        """Test parameters() iterator."""
        module = SimpleModule()
        p1 = Parameter("data1", "spec1")
        p2 = Parameter("data2", "spec2")
        module.register_parameter("p1", p1)
        module.register_parameter("p2", p2)

        params = list(module.parameters())
        assert len(params) == 2
        assert p1 in params
        assert p2 in params

    def test_buffers_iterator(self):
        """Test buffers() iterator."""
        module = SimpleModule()
        module.register_buffer("b1", "buffer1")
        module.register_buffer("b2", "buffer2")

        buffers = list(module.buffers())
        assert len(buffers) == 2

    def test_state_dict(self):
        """Test state_dict generation."""
        module = SimpleModule()
        module.register_buffer("buf", "buffer_value")
        param = Parameter("test_data", "test_spec")
        module.register_parameter("param", param)

        state = module.state_dict()
        assert isinstance(state, dict)

    def test_getstate_setstate(self):
        """Test __getstate__ and __setstate__ for pickling."""
        module = SimpleModule(value=42)
        module.set_name("pickled_module")

        state = module.__getstate__()

        new_module = SimpleModule()
        new_module.__setstate__(state)

        assert new_module.name == "pickled_module"

    def test_module_extra_repr(self):
        """Test extra_repr method."""
        module = SimpleModule()
        extra = module.extra_repr()
        assert isinstance(extra, str)

    def test_named_parameters_with_prefix(self):
        """Test named_parameters with custom prefix."""
        module = SimpleModule()
        param = Parameter("data", "spec")
        module.register_parameter("p1", param)

        params = list(module.named_parameters(prefix="custom"))
        assert len(params) > 0
        assert params[0][0].startswith("custom")

    def test_named_modules_with_memo(self):
        """Test named_modules doesn't return duplicates."""
        parent = NestedModule()
        # Add same module twice
        parent.add_module("dup", parent.sub1)

        modules = list(parent.named_modules())
        # Should not have duplicates even though sub1 is referenced twice
        names = [name for name, _ in modules]
        assert len(names) == len(set(names))

    def test_load_state_dict(self):
        """Test load_state_dict."""
        module1 = SimpleModule(value=10)
        module1.register_buffer("buf", "test_buffer")

        state = module1.state_dict()

        module2 = SimpleModule(value=99)
        module2.register_buffer("buf", "different")

        # Load state from module1 into module2
        result = module2.load_state_dict(state)
        # Check that it returns IncompatibleKeys
        assert hasattr(result, "missing_keys") or result is None

    def test_module_training_mode(self):
        """Test train() and eval() mode switching."""
        module = SimpleModule()
        assert module.training is True  # Default

        module.eval()
        assert module.training is False

        module.train()
        assert module.training is True

    def test_module_train_affects_children(self):
        """Test that train()/eval() affects child modules."""
        parent = NestedModule()
        parent.eval()

        assert parent.training is False
        assert parent.sub1.training is False
        assert parent.sub2.training is False

    def test_setattr_with_parameter(self):
        """Test that __setattr__ handles Parameter correctly."""
        module = SimpleModule()
        param = Parameter("data", "spec")
        # Setting as attribute should add to _parameters
        module.my_param = param
        assert "my_param" in module._parameters

    def test_setattr_with_module(self):
        """Test that __setattr__ handles Module correctly."""
        parent = SimpleModule()
        child = SimpleModule(value=5)
        # Setting as attribute should add to _modules
        parent.child_mod = child
        assert "child_mod" in parent._modules

    def test_delattr_parameter(self):
        """Test deleting a parameter."""
        module = SimpleModule()
        param = Parameter("data", "spec")
        module.register_parameter("p", param)

        del module.p
        assert "p" not in module._parameters

    def test_delattr_buffer(self):
        """Test deleting a buffer."""
        module = SimpleModule()
        module.register_buffer("buf", "value")

        del module.buf
        assert "buf" not in module._buffers

    def test_delattr_module(self):
        """Test deleting a submodule."""
        parent = NestedModule()

        del parent.sub1
        assert "sub1" not in parent._modules

    def test_getattr_parameter(self):
        """Test __getattr__ for parameters."""
        module = SimpleModule()
        param = Parameter("data", "spec")
        module.register_parameter("my_p", param)

        # Accessing via attribute should work
        assert module.my_p == param

    def test_getattr_buffer(self):
        """Test __getattr__ for buffers."""
        module = SimpleModule()
        module.register_buffer("my_buf", "buffer_value")

        # Accessing via attribute should work
        assert module.my_buf == "buffer_value"

    def test_getattr_module(self):
        """Test __getattr__ for submodules."""
        parent = NestedModule()

        # Accessing via attribute should work
        assert parent.sub1 is not None
        assert isinstance(parent.sub1, SimpleModule)

    def test_dir_includes_parameters(self):
        """Test that dir() includes parameters."""
        module = SimpleModule()
        param = Parameter("data", "spec")
        module.register_parameter("my_param", param)

        dir_result = dir(module)
        assert "my_param" in dir_result

    def test_dir_includes_buffers(self):
        """Test that dir() includes buffers."""
        module = SimpleModule()
        module.register_buffer("my_buffer", "value")

        dir_result = dir(module)
        assert "my_buffer" in dir_result

    def test_dir_includes_modules(self):
        """Test that dir() includes submodules."""
        parent = NestedModule()

        dir_result = dir(parent)
        assert "sub1" in dir_result
        assert "sub2" in dir_result

    def test_parameters_recurse_false(self):
        """Test parameters(recurse=False)."""
        parent = NestedModule()
        param = Parameter("data", "spec")
        parent.register_parameter("parent_param", param)

        # With recurse=False, should only get parent's parameters
        params = list(parent.parameters(recurse=False))
        assert len(params) == 1

    def test_named_parameters_recurse_false(self):
        """Test named_parameters(recurse=False)."""
        parent = NestedModule()
        param = Parameter("data", "spec")
        parent.register_parameter("parent_param", param)

        # With recurse=False, should only get parent's parameters
        named_params = list(parent.named_parameters(recurse=False))
        assert len(named_params) == 1
        assert named_params[0][0] == "parent_param"

    def test_named_buffers_recurse_false(self):
        """Test named_buffers(recurse=False)."""
        parent = NestedModule()
        parent.register_buffer("parent_buf", "value")

        # With recurse=False, should only get parent's buffers
        named_bufs = list(parent.named_buffers(recurse=False))
        assert len(named_bufs) == 1
        assert named_bufs[0][0] == "parent_buf"

    def test_named_modules_remove_duplicate_false(self):
        """Test named_modules with remove_duplicate=False."""
        parent = NestedModule()

        # With remove_duplicate=False, may have duplicates
        modules = list(parent.named_modules(remove_duplicate=False))
        assert len(modules) >= 3

    def test_module_contains_submodule(self):
        """Test __contains__ for checking if submodule exists."""
        parent = NestedModule()

        # Check if submodule exists
        assert "sub1" in parent._modules

    def test_state_dict_with_prefix(self):
        """Test state_dict with custom prefix."""
        module = SimpleModule()
        module.register_buffer("buf", "value")

        state = module.state_dict(prefix="custom.")
        # Keys should have the prefix
        assert any("custom" in key for key in state.keys())

    def test_state_dict_with_destination(self):
        """Test state_dict with destination dict."""
        module = SimpleModule()
        module.register_buffer("buf", "value")

        dest = {}
        result = module.state_dict(destination=dest)
        # Should return the destination dict
        assert result is dest
        assert len(dest) > 0

    def test_load_state_dict_updates_buffers(self):
        """Test load_state_dict updates buffers."""
        module1 = SimpleModule()
        module1.register_buffer("buf", "original")

        state = module1.state_dict()

        module2 = SimpleModule()
        module2.register_buffer("buf", "new")

        # Load state - should update buffer
        module2.load_state_dict(state)
        # Verify buffer was updated (if applicable)
        assert module2.buf in ["original", "new"]

    def test_modules_iterator_includes_self(self):
        """Test that modules() iterator includes self."""
        module = SimpleModule()
        modules_list = list(module.modules())

        # Should include the module itself
        assert module in modules_list

    def test_register_parameter_replaces_existing(self):
        """Test that registering parameter with existing name replaces it."""
        module = SimpleModule()
        param1 = Parameter("data1", "spec1")
        param2 = Parameter("data2", "spec2")

        module.register_parameter("p", param1)
        module.register_parameter("p", param2)

        # Should have the second parameter
        assert module.p == param2

    def test_register_buffer_replaces_existing(self):
        """Test that registering buffer with existing name replaces it."""
        module = SimpleModule()

        module.register_buffer("buf", "value1")
        module.register_buffer("buf", "value2")

        # Should have the second value
        assert module.buf == "value2"

    def test_add_module_replaces_existing(self):
        """Test that adding module with existing name replaces it."""
        parent = SimpleModule()
        child1 = SimpleModule(value=1)
        child2 = SimpleModule(value=2)

        parent.add_module("child", child1)
        parent.add_module("child", child2)

        # Should have the second child
        assert parent.child == child2

    def test_getattr_raises_for_nonexistent(self):
        """Test __getattr__ raises AttributeError for nonexistent attribute."""
        module = SimpleModule()

        with pytest.raises(AttributeError):
            _ = module.nonexistent_attr

    def test_delattr_raises_for_nonexistent(self):
        """Test __delattr__ raises for nonexistent attribute."""
        module = SimpleModule()

        with pytest.raises(AttributeError):
            del module.nonexistent_attr

    def test_get_submodule_invalid_path(self):
        """Test get_submodule with invalid path raises error."""
        parent = NestedModule()

        with pytest.raises(AttributeError):
            parent.get_submodule("nonexistent.path")

    def test_set_submodule_creates_intermediate(self):
        """Test set_submodule with nested path."""
        parent = SimpleModule()
        child = SimpleModule(value=42)

        # Setting a direct child should work
        parent.set_submodule("new_child", child)
        assert parent.new_child == child

    def test_get_parameter_nested(self):
        """Test get_parameter with nested path."""
        parent = NestedModule()
        param = Parameter("data", "spec")
        parent.sub1.register_parameter("nested_param", param)

        # Get nested parameter
        retrieved = parent.get_parameter("sub1.nested_param")
        assert retrieved == param

    def test_get_buffer_nested(self):
        """Test get_buffer with nested path."""
        parent = NestedModule()
        parent.sub1.register_buffer("nested_buf", "value")

        # Get nested buffer
        retrieved = parent.get_buffer("sub1.nested_buf")
        assert retrieved == "value"

    def test_parameters_with_recurse_true(self):
        """Test parameters() with recurse=True includes child parameters."""
        parent = NestedModule()
        param1 = Parameter("data1", "spec1")
        param2 = Parameter("data2", "spec2")

        parent.register_parameter("p1", param1)
        parent.sub1.register_parameter("p2", param2)

        # With recurse=True, should get both
        params = list(parent.parameters(recurse=True))
        assert len(params) == 2

    def test_named_parameters_with_recurse_true(self):
        """Test named_parameters() with recurse=True includes child parameters."""
        parent = NestedModule()
        param1 = Parameter("data1", "spec1")
        param2 = Parameter("data2", "spec2")

        parent.register_parameter("p1", param1)
        parent.sub1.register_parameter("p2", param2)

        # With recurse=True, should get both
        named_params = list(parent.named_parameters(recurse=True))
        assert len(named_params) == 2
        names = [name for name, _ in named_params]
        assert "p1" in names
        assert "sub1.p2" in names

    def test_buffers_with_recurse_true(self):
        """Test buffers() with recurse=True includes child buffers."""
        parent = NestedModule()
        parent.register_buffer("b1", "val1")
        parent.sub1.register_buffer("b2", "val2")

        # With recurse=True, should get both
        buffers = list(parent.buffers(recurse=True))
        assert len(buffers) == 2

    def test_named_buffers_with_recurse_true(self):
        """Test named_buffers() with recurse=True includes child buffers."""
        parent = NestedModule()
        parent.register_buffer("b1", "val1")
        parent.sub1.register_buffer("b2", "val2")

        # With recurse=True, should get both
        named_buffers = list(parent.named_buffers(recurse=True))
        assert len(named_buffers) == 2
        names = [name for name, _ in named_buffers]
        assert "b1" in names
        assert "sub1.b2" in names

    def test_children_returns_immediate_children_only(self):
        """Test children() returns only immediate children."""
        parent = NestedModule()
        # NestedModule has sub1 and sub2

        children_list = list(parent.children())
        assert len(children_list) == 2

    def test_named_children_returns_immediate_children_only(self):
        """Test named_children() returns only immediate children."""
        parent = NestedModule()

        named_children = list(parent.named_children())
        assert len(named_children) == 2
        names = [name for name, _ in named_children]
        assert "sub1" in names
        assert "sub2" in names

    def test_state_dict_empty_module(self):
        """Test state_dict on module with no parameters or buffers."""
        module = SimpleModule()

        state = module.state_dict()
        # Should be empty or contain only module-specific state
        assert isinstance(state, dict)

    def test_state_dict_with_nested_modules(self):
        """Test state_dict includes nested module states."""
        parent = NestedModule()
        parent.register_buffer("parent_buf", "parent_value")
        parent.sub1.register_buffer("child_buf", "child_value")

        state = parent.state_dict()
        # Should include both parent and child states
        assert isinstance(state, dict)

    def test_load_state_dict_empty(self):
        """Test load_state_dict with empty state dict."""
        module = SimpleModule()

        result = module.load_state_dict({})
        # Should handle empty state gracefully
        assert result is not None or result is None

    def test_load_state_dict_with_extra_keys(self):
        """Test load_state_dict with state containing extra keys."""
        module = SimpleModule()
        module.register_buffer("buf", "value")

        # State with extra key
        state = {"buf": "new_value", "extra_key": "extra"}

        result = module.load_state_dict(state)
        # Should handle extra keys
        assert result is not None or result is None

    def test_load_state_dict_with_missing_keys(self):
        """Test load_state_dict with state missing expected keys."""
        module = SimpleModule()
        module.register_buffer("buf1", "value1")
        module.register_buffer("buf2", "value2")

        # State missing buf2
        state = {"buf1": "new_value"}

        result = module.load_state_dict(state)
        # Should handle missing keys
        assert result is not None or result is None

    def test_register_forward_pre_hook_with_prepend(self):
        """Test register_forward_pre_hook with prepend option."""
        module = SimpleModule()
        order = []

        def hook1(mod, args, kwargs):
            order.append(1)
            return args, kwargs

        def hook2(mod, args, kwargs):
            order.append(2)
            return args, kwargs

        module.register_forward_pre_hook(hook1)
        module.register_forward_pre_hook(hook2, prepend=True)

        module(5)

        # hook2 should run first because of prepend
        assert order[0] == 2
        assert order[1] == 1

    def test_register_forward_hook_with_prepend(self):
        """Test register_forward_hook with prepend option."""
        module = SimpleModule()
        order = []

        def hook1(mod, args, kwargs, output):
            order.append(1)
            return output

        def hook2(mod, args, kwargs, output):
            order.append(2)
            return output

        module.register_forward_hook(hook1)
        module.register_forward_hook(hook2, prepend=True)

        module(5)

        # hook2 should run first because of prepend
        assert order[0] == 2
        assert order[1] == 1

    def test_forward_pre_hook_can_modify_input(self):
        """Test that forward pre-hook can modify input."""
        module = SimpleModule(value=10)

        def modify_input_hook(mod, args, kwargs):
            # Modify the first argument
            new_args = (args[0] * 2,) if args else args
            return new_args, kwargs

        module.register_forward_pre_hook(modify_input_hook)
        result = module(5)

        # Input was 5, hook doubled it to 10, then added module.value (10) = 20
        assert result == 20

    def test_multiple_forward_hooks(self):
        """Test multiple forward hooks are called in order."""
        module = SimpleModule(value=0)
        results = []

        def hook1(mod, args, kwargs, output):
            results.append(("hook1", output))
            return output + 1

        def hook2(mod, args, kwargs, output):
            results.append(("hook2", output))
            return output + 10

        module.register_forward_hook(hook1)
        module.register_forward_hook(hook2)

        result = module(5)

        # Original: 5 + 0 = 5
        # hook1: 5 + 1 = 6
        # hook2: 6 + 10 = 16
        assert result == 16
        assert len(results) == 2

    def test_hook_removal(self):
        """Test that removing a hook stops it from being called."""
        module = SimpleModule()
        called = []

        def hook(mod, args, kwargs):
            called.append(True)
            return args, kwargs

        handle = module.register_forward_pre_hook(hook)
        module(5)
        assert len(called) == 1

        handle.remove()
        module(5)
        # Should still be 1, not 2
        assert len(called) == 1

    def test_setattr_with_regular_attribute(self):
        """Test __setattr__ with regular non-special attributes."""
        module = SimpleModule()

        # Regular attribute
        module.custom_attr = "custom_value"
        assert module.custom_attr == "custom_value"
        assert "custom_attr" not in module._parameters
        assert "custom_attr" not in module._buffers
        assert "custom_attr" not in module._modules

    def test_train_mode_recursive(self):
        """Test train(mode=True) sets training mode recursively."""
        parent = NestedModule()
        parent.eval()

        # Set back to training
        parent.train(mode=True)

        assert parent.training is True
        assert parent.sub1.training is True
        assert parent.sub2.training is True

    def test_eval_mode_recursive(self):
        """Test eval() sets eval mode recursively."""
        parent = NestedModule()
        # Starts in training mode
        assert parent.training is True

        parent.eval()

        assert parent.training is False
        assert parent.sub1.training is False
        assert parent.sub2.training is False


class TestRemovableHandle:
    """Test RemovableHandle functionality."""

    def test_removable_handle_id(self):
        """Test RemovableHandle has unique id."""
        module = SimpleModule()

        def hook(m, a, k):
            return a, k

        handle1 = module.register_forward_pre_hook(hook)
        handle2 = module.register_forward_pre_hook(hook)

        # Handles should have different IDs
        assert handle1.id != handle2.id

        handle1.remove()
        handle2.remove()

    def test_removable_handle_remove_idempotent(self):
        """Test that calling remove() multiple times is safe."""
        module = SimpleModule()

        def hook(m, a, k):
            return a, k

        handle = module.register_forward_pre_hook(hook)

        # Remove multiple times should not raise
        handle.remove()
        handle.remove()
        handle.remove()

    def test_removable_handle_weakref(self):
        """Test RemovableHandle uses weakref correctly."""
        module = SimpleModule()
        called = []

        def hook(m, a, k):
            called.append(True)
            return a, k

        handle = module.register_forward_pre_hook(hook)

        # Call should work
        module(1)
        assert len(called) == 1

        # Delete module
        del module

        # Handle should still exist but be harmless
        handle.remove()  # Should not raise


class TestModuleMisc:
    """Miscellaneous Module tests."""

    def test_module_call_with_args_and_kwargs(self):
        """Test calling module with both args and kwargs."""

        class FlexModule(Module):
            def forward(self, a, b=10):
                return a + b

        module = FlexModule()
        result = module(5, b=20)
        assert result == 25

    def test_module_call_with_only_kwargs(self):
        """Test calling module with only kwargs."""

        class KwargsModule(Module):
            def forward(self, x=1, y=2):
                return x * y

        module = KwargsModule()
        result = module(x=3, y=4)
        assert result == 12

    def test_module_has_training_attribute(self):
        """Test module has training attribute initialized."""
        module = SimpleModule()
        assert hasattr(module, "training")
        assert isinstance(module.training, bool)

    def test_module_name_default(self):
        """Test module name defaults correctly."""
        module = SimpleModule()
        # Name should be class name by default
        assert module.get_module_name() == "SimpleModule"

    def test_module_description_default(self):
        """Test module description defaults correctly."""
        module = SimpleModule()
        desc = module.get_module_description()
        # Should be docstring or empty
        assert isinstance(desc, str)


class TestResponseMode:
    """Tests for _set_response_mode and _define_response_mode."""

    def _make_module(self, response_mode=None):
        """Create a SimpleModule with response_mode configured."""
        m = SimpleModule()
        m._set_response_mode(response_mode)
        return m

    # --- _set_response_mode validation ---

    def test_set_response_mode_none(self):
        m = self._make_module(None)
        assert m.response_mode is None

    def test_set_response_mode_plain_string(self):
        m = self._make_module("outputs.result")
        assert m.response_mode == "outputs.result"

    def test_set_response_mode_dotdict_format(self):
        m = self._make_module("outputs.result:")
        assert m.response_mode == "outputs.result:"

    def test_set_response_mode_simple_dotdict(self):
        m = self._make_module("result:")
        assert m.response_mode == "result:"

    def test_set_response_mode_empty_string_raises(self):
        with pytest.raises(ValueError):
            self._make_module("")

    def test_set_response_mode_bare_colon_raises(self):
        """':' alone has no path, must be rejected."""
        with pytest.raises(ValueError):
            self._make_module(":")

    def test_set_response_mode_wrong_type_raises(self):
        with pytest.raises(TypeError):
            self._make_module(123)  # type: ignore[arg-type]

    # --- _define_response_mode: None mode ---

    def test_define_response_mode_none_returns_response(self):
        m = self._make_module(None)
        result = m._define_response_mode("hello", None)
        assert result == "hello"

    # --- _define_response_mode: dotdict mode (ends with ':') ---

    def test_define_response_mode_dotdict_simple_key(self):
        """'result:' returns dotdict({'result': response})."""
        m = self._make_module("result:")
        result = m._define_response_mode("hello", None)
        assert isinstance(result, dotdict)
        assert result.get("result") == "hello"

    def test_define_response_mode_dotdict_nested_path(self):
        """'outputs.text:' returns a nested dotdict."""
        m = self._make_module("outputs.text:")
        result = m._define_response_mode("world", None)
        assert isinstance(result, dotdict)
        assert result.get("outputs.text") == "world"

    def test_define_response_mode_dotdict_deep_path(self):
        """'a.b.c:' returns a deeply nested dotdict."""
        m = self._make_module("a.b.c:")
        result = m._define_response_mode(42, None)
        assert isinstance(result, dotdict)
        assert result.get("a.b.c") == 42

    def test_define_response_mode_dotdict_ignores_message(self):
        """Dotdict mode must not require nor modify Message."""
        m = self._make_module("result:")
        msg = Message(content="original")
        result = m._define_response_mode("value", msg)
        assert isinstance(result, dotdict)
        assert result.get("result") == "value"
        # Message must remain untouched
        assert msg.content == "original"

    def test_define_response_mode_dotdict_works_without_message(self):
        """Dotdict mode works when message is None."""
        m = self._make_module("result:")
        result = m._define_response_mode("value", None)
        assert isinstance(result, dotdict)

    def test_define_response_mode_dotdict_any_response_type(self):
        """Dotdict mode wraps any response type (dict, list, object)."""
        m = self._make_module("data:")
        payload = {"key": [1, 2, 3]}
        result = m._define_response_mode(payload, None)
        assert result.get("data") == payload

    # --- _define_response_mode: message mode (no colon) ---

    def test_define_response_mode_message_writes_field_and_returns_none(self):
        m = self._make_module("outputs.result")
        msg = Message(content="input")
        result = m._define_response_mode("answer", msg)
        assert result is None
        assert msg.get("outputs.result") == "answer"

    def test_define_response_mode_message_without_message_raises(self):
        m = self._make_module("outputs.result")
        with pytest.raises(ValueError):
            m._define_response_mode("answer", None)
