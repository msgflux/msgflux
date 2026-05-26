"""Tests for Guard class and guard integration."""

import pytest

from msgflux.exceptions import UnsafeModelResponseError, UnsafeUserInputError
from msgflux.exceptions import _GuardInterrupt
from msgflux.nn.hooks import Guard


# ---------------------------------------------------------------------------
# Guard instantiation
# ---------------------------------------------------------------------------


class TestGuardInit:
    def test_valid_guard(self):
        guard = Guard(validator=lambda data: {"safe": True}, on="pre")
        assert guard.on == "pre"
        assert guard.message is None

    def test_with_message(self):
        guard = Guard(
            validator=lambda data: {"safe": True},
            on="pre",
            message="Blocked.",
        )
        assert guard.message == "Blocked."

    def test_invalid_on(self):
        with pytest.raises(ValueError, match="`on` must be one of"):
            Guard(validator=lambda data: {"safe": True}, on="middle")

    def test_non_callable_validator(self):
        with pytest.raises(TypeError, match="`validator` must be callable"):
            Guard(validator="not_callable", on="pre")


# ---------------------------------------------------------------------------
# Guard hooks
# ---------------------------------------------------------------------------


class TestGuardHooks:
    def test_pre_call_safe(self):
        guard = Guard(validator=lambda data: {"safe": True}, on="pre")
        guard(None, (), {"data": "hello"})  # Should not raise

    def test_pre_call_unsafe_no_message(self):
        guard = Guard(validator=lambda data: {"safe": False}, on="pre")
        with pytest.raises(UnsafeUserInputError):
            guard(None, (), {"data": "bad"})

    def test_pre_call_unsafe_with_message(self):
        guard = Guard(
            validator=lambda data: {"safe": False},
            on="pre",
            message="Blocked.",
        )
        with pytest.raises(_GuardInterrupt) as exc_info:
            guard(None, (), {"data": "bad"})
        assert exc_info.value.response == "Blocked."

    def test_post_call_safe(self):
        guard = Guard(validator=lambda data: {"safe": True}, on="post")
        guard(None, (), {}, "output")  # Should not raise

    def test_post_call_unsafe_no_message(self):
        guard = Guard(validator=lambda data: {"safe": False}, on="post")
        with pytest.raises(UnsafeModelResponseError):
            guard(None, (), {}, "output")

    def test_post_call_unsafe_with_message(self):
        guard = Guard(
            validator=lambda data: {"safe": False},
            on="post",
            message="Toxic.",
        )
        with pytest.raises(_GuardInterrupt) as exc_info:
            guard(None, (), {}, "output")
        assert exc_info.value.response == "Toxic."

    @pytest.mark.asyncio
    async def test_acall_with_sync_validator(self):
        guard = Guard(validator=lambda data: {"safe": True}, on="pre")
        await guard.acall(None, (), {"data": "hello"})  # Should not raise

    @pytest.mark.asyncio
    async def test_acall_with_coroutine_validator(self):
        async def async_validator(data):
            return {"safe": "bad" not in str(data)}

        guard = Guard(validator=async_validator, on="pre")
        await guard.acall(None, (), {"text": "good"})  # Should not raise

        with pytest.raises(UnsafeUserInputError):
            await guard.acall(None, (), {"text": "bad"})

    @pytest.mark.asyncio
    async def test_acall_with_acall_method(self):
        class ValidatorWithAcall:
            def __call__(self, data):
                return {"safe": True}

            async def acall(self, data):
                return {"safe": False}

        guard = Guard(validator=ValidatorWithAcall(), on="pre")
        with pytest.raises(UnsafeUserInputError):
            await guard.acall(None, (), {"data": "anything"})


# ---------------------------------------------------------------------------
# _GuardInterrupt
# ---------------------------------------------------------------------------


class TestGuardInterrupt:
    def test_stores_response(self):
        exc = _GuardInterrupt("blocked message")
        assert exc.response == "blocked message"

    def test_is_exception(self):
        assert issubclass(_GuardInterrupt, Exception)


# ---------------------------------------------------------------------------
# Guard registration on Module
# ---------------------------------------------------------------------------


class TestGuardRegistration:
    def test_register_pre_hook(self):
        from msgflux.nn.modules.generator import Generator
        from unittest.mock import Mock

        model = Mock()
        gen = Generator(model=model)
        guard = Guard(validator=lambda data: {"safe": True}, on="pre")
        handle = guard.register(gen)

        assert len(gen._forward_pre_hooks) == 1
        handle.remove()
        assert len(gen._forward_pre_hooks) == 0

    def test_register_post_hook(self):
        from msgflux.nn.modules.generator import Generator
        from unittest.mock import Mock

        model = Mock()
        gen = Generator(model=model)
        guard = Guard(validator=lambda data: {"safe": True}, on="post")
        handle = guard.register(gen)

        assert len(gen._forward_hooks) == 1
        handle.remove()
        assert len(gen._forward_hooks) == 0

    def test_register_method_pre_hook(self):
        from msgflux.nn.modules.module import Module

        class GuardedModule(Module):
            def forward(self, x):
                return x

            def validate(self, *, data):
                return data

        module = GuardedModule()
        guard = Guard(
            validator=lambda data: {"safe": True}, on="pre", method="validate"
        )
        handle = guard.register(module)

        assert len(module._method_pre_hooks["validate"]) == 1
        handle.remove()
        assert len(module._method_pre_hooks["validate"]) == 0

    def test_method_pre_guard_rejects_positional_args(self):
        from msgflux.nn.modules.module import Module

        class GuardedModule(Module):
            def forward(self, data):
                return self.validate(data)

            def validate(self, data):
                return data

        module = GuardedModule()
        Guard(
            validator=lambda data: {"safe": True}, on="pre", method="validate"
        ).register(module)

        with pytest.raises(TypeError, match="keyword arguments"):
            module("safe")

    @pytest.mark.asyncio
    async def test_async_method_pre_guard_rejects_positional_args(self):
        from msgflux.nn.modules.module import Module

        class GuardedModule(Module):
            async def aforward(self, data):
                return await self.validate(data)

            async def validate(self, data):
                return data

        module = GuardedModule()
        Guard(
            validator=lambda data: {"safe": True}, on="pre", method="validate"
        ).register(module)

        with pytest.raises(TypeError, match="keyword arguments"):
            await module.acall("safe")


# ---------------------------------------------------------------------------
# Guard target and processor_key
# ---------------------------------------------------------------------------


class TestGuardTarget:
    def test_default_target_is_generator(self):
        guard = Guard(validator=lambda data: {"safe": True}, on="pre")
        assert guard.target == "generator"

    def test_custom_target(self):
        guard = Guard(
            validator=lambda data: {"safe": True}, on="pre", target="embedder"
        )
        assert guard.target == "embedder"

    def test_method_hook_default_target_is_none(self):
        guard = Guard(validator=lambda data: {"safe": True}, on="pre", method="format")
        assert guard.target is None

    def test_processor_key_pre(self):
        guard = Guard(validator=lambda data: {"safe": True}, on="pre")
        assert guard.processor_key == "guard_pre"

    def test_processor_key_post(self):
        guard = Guard(validator=lambda data: {"safe": True}, on="post")
        assert guard.processor_key == "guard_post"

    def test_processor_key_is_none_for_method_hooks(self):
        guard = Guard(validator=lambda data: {"safe": True}, on="post", method="format")
        assert guard.processor_key is None

    def test_hook_base_processor_key_is_none(self):
        from msgflux.nn.hooks import Hook

        hook = Hook(on="pre")
        assert hook.processor_key is None

    def test_hook_base_target_default_none(self):
        from msgflux.nn.hooks import Hook

        hook = Hook(on="pre")
        assert hook.target is None


# ---------------------------------------------------------------------------
# include_data opt-in
# ---------------------------------------------------------------------------


class TestGuardIncludeData:
    def test_include_data_default_false(self):
        guard = Guard(validator=lambda data: {"safe": True}, on="pre")
        assert guard.include_data is False

    def test_include_data_off_no_data_in_exception(self):
        guard = Guard(validator=lambda data: {"safe": False}, on="pre")
        with pytest.raises(UnsafeUserInputError) as exc_info:
            guard(None, (), {"text": "bad input"})
        assert exc_info.value.data is None

    def test_include_data_on_has_data_in_exception(self):
        guard = Guard(
            validator=lambda data: {"safe": False}, on="pre", include_data=True
        )
        with pytest.raises(UnsafeUserInputError) as exc_info:
            guard(None, (), {"text": "bad input"})
        assert exc_info.value.data == {"text": "bad input"}

    def test_include_data_post_hook(self):
        guard = Guard(
            validator=lambda data: {"safe": False}, on="post", include_data=True
        )
        with pytest.raises(UnsafeModelResponseError) as exc_info:
            guard(None, (), {}, "toxic output")
        assert exc_info.value.data == "toxic output"

    def test_include_data_not_set_on_guard_interrupt(self):
        guard = Guard(
            validator=lambda data: {"safe": False},
            on="pre",
            message="Blocked.",
            include_data=True,
        )
        with pytest.raises(_GuardInterrupt):
            guard(None, (), {"text": "bad"})


# ---------------------------------------------------------------------------
# Exception messages
# ---------------------------------------------------------------------------


class TestExceptionMessages:
    def test_unsafe_user_input_default(self):
        exc = UnsafeUserInputError()
        assert str(exc) == "Unsafe user input detected"

    def test_unsafe_user_input_custom(self):
        exc = UnsafeUserInputError("custom message")
        assert str(exc) == "custom message"

    def test_unsafe_model_response_default(self):
        exc = UnsafeModelResponseError()
        assert str(exc) == "Unsafe model response detected"

    def test_unsafe_model_response_custom(self):
        exc = UnsafeModelResponseError("toxic content")
        assert str(exc) == "toxic content"
