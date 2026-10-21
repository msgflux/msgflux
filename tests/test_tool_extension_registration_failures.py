from __future__ import annotations

import asyncio

import pytest

from msgflux.nn import ToolLibrary
from msgflux.nn.extensions import ToolLibraryExtension
from msgflux.nn.modules.tool_runtime import (
    ToolContextProvider,
    ToolDispatch,
    ToolExtensionRegistry,
    ToolPolicy,
)


def test_runtime_dispatch_registration_preserves_primary_and_rolls_back():
    primary = ValueError("dispatch setup failed")
    cleanup = OSError("dispatch cleanup failed")

    class QueueDispatch(ToolDispatch):
        def __init__(self):
            super().__init__("dispatch_queue", dispatch_name="queue")
            self.fail_setup = True
            self.fail_cleanup = True

        async def dispatch(self, request):
            raise NotImplementedError

        def on_register(self, _registry):
            if self.fail_setup:
                raise primary

        def on_remove(self, _registry):
            if self.fail_cleanup:
                raise cleanup

    registry = ToolExtensionRegistry()
    extension = QueueDispatch()
    with pytest.raises(ValueError) as raised:
        registry.register(extension)
    assert raised.value is primary
    assert any("cleanup failed" in note for note in raised.value.__notes__)
    assert not registry.has(extension.name)
    with pytest.raises(ValueError, match="not registered"):
        registry.get_dispatch("queue")
    with pytest.raises(RuntimeError, match="not registered"):
        _ = extension.registry

    extension.fail_setup = False
    extension.fail_cleanup = False
    handle = registry.register(extension)
    assert handle.active
    assert registry.get_dispatch("queue") is extension
    handle.remove()


@pytest.mark.asyncio
async def test_runtime_context_registration_preserves_primary_and_rolls_back():
    primary = RuntimeError("provider setup failed")
    cleanup = LookupError("provider cleanup failed")

    class TenantProvider(ToolContextProvider):
        def __init__(self):
            super().__init__("context_tenant", sources=("tenant",))
            self.fail_setup = True
            self.fail_cleanup = True

        async def resolve(self, _request):
            return None

        async def aon_register(self, _registry):
            if self.fail_setup:
                raise primary

        async def aon_remove(self, _registry):
            if self.fail_cleanup:
                raise cleanup

    registry = ToolExtensionRegistry()
    extension = TenantProvider()
    with pytest.raises(RuntimeError) as raised:
        await registry.aregister(extension)
    assert raised.value is primary
    assert any("provider cleanup failed" in note for note in raised.value.__notes__)
    assert not registry.has(extension.name)
    with pytest.raises(ValueError, match="not registered"):
        registry.get_context_provider("tenant")
    with pytest.raises(RuntimeError, match="not registered"):
        _ = extension.registry

    extension.fail_setup = False
    extension.fail_cleanup = False
    handle = await registry.aregister(extension)
    assert handle.active
    assert registry.get_context_provider("tenant") is extension
    handle.remove()


@pytest.mark.asyncio
async def test_runtime_registration_does_not_swallow_cleanup_cancellation():
    primary = ValueError("setup failed")
    cancellation = asyncio.CancelledError("cleanup cancelled")

    class Policy(ToolPolicy):
        def __init__(self):
            super().__init__("policy")

        async def aon_register(self, _registry):
            raise primary

        async def aon_remove(self, _registry):
            raise cancellation

    registry = ToolExtensionRegistry()
    extension = Policy()
    with pytest.raises(asyncio.CancelledError) as raised:
        await registry.aregister(extension)
    assert raised.value is cancellation
    assert not registry.has(extension.name)
    assert extension.name not in registry._policies
    with pytest.raises(RuntimeError, match="not registered"):
        _ = extension.registry


def test_library_extension_registration_preserves_primary_and_rolls_back():
    primary = ValueError("library setup failed")
    cleanup = OSError("library cleanup failed")

    class ExampleExtension(ToolLibraryExtension):
        def __init__(self):
            super().__init__("example")
            self.fail_setup = True
            self.fail_cleanup = True

        def tools(self):
            def supplied() -> str:
                """Return the extension-owned value."""
                return "value"

            return (supplied,)

        def on_register(self, _library):
            if self.fail_setup:
                raise primary

        def on_remove(self, _library):
            if self.fail_cleanup:
                raise cleanup

    library = ToolLibrary("registration", [])
    extension = ExampleExtension()
    with pytest.raises(ValueError) as raised:
        library.register_extension("example", extension)
    assert raised.value is primary
    assert any("library cleanup failed" in note for note in raised.value.__notes__)
    assert not library.has_extension("example")
    assert "supplied" not in library.get_tool_names()
    with pytest.raises(RuntimeError, match="not registered"):
        _ = extension.library

    extension.fail_setup = False
    extension.fail_cleanup = False
    handle = library.register_extension("example", extension)
    assert handle.active
    assert "supplied" in library.get_tool_names()
    handle.remove()
