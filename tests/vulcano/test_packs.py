import pytest

from msgflux.vulcano import (
    CommandOptions,
    CommandResult,
    ExtensionPackRegistration,
    VulcanoRuntime,
)


class _DemoPack:
    name = "demo"

    def setup(self, api):
        api.register_command(
            "demo-command",
            CommandOptions(
                description="Command owned by a demo pack.",
                handler=lambda _arguments, _context: CommandResult(),
            ),
        )
        api.ui.set_status("demo", "demo pack active")
        api.ui.set_widget("demo", ["Demo pack"])


def test_extension_pack_groups_and_removes_registered_capabilities():
    runtime = VulcanoRuntime(stream_delay=0, extensions_enabled=False)
    api = runtime.extensions.api

    registration = api.register_pack(_DemoPack())

    assert isinstance(registration, ExtensionPackRegistration)
    assert registration.active
    assert api.packs == ("session-workspace", "demo")
    assert "demo-command" in runtime.commands
    assert [status.text for status in runtime.ui.state.statuses] == ["demo pack active"]
    assert [widget.key for widget in runtime.ui.state.widgets] == ["demo"]

    registration.remove()

    assert not registration.active
    assert api.packs == ("session-workspace",)
    assert "demo-command" not in runtime.commands
    assert runtime.ui.state.statuses == ()
    assert runtime.ui.state.widgets == ()


def test_extension_pack_setup_rolls_back_every_partial_registration():
    runtime = VulcanoRuntime(stream_delay=0, extensions_enabled=False)
    api = runtime.extensions.api

    class BrokenPack:
        name = "broken"

        def setup(self, pack_api):
            pack_api.register_command(
                "partial-pack-command",
                CommandOptions(
                    description="Must be rolled back.",
                    handler=lambda _arguments, _context: CommandResult(),
                ),
            )
            pack_api.ui.set_status("partial", "must disappear")
            raise RuntimeError("pack setup exploded")

    with pytest.raises(RuntimeError, match="pack setup exploded"):
        api.register_pack(BrokenPack())

    assert api.packs == ("session-workspace",)
    assert "partial-pack-command" not in runtime.commands
    assert runtime.ui.state.statuses == ()


def test_extension_pack_rejects_duplicate_names_and_async_setup():
    runtime = VulcanoRuntime(stream_delay=0, extensions_enabled=False)
    api = runtime.extensions.api
    api.register_pack(_DemoPack())

    with pytest.raises(ValueError, match="already registered"):
        api.register_pack(_DemoPack())

    class AsyncPack:
        name = "async-pack"

        async def setup(self, _api):
            return None

    with pytest.raises(TypeError, match="must be synchronous"):
        api.register_pack(AsyncPack())
