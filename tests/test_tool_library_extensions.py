import asyncio
from copy import deepcopy
from dataclasses import replace

import pytest

from msgflux.nn import ToolLibrary
from msgflux.nn.extensions import BackgroundTasksExtension, ToolLibraryExtension
from msgflux.nn.hooks import Hook
from msgflux.nn.modules.module import Module
from msgflux.tools.config import tool_config


class MathExtension(ToolLibraryExtension):
    def __init__(self):
        super().__init__("math")

    def tools(self):
        def double(value: int) -> int:
            """Double a value."""
            return value * 2

        return (double,)

    def hooks(self):
        return (
            Hook(
                event="before_tool",
                handler=lambda event: replace(
                    event,
                    arguments={"value": event.arguments["value"] + 1},
                ),
            ),
            Hook(
                event="after_tool",
                handler=lambda event: replace(event, result=event.result + 1),
            ),
        )


def test_tool_library_extension_contributes_tools_and_lifecycle_hooks():
    library = ToolLibrary("math", [], extensions=[MathExtension()])

    response = library([("call_1", "double", {"value": 2})])

    assert response.tool_calls[0].parameters == {"value": 3}
    assert response.tool_calls[0].result == 7


def test_tool_library_extension_handle_removes_owned_contributions():
    library = ToolLibrary("math", [])
    handle = library.register_extension("math", MathExtension())

    assert handle.active
    assert "double" in library.get_tool_names()

    handle.remove()

    assert not handle.active
    assert "double" not in library.get_tool_names()
    response = library([("call_1", "double", {"value": 2})])
    assert response.tool_calls[0].error == "Error: Tool `double` not found."


@pytest.mark.asyncio
async def test_tool_library_extension_supports_async_hooks_and_cleanup():
    cleaned = asyncio.Event()

    class AsyncExtension(MathExtension):
        async def increase(self, event):
            await asyncio.sleep(0)
            return replace(
                event,
                arguments={"value": event.arguments["value"] + 2},
            )

        def hooks(self):
            return (Hook(event="before_tool", handler=self.increase),)

        async def aon_remove(self, _library):
            cleaned.set()

    library = ToolLibrary("math", [])
    handle = library.register_extension("math", AsyncExtension())

    response = await library.acall([("call_1", "double", {"value": 2})])
    assert response.tool_calls[0].result == 8

    await handle.aremove()
    assert cleaned.is_set()


def test_deferred_tools_install_tool_search_as_an_extension():
    def lookup(query: str) -> str:
        """Look up a value."""
        return query

    lookup.tool_config = {"defer_loading": True}
    library = ToolLibrary("search", [lookup])

    assert library.has_extension("tool_search")
    assert library.bucket_has_tool("tool_search", "lookup")

    library.remove("lookup")
    library.add(lookup)

    assert library.bucket_has_tool("tool_search", "lookup")


def test_background_controls_are_managed_by_builtin_extension():
    def long_job() -> str:
        """Run a background job."""
        return "done"

    long_job.tool_config = {"background": True}
    library = ToolLibrary("tasks", [long_job])

    assert library.has_extension("background_tasks")
    assert "task_status" in library.get_tool_names()


def test_library_extensions_receive_sequential_tool_lifecycle_callbacks():
    events = []

    class LifecycleExtension(ToolLibraryExtension):
        def __init__(self):
            super().__init__("lifecycle")

        def tools(self):
            def owned() -> str:
                """Return an extension-owned value."""
                return "owned"

            return (owned,)

        def validate_tool(self, _library, definition):
            events.append(("validate", definition.name))

        def on_tool_added(self, _library, definition):
            events.append(("added", definition.name))

        def on_tool_removed(self, _library, definition):
            events.append(("removed", definition.name))

        def on_clear(self, _library):
            events.append(("clear", None))

    def lookup() -> str:
        """Look up one value."""
        return "ok"

    library = ToolLibrary("search", [], extensions=[LifecycleExtension()])
    library.add(lookup)
    library.remove("lookup")
    library.add(lookup)
    library.clear()

    assert events == [
        ("validate", "owned"),
        ("added", "owned"),
        ("validate", "lookup"),
        ("added", "lookup"),
        ("removed", "lookup"),
        ("validate", "lookup"),
        ("added", "lookup"),
        ("clear", None),
    ]


def test_extension_validation_prevents_registration_before_state_changes():
    class RejectDangerousTool(ToolLibraryExtension):
        def __init__(self):
            super().__init__("reject_dangerous")

        def validate_tool(self, _library, definition):
            if definition.name == "dangerous":
                raise ValueError("Dangerous tool rejected.")

    def dangerous() -> str:
        """Perform a dangerous action."""
        return "unsafe"

    library = ToolLibrary("guarded", [], extensions=[RejectDangerousTool()])

    with pytest.raises(ValueError, match="Dangerous tool rejected"):
        library.add(dangerous)

    assert not library.registry.has("dangerous")
    assert "dangerous" not in library.library


def test_background_extension_owns_disabled_controls_and_restores_on_readd():
    def long_job() -> str:
        """Run a background job."""
        return "done"

    long_job.tool_config = {"background": True}
    library = ToolLibrary("tasks", [long_job])
    background = library.extensions["background_tasks"]

    library.remove("task_status")

    assert "task_status" in background.disabled_tool_names
    assert not hasattr(library, "_disabled_background_task_tool_names")
    copied = deepcopy(library)
    assert "task_status" in copied.extensions["background_tasks"].disabled_tool_names
    copied.clear()
    assert copied.extensions["background_tasks"].disabled_tool_names == set()

    library.remove_extension("background_tasks")
    library.register_extension("background_tasks", BackgroundTasksExtension())

    assert "task_status" in library.get_tool_names()


def test_before_tool_stops_current_call_after_first_block():
    calls = []

    def guarded(value: int) -> int:
        """Return a guarded value."""
        calls.append("tool")
        return value

    class PolicyExtension(ToolLibraryExtension):
        def __init__(self, name, handler):
            super().__init__(name)
            self.handler = handler

        def hooks(self):
            return (Hook(event="before_tool", handler=self.handler),)

    def allow(event):
        calls.append("allow")
        return event

    def deny(event):
        calls.append("deny")
        return replace(event, block="Denied by policy.")

    def must_not_run(event):
        calls.append("late_library_hook")
        return event

    library = ToolLibrary(
        "guarded",
        [guarded],
        extensions=[
            PolicyExtension("allow", allow),
            PolicyExtension("deny", deny),
            PolicyExtension("late", must_not_run),
        ],
    )
    owner = Module()
    Hook(
        event="before_tool",
        handler=lambda event: calls.append("owner_hook") or event,
    ).register(owner)
    library.set_lifecycle_owner(owner)

    response = library([("call_1", "guarded", {"value": 3})])

    assert calls == ["allow", "deny"]
    assert response.tool_calls[0].error == "Denied by policy."


@pytest.mark.asyncio
async def test_async_before_tool_is_sequential_and_block_is_call_local():
    calls = []

    def left() -> str:
        """Return left."""
        calls.append("left.tool")
        return "left"

    def right() -> str:
        """Return right."""
        calls.append("right.tool")
        return "right"

    class PolicyExtension(ToolLibraryExtension):
        def __init__(self):
            super().__init__("policy")

        async def check(self, event):
            calls.append(f"{event.tool_name}.first")
            await asyncio.sleep(0)
            if event.tool_name == "left":
                return replace(event, block="Left is blocked.")
            return event

        async def later(self, event):
            calls.append(f"{event.tool_name}.second")
            await asyncio.sleep(0)
            return event

        def hooks(self):
            return (
                Hook(event="before_tool", handler=self.check),
                Hook(event="before_tool", handler=self.later),
            )

    library = ToolLibrary("pair", [left, right], extensions=[PolicyExtension()])

    response = await library.acall(
        [("call_left", "left", {}), ("call_right", "right", {})]
    )

    assert calls == [
        "left.first",
        "right.first",
        "right.second",
        "right.tool",
    ]
    assert response.tool_calls[0].error == "Left is blocked."
    assert response.tool_calls[1].result == "right"


def test_before_dispatch_can_reduce_detached_to_foreground():
    observed_modes = []

    @tool_config(detached=True)
    def report() -> str:
        """Return a report."""
        return "ready"

    class ForegroundPolicy(ToolLibraryExtension):
        def __init__(self):
            super().__init__("foreground_policy")

        def hooks(self):
            def force_foreground(event):
                observed_modes.append(event.dispatch_mode)
                return replace(event, dispatch_mode="foreground")

            return (Hook(event="before_dispatch", handler=force_foreground),)

    library = ToolLibrary("reports", [report], extensions=[ForegroundPolicy()])

    response = library([("call_1", "report", {})])

    assert observed_modes == ["detached"]
    assert response.tool_calls[0].result == "ready"


def test_before_dispatch_stops_after_first_block_and_fails_closed():
    calls = []

    def guarded() -> str:
        """Return a guarded result."""
        calls.append("tool")
        return "unsafe"

    class DispatchPolicy(ToolLibraryExtension):
        def __init__(self):
            super().__init__("dispatch_policy")

        def hooks(self):
            def deny(event):
                calls.append("deny")
                return replace(event, block="Dispatch denied.")

            def must_not_run(event):
                calls.append("late")
                return event

            return (
                Hook(event="before_dispatch", handler=deny),
                Hook(event="before_dispatch", handler=must_not_run),
                Hook(
                    event="after_tool",
                    handler=lambda event: calls.append("after") or event,
                ),
            )

    library = ToolLibrary(
        "guarded",
        [guarded],
        extensions=[DispatchPolicy()],
    )

    response = library([("call_1", "guarded", {})])

    assert calls == ["deny"]
    assert response.tool_calls[0].error == "Dispatch denied."


def test_before_dispatch_cannot_promote_foreground_to_detached():
    def guarded() -> str:
        """Return a guarded result."""
        return "unsafe"

    class InvalidPolicy(ToolLibraryExtension):
        def __init__(self):
            super().__init__("invalid_policy")

        def hooks(self):
            return (
                Hook(
                    event="before_dispatch",
                    handler=lambda event: replace(event, dispatch_mode="detached"),
                ),
            )

    library = ToolLibrary(
        "guarded",
        [guarded],
        extensions=[InvalidPolicy()],
    )

    response = library([("call_1", "guarded", {})])

    assert response.tool_calls[0].error == (
        "before_dispatch hook failed closed: before_dispatch may only keep the "
        "selected mode or reduce `background`/`detached` dispatch to `foreground`"
    )


def test_loading_tools_emits_only_successful_new_additions():
    from msgflux.chat_messages import ChatMessages
    from msgflux.runtime.events import _capture_events, _EventSink

    def lookup(query: str) -> str:
        """Look up a value."""
        return query

    lookup.tool_config = {"defer_loading": True}
    library = ToolLibrary("search", [lookup])
    messages = ChatMessages(thread_id="discovery")
    events = []
    with _capture_events(_EventSink(events.append)):
        with pytest.raises(ValueError):
            library.load_tools(messages, ["missing"])
        assert library.load_tools(messages, ["lookup"]) == ["lookup"]
        assert library.load_tools(messages, ["lookup"]) == ["lookup"]
    assert [(event.type, event.data) for event in events] == [
        (
            "tools.updated",
            {
                "catalog_id": library.name,
                "loaded_tools": ["lookup"],
                "execution": "client",
            },
        )
    ]
    assert messages.get_loaded_tools(library.name) == {"lookup"}


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_portable_search_loading_and_execution_event_order(asynchronous):
    from msgflux.chat_messages import ChatMessages
    from msgflux.runtime.events import _capture_events, _EventSink
    from msgflux.tools.runtime import ToolIntent

    def lookup(query: str) -> str:
        """Look up a value."""
        return query.upper()

    lookup.tool_config = {"defer_loading": True}
    library = ToolLibrary("search", [lookup])
    messages = ChatMessages(thread_id="discovery")
    events = []
    execute = library.aexecute_intents if asynchronous else library.execute_intents

    async def run(name, arguments, call_id):
        result = execute(
            [ToolIntent(id=call_id, name=name, arguments=arguments)], messages=messages
        )
        return await result if asynchronous else result

    with _capture_events(_EventSink(events.append)):
        searched = await run("tool_search", {"query": "lookup"}, "find")
        assert searched[0].result["matches"] == ["lookup"]
        assert not messages.get_loaded_tools(library.name)
        loaded = await run("tool_search", {"select": ["lookup"]}, "load")
        assert loaded[0].result["loaded"] == ["lookup"]
        assert "lookup" in {
            entry.name
            for entry in library.get_tool_catalog_view(messages).visible_entries()
        }
        assert (await run("lookup", {"query": "value"}, "use"))[0].result == "VALUE"
    assert [event.type for event in events] == [
        "tool.start",
        "tool.end",
        "tool.start",
        "tools.updated",
        "tool.end",
        "tool.start",
        "tool.end",
    ]


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("empty_message", ["", "   "])
@pytest.mark.asyncio
async def test_dispatch_task_errors_keep_structured_details(
    asynchronous, empty_message, tmp_path
):
    from msgflux.runtime import AgentWorkspace, ExecutionScope, execution_context
    from msgflux.runtime.events import _EventSink, _capture_events
    from msgflux.tools.builtin import ReadFileTool
    from msgflux.tools.runtime import ToolIntent

    def empty_failure() -> None:
        raise RuntimeError(empty_message)

    library = ToolLibrary("workspace-errors", [ReadFileTool(), empty_failure])
    workspace = AgentWorkspace.local(tmp_path)
    intents = (
        ToolIntent(id="missing-file", name="read", arguments={"path": "absent.txt"}),
        ToolIntent(id="empty-failure", name="empty_failure"),
    )
    events = []
    try:
        with execution_context(scope=ExecutionScope(workspace=workspace)):
            with _capture_events(_EventSink(events.append)):
                outcomes = (
                    await library.aexecute_intents(intents)
                    if asynchronous
                    else library.execute_intents(intents)
                )

        assert [outcome.status for outcome in outcomes] == [
            "execution_failed",
            "execution_failed",
        ]
        read_error, empty_error = (outcome.error for outcome in outcomes)
        assert read_error.code == empty_error.code == "tool_execution_failed"
        assert "Task " in read_error.message and "absent.txt" in read_error.message
        assert "Task " in empty_error.message
        assert read_error.details["exception_type"] == "FileNotFoundError"
        assert empty_error.details["exception_type"] == "RuntimeError"

        ends = {
            event.data["tool_call_id"]: event.data
            for event in events
            if event.type == "tool.end"
        }
        assert "absent.txt" in ends["missing-file"]["error_info"]["message"]
        assert (
            ends["missing-file"]["error"]
            == ends["missing-file"]["error_info"]["message"]
        )
        assert ends["missing-file"]["error_info"]["code"] == read_error.code
        assert ends["missing-file"]["error_info"]["details"] == dict(read_error.details)
        assert ends["empty-failure"]["error"] == empty_message
        assert ends["empty-failure"]["error_info"]["message"] == "RuntimeError"
    finally:
        await workspace.aclose()
