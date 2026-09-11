import asyncio

import pytest

from msgflux.chat_messages import ChatMessages
from msgflux.nn.modules.tool import ToolLibrary
from msgflux.nn.modules.tool_runtime import ToolDispatch, ToolPolicy, ToolRef
from msgflux.tools.config import tool_config
from msgflux.tools.runtime import ToolOutcome
from msgflux.tools.types import ToolBucket, ToolBucketEntry, ToolLibraryOperator


def test_handle_executes_stable_tool_ref_through_library_pipeline():
    def double(value: int) -> int:
        """Double a value."""
        return value * 2

    library = ToolLibrary(name="math", tools=[double])
    ref = library.get_tool_ref("double")

    result = library.get_handle()(ref, value=3)

    assert ref == ToolRef(library_id=library.name, tool_id="double")
    assert result == 6


@pytest.mark.asyncio
async def test_handle_rejects_reference_owned_by_another_library():
    def double(value: int) -> int:
        """Double a value."""
        return value * 2

    library = ToolLibrary(name="math", tools=[double])
    foreign = ToolRef(library_id="foreign", tool_id="double")

    with pytest.raises(ValueError, match="belongs to `foreign`"):
        await library.get_handle().acall(foreign, value=3)


def test_bucket_handle_resolves_captured_tool_by_ref_without_accessing_impl():
    class WorkerBucket(ToolBucket, ToolLibraryOperator):
        """Proxy worker tools."""

        name = "worker"
        capture = {"tool_kind": "worker", "defer_loading": False}

        def __call__(self) -> str:
            return "worker"

    @tool_config(tool_kind="worker")
    def double(value: int) -> int:
        """Double a value."""
        return value * 2

    library = ToolLibrary(name="math", tools=[WorkerBucket(), double])
    bucket = library.library["worker"].impl
    ref = bucket.get_ref("double")
    handle = library.get_handle().for_tool(tool_name="worker")

    result = handle(ref, value=4)
    view = library.get_tool_catalog_view(ChatMessages(thread_id="thread_1"))

    assert ref == library.get_tool_ref("double")
    assert result == 8
    assert [entry.name for entry in view.entries] == ["worker"]


def test_bucket_refresh_receives_execution_free_entries():
    class WorkerBucket(ToolBucket, ToolLibraryOperator):
        """Proxy worker tools."""

        name = "worker"
        capture = {"tool_kind": "worker", "defer_loading": False}

        def __init__(self):
            self.entries = ()

        def refresh(self, entries=()):
            self.entries = entries
            names = ", ".join(entry.name for entry in entries) or "none"
            self.description = f"Available workers: {names}."

        def __call__(self) -> str:
            return "worker"

    @tool_config(
        tool_kind="worker",
        display_name="Doubler",
        usage_guidance="Use for multiplication by two.",
    )
    def double(value: int) -> int:
        """Double a value."""
        return value * 2

    bucket = WorkerBucket()
    library = ToolLibrary(name="math", tools=[bucket, double])

    assert bucket.tools == {"double": library.get_tool_ref("double")}
    assert len(bucket.entries) == 1
    [entry] = bucket.entries
    assert isinstance(entry, ToolBucketEntry)
    assert entry.name == "double"
    assert entry.description == "Double a value."
    assert entry.display_name == "Doubler"
    assert entry.usage_guidance == "Use for multiplication by two."
    assert not hasattr(entry, "executor")
    assert (
        library.get_handle().for_tool(tool_name="worker").get_entry("double") == entry
    )
    assert library.get_tool_definition("worker").description == (
        "Available workers: double."
    )
    assert library.get_tool_json_schemas()[0]["function"]["description"] == (
        "Available workers: double."
    )

    library.remove("double")

    assert bucket.entries == ()
    assert library.get_tool_definition("worker").description == (
        "Available workers: none."
    )

    library.add(double)

    assert bucket.tools == {"double": library.get_tool_ref("double")}
    assert library.get_tool_definition("worker").description == (
        "Available workers: double."
    )


def test_bucket_handle_applies_policy_before_captured_tool_execution():
    called = False

    class WorkerBucket(ToolBucket, ToolLibraryOperator):
        """Proxy worker tools."""

        name = "worker"
        capture = {"tool_kind": "worker", "defer_loading": False}

        def __call__(self) -> str:
            return "worker"

    class BlockDouble(ToolPolicy):
        def __init__(self):
            super().__init__("block_double")

        async def before_tool(self, payload):
            if payload.intent.name != "double":
                return payload
            return ToolOutcome.failed(
                payload.intent,
                status="blocked",
                code="policy_denied",
                message="Doubling is disabled.",
            )

    @tool_config(tool_kind="worker")
    def double(value: int) -> int:
        """Double one value."""
        nonlocal called
        called = True
        return value * 2

    library = ToolLibrary(
        name="math",
        tools=[WorkerBucket(), double],
        extensions=[BlockDouble()],
    )
    handle = library.get_handle().for_tool(tool_name="worker")

    with pytest.raises(RuntimeError, match="Doubling is disabled"):
        handle("double", value=4)

    assert called is False


@pytest.mark.asyncio
async def test_handle_routes_through_custom_dispatch_extension():
    class QueueDispatch(ToolDispatch):
        def __init__(self):
            super().__init__("queue_dispatch", dispatch_name="queue")
            self.requests = []

        async def dispatch(self, request):
            self.requests.append(request)
            return ToolOutcome.dispatched(request.plan.intent, result="queued")

    queue = QueueDispatch()

    @tool_config(dispatch="queue")
    async def publish(report_id: str) -> str:
        """Publish one report."""
        raise AssertionError("the queue dispatcher must own execution")

    library = ToolLibrary(name="reports", tools=[publish], extensions=[queue])

    result = await library.get_handle().acall("publish", report_id="rpt_42")

    assert result == "queued"
    assert len(queue.requests) == 1
    assert queue.requests[0].plan.visible_arguments == {"report_id": "rpt_42"}


@pytest.mark.asyncio
async def test_concurrent_bucket_handles_keep_runtime_inputs_isolated():
    class WorkerBucket(ToolBucket, ToolLibraryOperator):
        """Proxy worker tools."""

        name = "worker"
        capture = {"tool_kind": "worker", "defer_loading": False}

        def __call__(self) -> str:
            return "worker"

    @tool_config(tool_kind="worker", runtime_inputs=["vars"])
    def identify(value: str, vars: dict) -> str:
        """Identify one value in the current tenant."""
        return f"{vars['tenant']}:{value}"

    library = ToolLibrary(name="workers", tools=[WorkerBucket(), identify])
    first = library.get_handle().for_tool(
        tool_name="worker",
        vars={"tenant": "acme"},
    )
    second = library.get_handle().for_tool(
        tool_name="worker",
        vars={"tenant": "globex"},
    )

    results = await asyncio.gather(
        first.acall("identify", value="one"),
        second.acall("identify", value="two"),
    )

    assert results == ["acme:one", "globex:two"]
