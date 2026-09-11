import asyncio
from copy import deepcopy

import msgspec
import pytest

from msgflux.nn.modules.module import Module
from msgflux.nn.modules.tool_runtime import (
    LoadingSpec,
    ToolDefinition,
    ToolRef,
    ToolRegistry,
)
from msgflux.tools.definitions import ToolCatalog


class RecordingExecutor(Module):
    def __init__(self, *, result=None, error=None):
        super().__init__()
        self.result = result
        self.error = error
        self.calls = []
        self.started = asyncio.Event()
        self.release = None

    def forward(self, **arguments):
        self.calls.append(arguments)
        if self.error is not None:
            raise self.error
        return self.result

    async def aforward(self, **arguments):
        self.calls.append(arguments)
        self.started.set()
        if self.release is not None:
            await self.release.wait()
        if self.error is not None:
            raise self.error
        return self.result


def make_definition(name="lookup", **overrides):
    values = {
        "name": name,
        "executor": RecordingExecutor(result={"found": True}),
        "description": "Look up an inventory item.",
        "input_schema": {
            "type": "object",
            "properties": {"sku": {"type": "string"}},
            "required": ["sku"],
        },
    }
    values.update(overrides)
    return ToolDefinition(**values)


def test_registry_owns_stable_definitions_without_executor_modules():
    definition = make_definition()
    registry = ToolRegistry("warehouse_tools", [definition])

    ref = registry.ref("lookup")

    assert ref == ToolRef(library_id="warehouse_tools", tool_id="lookup")
    assert registry.get(ref) is definition
    assert not hasattr(registry, "executors")
    with pytest.raises(ValueError, match="already registered"):
        registry.add(definition)

    removed = registry.remove(ref)

    assert removed is definition
    assert not registry.has("lookup")


def test_registry_rejects_foreign_refs():
    registry = ToolRegistry("warehouse_tools", [make_definition()])
    foreign = ToolRef(library_id="other_tools", tool_id="lookup")

    with pytest.raises(ValueError, match="belongs to `other_tools`"):
        registry.get(foreign)


def test_registry_replace_preserves_executor_ownership():
    definition = make_definition()
    registry = ToolRegistry("warehouse_tools", [definition])
    updated = msgspec.structs.replace(definition, description="Updated lookup.")

    previous = registry.replace(updated)

    assert previous is definition
    assert registry.get("lookup") is updated
    with pytest.raises(ValueError, match="cannot change its executor"):
        registry.replace(
            msgspec.structs.replace(updated, executor=RecordingExecutor(result="new"))
        )


def test_registry_deepcopy_copies_definition_executor_reference():
    registry = ToolRegistry("warehouse_tools", [make_definition()])

    copied = deepcopy(registry)
    definition = copied.get("lookup")

    assert definition.executor is not registry.get("lookup").executor


def test_catalog_views_isolate_deferred_loading_by_thread():
    registry = ToolRegistry(
        "warehouse_tools",
        [
            make_definition(),
            make_definition(
                "reconcile_inventory",
                loading=LoadingSpec(deferred=True),
            ),
        ],
    )

    first = registry.catalog_view(
        "thread_a",
        loaded_tools=("reconcile_inventory",),
    )
    second = registry.catalog_view("thread_b")

    assert [entry.name for entry in first.visible_entries()] == [
        "lookup",
        "reconcile_inventory",
    ]
    assert [entry.name for entry in second.visible_entries()] == ["lookup"]
    assert first.entries[1].loaded
    assert not second.entries[1].loaded
    assert registry.get("reconcile_inventory").loading.deferred


def test_catalog_selection_can_expose_one_unloaded_deferred_tool():
    registry = ToolRegistry(
        "warehouse_tools",
        [make_definition("reconcile", loading=LoadingSpec(deferred=True))],
    )

    view = registry.catalog_view("thread_a", choice="reconcile")

    assert [entry.name for entry in view.visible_entries()] == ["reconcile"]
    assert view.choice.mode == "tool"


@pytest.mark.parametrize(
    ("choice", "expected_name"),
    [
        ({"type": "function", "function": {"name": "reconcile"}}, "reconcile"),
        ({"type": "function", "name": "reconcile"}, "reconcile"),
    ],
)
def test_catalog_choice_normalizes_provider_function_shapes(choice, expected_name):
    registry = ToolRegistry("warehouse_tools", [make_definition("reconcile")])

    view = registry.catalog_view("thread_a", choice=choice)

    assert view.choice.mode == "tool"
    assert view.choice.name == expected_name


def test_catalog_choice_rejects_provider_specific_non_function_shape():
    registry = ToolRegistry("warehouse_tools", [make_definition("reconcile")])

    with pytest.raises(ValueError, match="must select a function tool"):
        registry.catalog_view(
            "thread_a",
            choice={"type": "computer", "name": "reconcile"},
        )


def test_catalog_view_identifies_search_by_role_and_preserves_entry_metadata():
    search = make_definition(
        "discover",
        metadata={"catalog_role": "search"},
    )
    deferred = make_definition(
        "remote_lookup",
        loading=LoadingSpec(deferred=True),
        metadata={
            "strict": True,
            "execution_namespace": "remote",
        },
    )
    registry = ToolRegistry("warehouse_tools", [search, deferred])

    unloaded = registry.catalog_view("thread_a")
    loaded = registry.catalog_view(
        "thread_a",
        loaded_tools=("remote_lookup",),
    )
    selected = registry.catalog_view("thread_b", choice="remote_lookup")

    assert unloaded.search_entry.name == "discover"
    assert [entry.name for entry in unloaded.visible_entries()] == ["discover"]
    assert [entry.name for entry in loaded.visible_entries()] == ["remote_lookup"]
    assert [entry.name for entry in selected.visible_entries()] == ["remote_lookup"]
    assert unloaded.tool_entries()[0].strict is True
    assert unloaded.tool_entries()[0].namespace == "remote"


def test_legacy_catalog_adapter_preserves_canonical_view_semantics():
    registry = ToolRegistry(
        "warehouse_tools",
        [
            make_definition("discover", metadata={"catalog_role": "search"}),
            make_definition(
                "remote_lookup",
                loading=LoadingSpec(deferred=True),
                metadata={"strict": True, "execution_namespace": "remote"},
            ),
        ],
    )
    view = registry.catalog_view("thread_a", choice="remote_lookup")

    catalog = ToolCatalog.from_view(view)

    assert catalog.catalog_id == "warehouse_tools"
    assert catalog.choice == "remote_lookup"
    assert catalog.search_tool.name == "discover"
    assert catalog.tools[0].strict is True
    assert catalog.tools[0].namespace == "remote"
    assert catalog.tools[0].ref == registry.ref("remote_lookup")
    assert [tool.name for tool in catalog.portable_tools()] == ["remote_lookup"]


def test_catalog_view_can_project_public_subset_without_losing_registry_entries():
    registry = ToolRegistry(
        "warehouse_tools",
        [make_definition("bucket"), make_definition("captured")],
    )

    view = registry.catalog_view("thread_a", include_tools=("bucket",))

    assert [entry.name for entry in view.entries] == ["bucket"]
    assert registry.has("captured")


def test_catalog_view_filters_regular_tools_without_losing_search_role():
    registry = ToolRegistry(
        "warehouse_tools",
        [
            make_definition("discover", metadata={"catalog_role": "search"}),
            make_definition("lookup"),
            make_definition("reconcile", loading=LoadingSpec(deferred=True)),
        ],
    )

    view = registry.catalog_view("thread_a", choice="reconcile")
    filtered = view.with_tools(("lookup",))

    assert [entry.name for entry in filtered.entries] == ["discover", "lookup"]
    assert [entry.name for entry in filtered.visible_entries()] == ["lookup"]
    assert filtered.choice.mode == "auto"
    assert len(view.entries) == 3


def test_catalog_view_applies_choice_after_filtering():
    registry = ToolRegistry("warehouse_tools", [make_definition("lookup")])
    view = registry.catalog_view("thread_a")

    selected = view.with_choice({"type": "function", "function": {"name": "lookup"}})
    empty = view.with_tools(()).with_choice("lookup")

    assert selected.choice.mode == "tool"
    assert selected.choice.name == "lookup"
    assert empty.choice.mode == "none"


def test_catalog_view_projects_portable_schemas_and_annotations():
    definition = make_definition(
        "lookup",
        annotations={"sku": str},
        metadata={"strict": True},
    )
    view = ToolRegistry("warehouse_tools", [definition]).catalog_view("thread_a")

    assert view.portable_schemas() == [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Look up an inventory item.",
                "parameters": definition.input_schema,
                "strict": True,
            },
        }
    ]
    assert view.annotations == {"lookup": {"sku": str}}
