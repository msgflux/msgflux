"""Tests for the read-only ToolBucket ownership graph."""

import msgflux as mf
import pytest

from msgflux.nn.modules.tool import ToolLibrary
from msgflux.tools import ToolBucket
from msgflux.tools.dataclasses import ToolMetadata


class _InnerBucket(ToolBucket):
    """Capture graph leaves."""

    name = "inner"
    capture = {"tool_kind": "graph_leaf"}

    def __call__(self) -> str:
        return "inner"


class _OuterBucket(ToolBucket):
    """Capture the inner bucket."""

    name = "outer"
    capture = {"source": "bucket", "name": "inner"}

    def __call__(self) -> str:
        return "outer"


@mf.tool_config(tool_kind="graph_leaf")
def _leaf() -> str:
    """Return a graph leaf."""
    return "leaf"


def test_bucket_graph_exposes_nested_nodes_and_owners():
    library = ToolLibrary(
        name="graph",
        tools=[_OuterBucket(), _InnerBucket(), _leaf],
    )
    graph = library._bucket_graph

    assert [node.name for node in graph.iter_nodes()] == [
        "outer",
        "inner",
        "_leaf",
    ]
    assert graph.find_owner("inner") == "outer"
    assert graph.find_owner("_leaf") == "inner"
    assert graph.require_bucket("inner").tools["_leaf"].impl() == "leaf"


def test_bucket_graph_rejects_non_bucket_nodes():
    library = ToolLibrary(name="graph", tools=[_leaf])

    with pytest.raises(ValueError, match="cannot capture tools"):
        library._bucket_graph.require_bucket("_leaf")


def test_library_rejects_collision_inside_prepopulated_bucket():
    def lookup() -> str:
        """Return a lookup result."""
        return "root"

    class LookupBucket(ToolBucket):
        """Capture lookup tools."""

        name = "lookup_bucket"
        capture = {"name": "lookup"}

        def __call__(self) -> str:
            return "bucket"

    bucket = LookupBucket()
    bucket.add(
        ToolMetadata(
            name="lookup",
            description="Captured lookup.",
            annotations={"return": str},
            tool_config={"tool_kind": "tool", "on_demand": False},
            impl=lambda: "captured",
        )
    )
    library = ToolLibrary(name="graph", tools=[lookup])

    with pytest.raises(ValueError, match="Duplicate tool name `lookup`"):
        library.add(bucket)

    assert list(library.library) == ["lookup"]
