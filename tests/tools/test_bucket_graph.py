"""Tests for the read-only ToolBucket ownership graph."""

import msgflux as mf
import pytest

from msgflux.nn.modules.tool import ToolLibrary
from msgflux.tools import ToolBucket


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
