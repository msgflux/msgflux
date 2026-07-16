"""Ownership graph used to compose ToolBucket instances."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterator, Mapping

from msgflux.tools.dataclasses import ToolMetadata
from msgflux.tools.types import ToolBucket


@dataclass(frozen=True)
class ToolGraphNode:
    """One visible or captured tool in the bucket ownership tree."""

    name: str
    tool: Any
    config: Mapping[str, Any]
    parent: str | None = None

    @property
    def impl(self) -> Any:
        return getattr(self.tool, "impl", self.tool)

    @property
    def bucket(self) -> ToolBucket | None:
        impl = self.impl
        return impl if isinstance(impl, ToolBucket) else None


class ToolBucketGraph:
    """Query and validate the exclusive ownership tree of a tool library.

    The graph holds references to the library's mutable root mappings. It never
    mutates them or a bucket's captured metadata; ToolLibrary owns all state
    transitions and rollback behavior.
    """

    def __init__(
        self,
        tools: Mapping[str, Any],
        tool_configs: Mapping[str, Mapping[str, Any]],
    ) -> None:
        self._tools = tools
        self._tool_configs = tool_configs

    def iter_nodes(self) -> Iterator[ToolGraphNode]:
        """Yield public roots followed by their recursively captured children."""
        visited_buckets: set[int] = set()

        def visit(node: ToolGraphNode) -> Iterator[ToolGraphNode]:
            yield node
            bucket = node.bucket
            if bucket is None:
                return
            bucket_id = id(bucket)
            if bucket_id in visited_buckets:
                raise ValueError(
                    f"Cycle detected in bucket ownership at `{node.name}`."
                )
            visited_buckets.add(bucket_id)
            for child_name, metadata in bucket.tools.items():
                yield from visit(
                    ToolGraphNode(
                        name=child_name,
                        tool=metadata.source_tool or metadata.impl,
                        config=metadata.tool_config,
                        parent=node.name,
                    )
                )

        for name, tool in self._tools.items():
            yield from visit(
                ToolGraphNode(
                    name=name,
                    tool=tool,
                    config=self._tool_configs.get(name, {}),
                )
            )

    def find_node(self, tool_name: str) -> ToolGraphNode | None:
        """Return the first node with the canonical name, if registered."""
        return next(
            (node for node in self.iter_nodes() if node.name == tool_name),
            None,
        )

    def require_bucket(self, bucket_name: str) -> ToolBucket:
        """Return a bucket implementation or reject an invalid structural node."""
        node = self.find_node(bucket_name)
        bucket = node.bucket if node is not None else None
        if bucket is None:
            raise ValueError(f"The bucket tool `{bucket_name}` cannot capture tools.")
        return bucket

    def find_owner(self, tool_name: str) -> str | None:
        """Return the canonical name of a node's consuming parent."""
        node = self.find_node(tool_name)
        return node.parent if node is not None else None

    def matching_buckets(
        self,
        metadata: ToolMetadata,
    ) -> list[str]:
        """Return the matching owner, evaluating nested buckets first."""
        matches = [
            node.name
            for node in self.iter_nodes()
            if node.bucket is not None
            and node.bucket.captures(metadata)
        ]
        matches.reverse()
        if len(matches) > 1:
            names = ", ".join(f"`{name}`" for name in matches)
            raise ValueError(
                f"Tool `{metadata.name}` matches multiple buckets: {names}."
            )
        return matches

    def capture_candidates(
        self,
        bucket: ToolBucket,
        metadata_factory: Callable[[Any], ToolMetadata],
    ) -> list[ToolGraphNode]:
        """Return visible roots a newly registered bucket would consume."""
        return [
            ToolGraphNode(
                name=name,
                tool=tool,
                config=self._tool_configs.get(name, {}),
            )
            for name, tool in self._tools.items()
            if bucket.captures(metadata_factory(tool))
        ]

    def bucket_descendants(self, bucket_name: str) -> list[ToolGraphNode]:
        """Return only bucket descendants below the selected bucket."""
        descendants: list[ToolGraphNode] = []

        def collect(parent_name: str, bucket: ToolBucket) -> None:
            for name, metadata in bucket.tools.items():
                tool = metadata.source_tool or metadata.impl
                child = ToolGraphNode(
                    name=name,
                    tool=tool,
                    config=metadata.tool_config,
                    parent=parent_name,
                )
                if child.bucket is not None:
                    descendants.append(child)
                    collect(child.name, child.bucket)

        collect(bucket_name, self.require_bucket(bucket_name))
        return descendants

    def validate_registration(
        self,
        metadata: ToolMetadata,
        metadata_factory: Callable[[Any], ToolMetadata],
    ) -> list[ToolGraphNode]:
        """Validate a bucket and return the visible roots it would consume."""
        bucket = metadata.impl
        if not isinstance(bucket, ToolBucket):
            raise ValueError(
                f"The bucket tool `{metadata.name}` must inherit ToolBucket."
            )
        bucket.capture_rules

        for node in self.iter_nodes():
            if node.bucket is None:
                continue
            if ToolBucket.capture_overlaps(bucket, node.bucket):
                raise ValueError(
                    f"The bucket capture for `{metadata.name}` overlaps with "
                    f"`{node.name}`."
                )

        candidates = self.capture_candidates(bucket, metadata_factory)
        for candidate in candidates:
            if candidate.bucket is None:
                continue
            subtree = [candidate, *self.bucket_descendants(candidate.name)]
            for owner in subtree:
                if owner.bucket is not None and owner.bucket.captures(metadata):
                    raise ValueError(
                        f"Bucket capture cycle between `{metadata.name}` and "
                        f"`{owner.name}`."
                    )
        return candidates
