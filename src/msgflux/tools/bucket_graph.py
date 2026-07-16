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

    The graph holds references to the library's flat registry, configuration
    map, and ownership edges. It never mutates them; ToolLibrary owns all state
    transitions and rollback behavior.
    """

    def __init__(
        self,
        tools: Mapping[str, Any],
        tool_configs: Mapping[str, Mapping[str, Any]],
        owners: Mapping[str, str],
    ) -> None:
        self._tools = tools
        self._tool_configs = tool_configs
        self._owners = owners

    def iter_nodes(self) -> Iterator[ToolGraphNode]:
        """Yield ownership roots followed by their recursively captured children."""
        children = self._children_by_owner()
        visited: set[str] = set()
        active: set[str] = set()

        def visit(name: str) -> Iterator[ToolGraphNode]:
            if name in active:
                raise ValueError(f"Cycle detected in bucket ownership at `{name}`.")
            if name in visited:
                return
            active.add(name)
            tool = self._tools[name]
            node = ToolGraphNode(
                name=name,
                tool=tool,
                config=self._tool_configs.get(name, {}),
                parent=self._owners.get(name),
            )
            yield node
            for child_name in children.get(name, ()):
                yield from visit(child_name)
            active.remove(name)
            visited.add(name)

        for name in self._tools:
            if name not in self._owners:
                yield from visit(name)
        for name in self._tools:
            if name not in visited:
                owner = self._owners.get(name)
                if owner not in self._tools:
                    raise ValueError(
                        f"Tool `{name}` references missing bucket owner `{owner}`."
                    )
                yield from visit(name)

    def find_node(self, tool_name: str) -> ToolGraphNode | None:
        """Return the unique node with the canonical name, if registered."""
        tool = self._tools.get(tool_name)
        if tool is None:
            return None
        owner = self._owners.get(tool_name)
        if owner is not None and owner not in self._tools:
            raise ValueError(
                f"Tool `{tool_name}` references missing bucket owner `{owner}`."
            )
        return ToolGraphNode(
            name=tool_name,
            tool=tool,
            config=self._tool_configs.get(tool_name, {}),
            parent=owner,
        )

    def _children_by_owner(self) -> dict[str, list[str]]:
        children: dict[str, list[str]] = {}
        for name in self._tools:
            owner = self._owners.get(name)
            if owner is not None:
                children.setdefault(owner, []).append(name)
        return children

    def validate_unique_names(self, metadata: ToolMetadata) -> None:
        """Reject collisions in a pending tool subtree and the current graph."""
        existing = set(self._tools)
        pending: set[str] = set()

        def visit(candidate: ToolMetadata) -> None:
            if candidate.name in existing or candidate.name in pending:
                raise ValueError(
                    f"Duplicate tool name `{candidate.name}`: already in tool library."
                )
            pending.add(candidate.name)
            bucket = candidate.impl
            if isinstance(bucket, ToolBucket):
                for child in bucket.tools.values():
                    visit(child)

        visit(metadata)

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
            if name not in self._owners and bucket.captures(metadata_factory(tool))
        ]

    def bucket_descendants(self, bucket_name: str) -> list[ToolGraphNode]:
        """Return all descendants below the selected bucket."""
        self.require_bucket(bucket_name)
        children = self._children_by_owner()
        descendants: list[ToolGraphNode] = []

        def collect(parent_name: str) -> None:
            for name in children.get(parent_name, ()):
                child = ToolGraphNode(
                    name=name,
                    tool=self._tools[name],
                    config=self._tool_configs.get(name, {}),
                    parent=parent_name,
                )
                descendants.append(child)
                if child.bucket is not None:
                    collect(child.name)

        collect(bucket_name)
        return descendants

    def is_descendant(self, bucket_name: str, tool_name: str) -> bool:
        """Return whether ``tool_name`` is owned below ``bucket_name``."""
        self.require_bucket(bucket_name)
        if tool_name not in self._tools:
            return False
        visited: set[str] = set()
        current = tool_name
        while current in self._owners:
            if current in visited:
                raise ValueError(f"Cycle detected in bucket ownership at `{current}`.")
            visited.add(current)
            owner = self._owners[current]
            if owner == bucket_name:
                return True
            if owner not in self._tools:
                raise ValueError(
                    f"Tool `{current}` references missing bucket owner `{owner}`."
                )
            current = owner
        return False

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
