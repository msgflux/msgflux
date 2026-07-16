"""Mutable ownership operations for tool buckets."""

from __future__ import annotations

from contextlib import suppress
from functools import partial
from typing import Any, Callable, Dict, Iterable, Mapping

from msgflux.tools.bucket_graph import ToolBucketGraph
from msgflux.tools.dataclasses import ToolMetadata
from msgflux.tools.registration import ToolRegistrationTransaction
from msgflux.tools.types import ToolBucket, split_hidden_annotations


class ToolBucketManager:
    """Manage bucket ownership over a library's canonical registry maps."""

    def __init__(
        self,
        tools: Mapping[str, Any],
        tool_configs: Dict[str, Dict[str, Any]],
        owners: Dict[str, str],
        metadata_factory: Callable[[Any], ToolMetadata],
    ) -> None:
        self._tools = tools
        self._tool_configs = tool_configs
        self._owners = owners
        self._metadata_factory = metadata_factory
        self.graph = ToolBucketGraph(tools, tool_configs, owners)

    def bind(self, bucket_name: str, bucket: ToolBucket) -> None:
        bucket._bind_tools(partial(self.get_child_metadata, bucket_name))

    @staticmethod
    def unbind(tool: Any, pending: Iterable[ToolMetadata] = ()) -> None:
        bucket = getattr(tool, "impl", tool)
        if isinstance(bucket, ToolBucket):
            bucket._unbind_tools(pending)

    def unbind_all(self) -> None:
        for tool in self._tools.values():
            self.unbind(tool)

    def get_child_metadata(self, bucket_name: str) -> Dict[str, ToolMetadata]:
        """Materialize metadata for one bucket's direct children."""
        return {
            name: self._metadata_factory(tool)
            for name, tool in self._tools.items()
            if self._owners.get(name) == bucket_name
        }

    def capture(
        self,
        bucket_name: str,
        metadata: ToolMetadata,
        *,
        transaction: ToolRegistrationTransaction | None = None,
    ) -> None:
        bucket = self.graph.require_bucket(bucket_name)
        bucket.validate_capture(metadata)
        if metadata.name not in self._tools:
            raise ValueError(f"Tool `{metadata.name}` is not registered.")
        if metadata.name in self._owners:
            raise ValueError(f"Tool `{metadata.name}` already has a bucket owner.")

        config = self._tool_configs[metadata.name]
        previous_exposed = bool(config.get("exposed", True))
        self._owners[metadata.name] = bucket_name
        config["exposed"] = False
        try:
            bucket.refresh()
            self.sync_presentation(bucket_name, bucket)
        except Exception:
            self._owners.pop(metadata.name, None)
            config["exposed"] = previous_exposed
            with suppress(Exception):
                bucket.refresh()
                self.sync_presentation(bucket_name, bucket)
            raise
        if transaction is not None:
            transaction.record(
                partial(
                    self._undo_capture,
                    bucket_name,
                    metadata.name,
                    exposed=previous_exposed,
                )
            )

    def _undo_capture(
        self,
        bucket_name: str,
        tool_name: str,
        *,
        exposed: bool,
    ) -> None:
        if self._owners.get(tool_name) != bucket_name:
            return
        self._owners.pop(tool_name)
        config = self._tool_configs.get(tool_name)
        if config is not None:
            config["exposed"] = exposed

    def release(
        self,
        bucket_name: str,
        tool_name: str,
        *,
        exposed: bool = True,
        refresh: bool = True,
    ) -> ToolMetadata:
        bucket = self.graph.require_bucket(bucket_name)
        if self._owners.get(tool_name) != bucket_name:
            raise ValueError(f"Tool `{tool_name}` is not captured by this bucket.")
        metadata = self._metadata_factory(self._tools[tool_name])
        config = self._tool_configs[tool_name]
        self._owners.pop(tool_name)
        config["exposed"] = exposed
        if not refresh:
            return metadata
        try:
            bucket.refresh()
            self.sync_presentation(bucket_name, bucket)
        except Exception:
            self._owners[tool_name] = bucket_name
            config["exposed"] = False
            with suppress(Exception):
                bucket.refresh()
                self.sync_presentation(bucket_name, bucket)
            raise
        return metadata

    def activate_on_demand(
        self,
        owner_name: str,
        tool_name: str,
        *,
        remove_owner: Callable[[str], None],
    ) -> str:
        """Promote one captured on-demand node without replacing its wrapper."""
        node = self.graph.find_node(tool_name)
        if node is None or node.parent != owner_name:
            raise ValueError(
                f"Tool `{tool_name}` is not directly captured by `{owner_name}`."
            )
        owner = self.graph.require_bucket(owner_name)
        config = self._tool_configs[tool_name]
        if not config.get("on_demand", False):
            raise ValueError(f"Tool `{tool_name}` is not configured for on-demand use.")

        promoted = self._metadata_factory(node.tool)
        promoted.tool_config = {**config, "on_demand": False, "exposed": True}
        target_names = self.graph.matching_buckets(promoted)
        if target_names:
            target = self.graph.require_bucket(target_names[0])
            target.validate_capture(promoted)
            if node.bucket is not None and (
                target_names[0] == tool_name
                or self.graph.is_descendant(tool_name, target_names[0])
            ):
                raise ValueError(
                    f"Activating `{tool_name}` through `{target_names[0]}` would "
                    "create a bucket capture cycle."
                )

        previous_exposed = bool(config.get("exposed", True))
        attached_target: str | None = None
        self._owners.pop(tool_name)
        config["on_demand"] = False
        config["exposed"] = True
        try:
            owner.refresh()
            self.sync_presentation(owner_name, owner)
            if target_names:
                attached_target = target_names[0]
                self.capture(attached_target, self._metadata_factory(node.tool))
            if owner.expose_captured_names and not owner.tools:
                remove_owner(owner_name)
        except Exception:
            if (
                attached_target is not None
                and self._owners.get(tool_name) == attached_target
            ):
                self._owners.pop(tool_name)
            config["on_demand"] = True
            config["exposed"] = previous_exposed
            self._owners[tool_name] = owner_name
            self.refresh_presentations()
            raise
        return tool_name

    def sync_presentation(self, bucket_name: str, bucket: ToolBucket) -> None:
        node = self.graph.find_node(bucket_name)
        if node is None:
            raise ValueError(f"The bucket tool `{bucket_name}` is not registered.")
        bucket_tool = node.tool
        if isinstance(getattr(bucket, "description", None), str):
            bucket_tool.set_description(bucket.description)
        annotations = getattr(bucket, "annotations", None)
        if isinstance(annotations, Mapping):
            public_annotations, _ = split_hidden_annotations(annotations)
            if getattr(bucket, "tool_config", {}).get("handle") is not None:
                public_annotations.pop("handle", None)
            bucket_tool.set_annotations(public_annotations)
        if hasattr(bucket, "usage_guidance"):
            bucket_tool.register_buffer("usage_guidance", bucket.usage_guidance)
        if node.parent is not None:
            parent = self.graph.require_bucket(node.parent)
            parent.refresh()
            self.sync_presentation(node.parent, parent)

    def refresh_presentations(self) -> None:
        """Rebuild surviving bucket metadata after a structural rollback."""
        with suppress(Exception):
            nodes = list(self.graph.iter_nodes())
            for node in reversed(nodes):
                if node.bucket is not None:
                    node.bucket.refresh()
            for node in reversed(nodes):
                if node.bucket is not None:
                    self.sync_presentation(node.name, node.bucket)
