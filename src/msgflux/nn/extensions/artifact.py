"""Immutable artifact registration and incremental output rendering."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Mapping

from msgflux.nn.extensions.base import AgentExtension
from msgflux.nn.hooks import Hook
from msgflux.nn.hooks.events import OutputContext

__all__ = [
    "Artifact",
    "ArtifactRegistry",
    "ArtifactReferenceRenderer",
    "ArtifactExtension",
]

_MARKER = re.compile(r"\{\{artifact:([A-Za-z0-9][A-Za-z0-9._:/-]{0,239})\}\}")
_PREFIX = "{{artifact:"
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,239}$")


@dataclass(frozen=True)
class Artifact:
    """An immutable, runtime-owned value addressable by a model output marker."""

    artifact_id: str
    content: str
    media_type: str = "text/plain"


class ArtifactRegistry:
    """Register immutable values under stable IDs; filesystem paths are rejected."""

    def __init__(self, artifacts: Mapping[str, Artifact] | None = None) -> None:
        self._artifacts: dict[str, Artifact] = {}
        for artifact_id, artifact in (artifacts or {}).items():
            if not _ID.fullmatch(artifact_id):
                raise ValueError(f"invalid artifact_id: {artifact_id!r}")
            if (
                not isinstance(artifact, Artifact)
                or artifact.artifact_id != artifact_id
            ):
                raise ValueError("artifacts must match their mapping IDs")
            self._artifacts[artifact_id] = artifact

    def register(
        self,
        content: str,
        *,
        artifact_id: str,
        media_type: str = "text/plain",
    ) -> Artifact:
        if (
            not artifact_id
            or not _ID.fullmatch(artifact_id)
        ):
            raise ValueError(
                "artifact_id must be a stable logical ID, not a filesystem path"
            )
        if not isinstance(content, str):
            raise TypeError("artifact content must be text")
        current = self._artifacts.get(artifact_id)
        artifact = Artifact(artifact_id, content, media_type)
        if current is not None and current != artifact:
            raise ValueError(
                "artifact_id is already registered with different content: "
                f"{artifact_id}"
            )
        self._artifacts[artifact_id] = artifact
        return artifact

    def get(self, artifact_id: str) -> Artifact | None:
        return self._artifacts.get(artifact_id)


class ArtifactReferenceRenderer:
    """Incrementally expand ``{{artifact:ID}}`` markers with bounded buffering."""

    def __init__(
        self, registry: ArtifactRegistry, *, max_marker_length: int = 256
    ) -> None:
        if max_marker_length < len(_PREFIX) + 3:
            raise ValueError("max_marker_length is too small for an artifact marker")
        self.registry = registry
        self.max_marker_length = max_marker_length
        self._buffer = ""

    def feed(self, chunk: str) -> str:  # noqa: C901
        if not isinstance(chunk, str):
            raise TypeError("artifact renderer accepts text chunks")
        self._buffer += chunk
        output = []
        while self._buffer:
            match = _MARKER.search(self._buffer)
            if match is not None:
                if len(match.group(0)) > self.max_marker_length:
                    output.append(self._buffer[: match.end()])
                    self._buffer = self._buffer[match.end() :]
                    continue
                prefix = self._buffer[: match.start()]
                if prefix.endswith("\\"):
                    output.append(prefix[:-1] + match.group(0))
                else:
                    output.append(prefix)
                    artifact = self.registry.get(match.group(1))
                    output.append(
                        artifact.content if artifact is not None else match.group(0)
                    )
                self._buffer = self._buffer[match.end() :]
                continue
            start = self._buffer.find(_PREFIX)
            if start < 0:
                keep = 0
                for size in range(1, min(len(_PREFIX) - 1, len(self._buffer)) + 1):
                    if self._buffer.endswith(_PREFIX[:size]):
                        keep = size
                if self._buffer.endswith("\\"):
                    keep = max(keep, 1)
                output.append(self._buffer[:-keep] if keep else self._buffer)
                self._buffer = self._buffer[-keep:] if keep else ""
                break
            if start:
                output.append(self._buffer[:start])
                self._buffer = self._buffer[start:]
                continue
            if len(self._buffer) > self.max_marker_length:
                output.append(self._buffer[0])
                self._buffer = self._buffer[1:]
                continue
            break
        return "".join(output)

    def finish(self) -> str:
        tail = self._buffer
        self._buffer = ""
        return tail

    def render(self, text: str) -> str:
        return self.feed(text) + self.finish()


class ArtifactExtension(AgentExtension):
    """Agent extension exposing an artifact registry and stream-safe renderer."""

    def __init__(
        self,
        registry: ArtifactRegistry | None = None,
        *,
        max_marker_length: int = 256,
    ) -> None:
        super().__init__("artifacts")
        self.registry = registry or ArtifactRegistry()
        self.max_marker_length = max_marker_length

    def hooks(self):
        def transform(context: OutputContext) -> OutputContext:
            if isinstance(context.output, str):
                return replace(
                    context,
                    output=ArtifactReferenceRenderer(
                        self.registry, max_marker_length=self.max_marker_length
                    ).render(context.output),
                )
            return context

        return (Hook(event="transform_output", handler=transform),)

    def create_output_transformer(self) -> ArtifactReferenceRenderer:
        return ArtifactReferenceRenderer(
            self.registry, max_marker_length=self.max_marker_length
        )
