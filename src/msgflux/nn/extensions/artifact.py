"""Immutable artifact registration and incremental output rendering."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from threading import RLock
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
        self._lock = RLock()
        self._artifacts: dict[str, Artifact] = {}
        for artifact_id, artifact in (artifacts or {}).items():
            if not _ID.fullmatch(artifact_id):
                raise ValueError(f"invalid artifact_id: {artifact_id!r}")
            if (
                not isinstance(artifact, Artifact)
                or artifact.artifact_id != artifact_id
            ):
                raise ValueError("artifacts must match their mapping IDs")
            self.register(
                artifact.content,
                artifact_id=artifact_id,
                media_type=artifact.media_type,
            )

    def register(
        self,
        content: str,
        *,
        artifact_id: str,
        media_type: str = "text/plain",
    ) -> Artifact:
        if not isinstance(artifact_id, str) or not _ID.fullmatch(artifact_id):
            raise ValueError(
                "artifact_id must be a stable logical ID, not a filesystem path"
            )
        if not isinstance(content, str):
            raise TypeError("artifact content must be text")
        artifact = Artifact(artifact_id, content, media_type)
        with self._lock:
            current = self._artifacts.get(artifact_id)
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
        if (
            isinstance(max_marker_length, bool)
            or not isinstance(max_marker_length, int)
            or max_marker_length < len(_PREFIX) + 3
        ):
            raise ValueError("max_marker_length is too small for an artifact marker")
        self.registry = registry
        self.max_marker_length = max_marker_length
        self._buffer = ""

    def feed(self, chunk: str) -> str:
        if not isinstance(chunk, str):
            raise TypeError("artifact renderer accepts text chunks")
        output = []
        # Only a possible marker (and its optional escape) survives between
        # characters. Chunk boundaries never affect token recognition.
        for char in chunk:
            self._buffer += char
            self._flush_prefix(output)
        return "".join(output)

    def _flush_prefix(self, output: list[str]) -> None:
        while self._buffer:
            escaped = self._buffer.startswith("\\")
            candidate = self._buffer[1:] if escaped else self._buffer
            if _PREFIX.startswith(candidate):
                return
            if (
                candidate.startswith(_PREFIX)
                and len(candidate) <= self.max_marker_length
            ):
                match = _MARKER.fullmatch(candidate)
                if match is not None:
                    artifact = None if escaped else self.registry.get(match.group(1))
                    output.append(
                        artifact.content if artifact is not None else candidate
                    )
                    self._buffer = ""
                    return
                if not candidate.endswith("}}"):
                    return
            output.append(self._buffer[0])
            self._buffer = self._buffer[1:]

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
