import pytest

from msgflux.nn import ArtifactExtension, ArtifactReferenceRenderer, ArtifactRegistry


def test_artifact_renderer_handles_split_markers_and_missing_values():
    registry = ArtifactRegistry()
    registry.register("REPORT", artifact_id="report-1")
    renderer = ArtifactReferenceRenderer(registry)

    assert renderer.feed("before {{arti") == "before "
    assert renderer.feed("fact:report-1}} after") == "REPORT after"
    assert renderer.render(" {{artifact:unknown}}") == " {{artifact:unknown}}"


def test_artifact_renderer_escapes_markers_and_bounds_buffer():
    registry = ArtifactRegistry()
    registry.register("x", artifact_id="x")
    renderer = ArtifactReferenceRenderer(registry, max_marker_length=32)

    assert renderer.render(r"\{{artifact:x}}") == "{{artifact:x}}"
    renderer.feed("{{artifact:" + "a" * 100)
    assert len(renderer._buffer) <= 32


def test_registry_rejects_paths_and_mutation():
    registry = ArtifactRegistry()
    registry.register("one", artifact_id="stable")
    with pytest.raises(ValueError):
        registry.register("two", artifact_id="stable")
    with pytest.raises(ValueError):
        registry.register("x", artifact_id="../secret")


def test_extension_exposes_incremental_renderer():
    extension = ArtifactExtension()
    extension.registry.register("value", artifact_id="v")
    renderer = extension.create_output_transformer()
    assert renderer.feed("{{artifact:v}}") == "value"
