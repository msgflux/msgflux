import json
from dataclasses import asdict, is_dataclass
from typing import Any, Mapping, Optional, Sequence

_STEP_KEY_ORDER = (
    "reasoning",
    "thought",
    "tool",
    "tool_call",
    "command",
    "output",
    "exit_code",
    "status",
    "observation",
    "verification",
    "error",
    "notes",
)


def format_terminal_trajectory(
    *,
    steps: Optional[Sequence[Any]] = None,
    summary: Optional[Any] = None,
    final_answer: Optional[Any] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> str:
    """Format terminal-task evidence into a verifier-friendly candidate string."""
    return _format_trajectory(
        title="Terminal Trajectory",
        steps=steps,
        summary=summary,
        final_answer=final_answer,
        metadata=metadata,
    )


def format_swe_bench_trajectory(
    *,
    steps: Optional[Sequence[Any]] = None,
    summary: Optional[Any] = None,
    patch: Optional[Any] = None,
    final_answer: Optional[Any] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> str:
    """Format SWE-style trace evidence and final patch for verifier inputs."""
    extra_sections = []
    if patch is not None:
        extra_sections.append(_format_section("Final Patch", patch))

    return _format_trajectory(
        title="SWE-bench Trajectory",
        steps=steps,
        summary=summary,
        final_answer=final_answer,
        metadata=metadata,
        extra_sections=extra_sections,
    )


def _format_trajectory(
    *,
    title: str,
    steps: Optional[Sequence[Any]],
    summary: Optional[Any],
    final_answer: Optional[Any],
    metadata: Optional[Mapping[str, Any]],
    extra_sections: Optional[Sequence[str]] = None,
) -> str:
    sections = [title]

    if summary is not None:
        sections.append(_format_section("Summary", summary))

    if metadata:
        sections.append(_format_mapping_section("Metadata", metadata))

    if steps:
        sections.extend(_format_steps(steps))

    if extra_sections:
        sections.extend(section for section in extra_sections if section)

    if final_answer is not None:
        sections.append(_format_section("Final Answer", final_answer))

    if len(sections) == 1:
        raise ValueError(
            "Provide at least one of `summary`, `steps`, `patch`, or `final_answer`"
        )

    return "\n\n".join(sections)


def _format_steps(steps: Sequence[Any]) -> list[str]:
    rendered_steps = []
    for index, step in enumerate(steps, start=1):
        heading = f"Step {index}"
        if isinstance(step, Mapping):
            title = step.get("title")
            if title is not None:
                heading = f"{heading} — {_render_value(title)}"
            rendered_steps.append(
                _format_step_mapping(
                    heading, {k: v for k, v in step.items() if k != "title"}
                )
            )
            continue
        rendered_steps.append(_format_section(heading, step))
    return rendered_steps


def _format_step_mapping(heading: str, step: Mapping[str, Any]) -> str:
    ordered_keys = [key for key in _STEP_KEY_ORDER if key in step]
    remaining_keys = sorted(key for key in step if key not in _STEP_KEY_ORDER)
    body_sections = [
        _format_section(_humanize_key(key), step[key])
        for key in ordered_keys + remaining_keys
    ]
    if not body_sections:
        return heading
    return f"{heading}\n\n" + "\n\n".join(body_sections)


def _format_mapping_section(title: str, mapping: Mapping[str, Any]) -> str:
    sections = [
        _format_section(_humanize_key(key), value) for key, value in mapping.items()
    ]
    return f"{title}\n\n" + "\n\n".join(sections)


def _format_section(title: str, value: Any) -> str:
    return f"{title}:\n{_render_value(value)}"


def _render_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, Mapping):
        return json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True)
    if isinstance(value, (list, tuple)):
        return json.dumps(value, ensure_ascii=True, indent=2)
    return str(value)


def _humanize_key(key: str) -> str:
    return key.replace("_", " ").strip().title()
