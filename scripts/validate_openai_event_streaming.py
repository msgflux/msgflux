"""Live validation for async execution events through OpenAI Responses.

Run from the repository root:

    uv run python scripts/validate_openai_event_streaming.py

The script loads ``OPENAI_API_KEY`` from ``.env`` and performs paid API calls.
It never prints credentials or request headers.
"""

# ruff: noqa: S101, T201

from __future__ import annotations

import argparse
import asyncio
import os
from collections import Counter
from collections.abc import AsyncIterator, Callable
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import msgflux as mf
from msgflux.chat_messages import ChatMessages
from msgflux.nn import Agent
from msgflux.nn.hooks import Hook
from msgflux.runtime import ExecutionEvent
from msgflux.runtime.events import EventType
from msgflux.tools.builtin import AgentTool

DEFAULT_MODEL = "openai/gpt-5.6-luna"


def make_model(model_path: str):
    return mf.Model.chat_completion(
        model_path,
        api_mode="responses",
        max_tokens=500,
        reasoning_effort="low",
        store=False,
    )


async def collect(events: AsyncIterator[ExecutionEvent]) -> list[ExecutionEvent]:
    collected = []
    async for event in events:
        collected.append(event)
    return collected


def require_event(events: list[ExecutionEvent], event_type: str) -> ExecutionEvent:
    try:
        return next(event for event in events if event.type == event_type)
    except StopIteration as exc:
        seen = sorted({event.type for event in events})
        raise AssertionError(f"Missing {event_type!r}; received {seen}") from exc


def assert_source_envelope(events: list[ExecutionEvent]) -> None:
    missing = [event for event in events if not event.source_path]
    assert not missing, f"Events without source identity: {missing}"
    assert all(event.run_id for event in events)


def assert_model_metrics(events: list[ExecutionEvent]) -> None:
    requests = [event for event in events if event.type == EventType.MODEL_REQUEST]
    responses = [event for event in events if event.type == EventType.MODEL_RESPONSE]
    assert len(responses) == len(requests)
    for response in responses:
        assert response.data["usage"]["input_tokens"] > 0
        assert response.data["usage"]["output_tokens"] > 0
        assert response.data["timing"]["source"] == "provider"
        assert response.data["timing"]["latency_ms"] >= 0


def print_summary(name: str, events: list[ExecutionEvent]) -> None:
    counts = Counter(event.type for event in events)
    sources = sorted({" > ".join(event.source_path) for event in events})
    print(f"\n[{name}] {len(events)} events")
    print("types:", dict(sorted(counts.items())))
    print("sources:")
    for source_path in sources:
        print(f"  - {source_path}")


async def validate_basic(model_path: str) -> None:
    agent = Agent(
        name="event_probe",
        model=make_model(model_path),
        system_prompt="Reply with exactly EVENT_STREAM_OK.",
        config={"stream": True},
    )

    events = await collect(agent.stream_events("Return the validation marker."))

    assert_source_envelope(events)
    assert_model_metrics(events)
    require_event(events, EventType.MODEL_REQUEST)
    model_response = require_event(events, EventType.MODEL_RESPONSE)
    require_event(events, EventType.MESSAGE_DELTA)
    message_end = require_event(events, EventType.MESSAGE_END)
    terminal = require_event(events, EventType.RUN_END)
    run_start = require_event(events, EventType.RUN_START)
    assert "EVENT_STREAM_OK" in str(message_end.data["content"])
    assert model_response.data["usage"]["input_tokens"] > 0
    assert model_response.data["usage"]["output_tokens"] > 0
    assert model_response.data["timing"]["source"] == "provider"
    assert model_response.data["timing"]["latency_ms"] >= 0
    assert model_response.data["timing"]["ttft_ms"] >= 0
    assert events.index(model_response) > max(
        index
        for index, event in enumerate(events)
        if event.type == EventType.MESSAGE_DELTA
    )
    assert events.index(model_response) < events.index(message_end)
    assert terminal.data == {"outcome": "completed"}
    assert run_start.data["namespace"] == "event_probe"
    print_summary("basic", events)


async def validate_reasoning_order(model_path: str) -> None:
    agent = Agent(
        name="reasoning_probe",
        model=make_model(model_path),
        system_prompt="Solve the arithmetic carefully, then answer briefly.",
        config={"stream": True},
    )

    events = await collect(
        agent.stream_events(
            "A warehouse has 27 racks with 43 bins each. How many bins are there?"
        )
    )

    assert_source_envelope(events)
    assert_model_metrics(events)
    first_summary = require_event(events, EventType.REASONING_SUMMARY_DELTA)
    first_message_delta = require_event(events, EventType.MESSAGE_DELTA)
    message_end = require_event(events, EventType.MESSAGE_END)
    assert events.index(first_summary) < events.index(first_message_delta)
    assert "1,161" in str(message_end.data["content"]) or "1161" in str(
        message_end.data["content"]
    )
    print_summary("reasoning_order", events)


async def validate_tool_loop(model_path: str) -> None:
    calls: list[str] = []

    def lookup_incident(code: str) -> str:
        """Return the current incident status for an exact incident code."""
        calls.append(code)
        return f"{code}: scanner recovered; reconciliation pending"

    agent = Agent(
        name="tool_probe",
        model=make_model(model_path),
        tools=[lookup_incident],
        system_prompt=(
            "For incident questions, call lookup_incident exactly once before "
            "answering. Include the returned status in the final answer."
        ),
        config={"stream": True, "max_tool_turns": 4},
    )

    events = await collect(
        agent.stream_events("What is the status of incident INC-42?")
    )

    assert_source_envelope(events)
    assert_model_metrics(events)
    tool_start = require_event(events, EventType.TOOL_START)
    tool_end = require_event(events, EventType.TOOL_END)
    message_end = require_event(events, EventType.MESSAGE_END)
    terminal = require_event(events, EventType.RUN_END)
    assert calls == ["INC-42"]
    assert tool_start.data["tool_name"] == "lookup_incident"
    assert tool_start.source_path[-1] == "tool:lookup_incident"
    assert tool_end.data["error"] is None
    assert "reconciliation" in str(message_end.data["content"]).lower()
    assert terminal.data == {"outcome": "completed"}
    print_summary("tool_loop", events)


async def validate_output_transform(model_path: str) -> None:
    reference = "artifact://incident-report"
    expanded = "Incident report: scanner recovered; reconciliation pending."

    def expand_reference(ctx):
        return replace(ctx, output=ctx.output.replace(reference, expanded))

    history = ChatMessages()
    agent = Agent(
        name="transform_probe",
        model=make_model(model_path),
        system_prompt=f"Reply with exactly {reference} and nothing else.",
        hooks=[Hook(event="transform_output", handler=expand_reference)],
        config={"stream": True},
    )

    events = await collect(
        agent.stream_events("Return the incident report reference.", messages=history)
    )

    assert_source_envelope(events)
    assert_model_metrics(events)
    assert not [event for event in events if event.type == EventType.MESSAGE_DELTA]
    message_start = require_event(events, EventType.MESSAGE_START)
    message_end = require_event(events, EventType.MESSAGE_END)
    terminal = require_event(events, EventType.RUN_END)
    assistant_messages = [
        item for item in history.to_chatml() if item.get("role") == "assistant"
    ]

    assert message_start.data["buffered"] is True
    assert reference not in str(message_end.data["content"])
    assert expanded in str(message_end.data["content"])
    assert terminal.data == {"outcome": "completed"}
    assert reference in assistant_messages[-1]["content"]
    assert expanded not in assistant_messages[-1]["content"]
    print_summary("output_transform", events)


async def validate_nested_agent(model_path: str) -> None:
    reviewer = Agent(
        name="reviewer",
        description="Reviews one short statement.",
        model=make_model(model_path),
        system_prompt="Review the statement briefly and include REVIEW_COMPLETE.",
    )
    root = Agent(
        name="root_probe",
        model=make_model(model_path),
        tools=[AgentTool(), reviewer],
        system_prompt=(
            "Delegate the requested review to the reviewer agent exactly once. "
            "After it returns, answer with its conclusion."
        ),
        config={"stream": True, "max_tool_turns": 4},
    )

    events = await collect(
        root.stream_events("Review this statement: inventory is reconciled.")
    )

    assert_source_envelope(events)
    assert_model_metrics(events)
    nested_start = next(
        event
        for event in events
        if event.type == EventType.RUN_START
        and event.source_path[-1] == "agent:reviewer"
    )
    nested_end = next(
        event
        for event in events
        if event.type == EventType.RUN_END and event.source_path[-1] == "agent:reviewer"
    )
    child_request = next(
        event
        for event in events
        if event.type == EventType.MODEL_REQUEST
        and event.source_path[-1] == "agent:reviewer"
    )
    nested_message = next(
        event
        for event in events
        if event.type == EventType.MESSAGE_END
        and event.source_path[-1] == "agent:reviewer"
    )
    agent_tool_start = next(
        event
        for event in events
        if event.type == EventType.TOOL_START and event.source_path[-1] == "tool:agent"
    )
    reviewer_tool_start = next(
        event
        for event in events
        if event.type == EventType.TOOL_START
        and event.source_path[-1] == "tool:reviewer"
    )
    root_start = events[0]
    root_message = [
        event
        for event in events
        if event.type == EventType.MESSAGE_END
        and event.source_path == ("agent:root_probe",)
    ][-1]
    terminal = [
        event
        for event in events
        if event.type == EventType.RUN_END
        and event.source_path == ("agent:root_probe",)
    ][-1]

    print_summary("nested_agent", events)
    assert nested_start.data["namespace"] == "reviewer"
    assert nested_start.run_id == root_start.run_id
    assert "parent_run_id" not in nested_start.data
    assert agent_tool_start.data["arguments"]["name"] == "reviewer"
    assert agent_tool_start.data["arguments"]["message"]
    assert reviewer_tool_start.data["arguments"] == {
        "message": agent_tool_start.data["arguments"]["message"]
    }, reviewer_tool_start.data["arguments"]
    assert nested_end.data == {"outcome": "completed"}
    assert "REVIEW_COMPLETE" in str(nested_message.data["content"])
    assert child_request.source_path[0] == "agent:root_probe"
    assert "tool:agent" in child_request.source_path
    assert str(root_message.data["content"]).strip()
    assert terminal.data == {"outcome": "completed"}


async def validate_skills_extension(model_path: str) -> None:
    with TemporaryDirectory() as temporary_directory:
        skill_directory = Path(temporary_directory) / "incident-validation"
        skill_directory.mkdir()
        (skill_directory / "SKILL.md").write_text(
            """---
name: incident-validation
description: Return the live extension validation marker.
---

# Validation

After loading this skill, reply with exactly EXTENSION_SKILL_OK.
""",
            encoding="utf-8",
        )
        agent = Agent(
            name="extension_probe",
            model=make_model(model_path),
            extensions=[mf.SkillsExtension({"paths": [skill_directory]})],
            system_prompt=(
                "Load incident-validation with the skill tool, then follow its "
                "instructions exactly."
            ),
            config={"stream": True, "max_tool_turns": 4},
        )

        events = await collect(
            agent.stream_events("Run the extension validation workflow.")
        )

    skill_call = next(
        event
        for event in events
        if event.type == EventType.TOOL_START and event.data["tool_name"] == "skill"
    )
    message_end = [event for event in events if event.type == EventType.MESSAGE_END][-1]
    assert skill_call.data["arguments"] == {"name": "incident-validation"}
    assert "EXTENSION_SKILL_OK" in str(message_end.data["content"])
    print_summary("skills_extension", events)


SCENARIOS: dict[str, Callable[[str], Any]] = {
    "basic": validate_basic,
    "reasoning": validate_reasoning_order,
    "tool": validate_tool_loop,
    "transform": validate_output_transform,
    "nested": validate_nested_agent,
    "skills": validate_skills_extension,
}


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"msgFlux model path (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--scenario",
        choices=["all", *SCENARIOS],
        default="all",
    )
    args = parser.parse_args()

    mf.load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY was not found in the environment or .env")

    selected = (
        SCENARIOS
        if args.scenario == "all"
        else {args.scenario: SCENARIOS[args.scenario]}
    )
    for scenario in selected.values():
        await scenario(args.model)
    print("\nAll selected live event-stream validations passed.")


if __name__ == "__main__":
    asyncio.run(main())
