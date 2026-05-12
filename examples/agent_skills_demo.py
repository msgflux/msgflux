# /// script
# dependencies = []
# ///
# ruff: noqa: T201

from pathlib import Path
from tempfile import TemporaryDirectory

from msgflux import nn
from msgflux.models.response import ModelResponse
from msgflux.models.tool_call_agg import ToolCallAggregator
from msgflux.utils.msgspec import msgspec_dumps


def tool_call_response(tool_name: str, parameters: dict, *, call_id: str):
    response = ModelResponse()
    response.set_response_type("tool_call")
    agg = ToolCallAggregator()
    agg.process(0, call_id, tool_name, msgspec_dumps(parameters))
    response.add(agg)
    response.reasoning = None
    response.metadata = {}
    return response


def text_response(text: str):
    response = ModelResponse()
    response.set_response_type("text_generation")
    response.add(text)
    response.reasoning = None
    response.metadata = {}
    return response


class ScriptedModel:
    model_type = "chat_completion"

    def __init__(self):
        self.responses = [
            tool_call_response(
                "skill_search",
                {"query": "release notes"},
                call_id="call_skill_search",
            ),
            tool_call_response(
                "activate_skill",
                {"name": "code-review"},
                call_id="call_activate_skill",
            ),
            text_response("I loaded the code-review skill and will use its checklist."),
        ]

    def __call__(self, **_kwargs):
        if not self.responses:
            raise RuntimeError("Scripted model exhausted.")
        return self.responses.pop(0)

    async def acall(self, **kwargs):
        return self(**kwargs)


def write_skill(
    root: Path,
    name: str,
    description: str,
    body: str,
    *,
    discoverable: bool = False,
) -> None:
    skill_dir = root / name
    skill_dir.mkdir(parents=True)
    discoverable_line = ["discoverable: true"] if discoverable else []
    (skill_dir / "SKILL.md").write_text(
        "\n".join(
            [
                "---",
                f"name: {name}",
                f"description: {description}",
                *discoverable_line,
                "---",
                body,
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    with TemporaryDirectory() as tmp:
        project_skills = Path(tmp) / ".agents" / "skills"
        codex_skills = Path(tmp) / ".codex" / "skills"

        write_skill(
            project_skills,
            "code-review",
            "Review code changes for bugs, regressions, and missing tests.",
            "# Code Review\n\nInspect correctness first, then tests and edge cases.",
        )
        write_skill(
            codex_skills,
            "release-notes",
            "Write concise release notes from merged changes.",
            "# Release Notes\n\nGroup changes by user-visible impact.",
            discoverable=True,
        )

        agent = nn.Agent(
            name="developer_agent",
            model=ScriptedModel(),
            skills={
                "paths": [project_skills, codex_skills],
                "catalog_limit": 1,
                "search_top_k": 3,
            },
            instructions="Use Agent Skills when they match the user's request.",
        )

        print("Available skills:", agent.agent_skill_manager.names())
        print("\nSystem prompt excerpt:")
        print(agent.get_system_prompt())
        print("\nAgent response:")
        print(agent("Review this pull request."))


if __name__ == "__main__":
    main()
