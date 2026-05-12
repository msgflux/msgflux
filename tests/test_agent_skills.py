import pytest

import msgflux as mf
from msgflux.models.response import ModelResponse
from msgflux.models.tool_call_agg import ToolCallAggregator
from msgflux.nn import Agent
from msgflux.runtime.skills import AgentSkillManager, parse_skill_file
from msgflux.utils.msgspec import msgspec_dumps


def _write_skill(
    root,
    name="pdf-processing",
    description=None,
    body=None,
    *,
    discoverable=None,
):
    skill_dir = root / name
    skill_dir.mkdir(parents=True)
    lines = [
        "---",
        f"name: {name}",
        "description: "
        + (description or "Extract PDF text and tables. Use when handling PDF files."),
    ]
    if discoverable is not None:
        lines.append(f"discoverable: {str(discoverable).lower()}")
    lines.extend(
        [
            "metadata:",
            "  owner: docs-team",
            "---",
            body or "# PDF Processing\n\nFollow the PDF workflow.",
        ]
    )
    (skill_dir / "SKILL.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    return skill_dir


def _tool_call_response(tool_name: str, parameters: dict, *, call_id: str):
    response = ModelResponse()
    response.set_response_type("tool_call")
    agg = ToolCallAggregator()
    agg.process(0, call_id, tool_name, msgspec_dumps(parameters))
    response.add(agg)
    response.reasoning = None
    response.metadata = {}
    return response


def _text_response(text: str):
    response = ModelResponse()
    response.set_response_type("text_generation")
    response.add(text)
    response.reasoning = None
    response.metadata = {}
    return response


class _ScriptedModel:
    model_type = "chat_completion"

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if not self._responses:
            raise AssertionError("Scripted model exhausted.")
        return self._responses.pop(0)

    async def acall(self, **kwargs):
        return self(**kwargs)


def test_parse_skill_file_reads_frontmatter_and_body(tmp_path):
    skill_dir = _write_skill(tmp_path)

    skill = parse_skill_file(skill_dir / "SKILL.md")

    assert skill.name == "pdf-processing"
    assert skill.description.startswith("Extract PDF")
    assert skill.metadata == {"owner": "docs-team"}
    assert skill.discoverable is False
    assert "Follow the PDF workflow" in skill.body


def test_agent_skill_manager_accepts_multiple_directories(tmp_path):
    project_skills = tmp_path / "project" / ".agents" / "skills"
    codex_skills = tmp_path / "project" / ".codex" / "skills"
    _write_skill(project_skills, name="code-review")
    _write_skill(codex_skills, name="slides")

    manager = AgentSkillManager([project_skills, codex_skills])

    assert manager.names() == ["code-review", "slides"]


def test_agent_skill_manager_expands_globs(tmp_path, monkeypatch):
    project = tmp_path / "project"
    _write_skill(project / "pkg_a" / "skills", name="alpha")
    _write_skill(project / "pkg_b" / "skills", name="beta")
    monkeypatch.chdir(project)

    manager = AgentSkillManager("*/skills")

    assert manager.names() == ["alpha", "beta"]


def test_agent_skill_catalog_is_rendered_in_system_prompt(tmp_path):
    skills_root = tmp_path / ".agents" / "skills"
    _write_skill(skills_root, name="code-review")

    agent = Agent(name="agent", model=_ScriptedModel([]), skills=[skills_root])
    system_prompt = agent.get_system_prompt()

    assert "<agent_skills>" in system_prompt
    assert "<available_skills>" in system_prompt
    assert "<name>code-review</name>" in system_prompt
    assert "activate_skill" in system_prompt


def test_agent_registers_builtin_skill_tools(tmp_path):
    agent_without_skills = Agent(name="agent", model=_ScriptedModel([]))
    assert "activate_skill" not in agent_without_skills.tool_library.library

    skills_root = tmp_path / ".agents" / "skills"
    _write_skill(skills_root, name="code-review")
    _write_skill(
        skills_root,
        name="release-notes",
        description="Write release notes",
        discoverable=True,
    )
    agent_with_skills = Agent(
        name="agent", model=_ScriptedModel([]), skills=skills_root
    )

    assert "activate_skill" in agent_with_skills.tool_library.library
    assert "skill_search" in agent_with_skills.tool_library.library
    assert (
        agent_with_skills.tool_library.library["activate_skill"].display_name == "Skill"
    )
    assert (
        agent_with_skills.tool_library.library["skill_search"].display_name
        == "Skill Search"
    )
    assert (
        agent_with_skills.tool_library.library["activate_skill"].description
        == "Activate an Agent Skill and return its full instructions."
    )
    assert (
        agent_with_skills.tool_library.library["skill_search"].description
        == "Search discoverable Agent Skills by name and description."
    )


def test_skill_search_is_not_registered_without_hidden_discoverable_skills(tmp_path):
    skills_root = tmp_path / ".agents" / "skills"
    _write_skill(skills_root, name="code-review")
    agent = Agent(name="agent", model=_ScriptedModel([]), skills=skills_root)

    assert "activate_skill" in agent.tool_library.library
    assert "skill_search" not in agent.tool_library.library


def test_activate_skill_returns_wrapped_content_and_resources(tmp_path):
    skills_root = tmp_path / ".agents" / "skills"
    skill_dir = _write_skill(skills_root, name="code-review")
    (skill_dir / "references").mkdir()
    (skill_dir / "references" / "checklist.md").write_text(
        "Review checklist",
        encoding="utf-8",
    )

    manager = AgentSkillManager(skills_root)
    content = manager.activate("code-review")

    assert '<skill_content name="code-review">' in content
    assert "Follow the PDF workflow" in content
    assert "Skill directory:" in content
    assert "<file>references/checklist.md</file>" in content


def test_agent_can_activate_skill_through_tool_call(tmp_path):
    skills_root = tmp_path / ".agents" / "skills"
    _write_skill(skills_root, name="code-review", body="# Code Review\n\nFind bugs.")
    model = _ScriptedModel(
        [
            _tool_call_response(
                "activate_skill",
                {"name": "code-review"},
                call_id="call_1",
            ),
            _text_response("Skill loaded."),
        ]
    )
    agent = Agent(name="agent", model=model, skills=skills_root)

    result = agent("Review this change.")

    assert result == "Skill loaded."
    assert len(model.calls) == 2
    messages = model.calls[1]["messages"]
    assert any(
        message.get("role") == "tool" and "Find bugs." in message.get("content", "")
        for message in messages
    )


def test_agent_can_search_hidden_discoverable_skills(tmp_path):
    skills_root = tmp_path / ".agents" / "skills"
    _write_skill(skills_root, name="visible-review")
    _write_skill(
        skills_root,
        name="hidden-release-notes",
        description="Write concise release notes from merged changes.",
        discoverable=True,
    )
    model = _ScriptedModel(
        [
            _tool_call_response(
                "skill_search",
                {"query": "release notes"},
                call_id="call_1",
            ),
            _text_response("Found skill."),
        ]
    )
    agent = Agent(name="agent", model=model, skills=skills_root)

    result = agent("Find a skill for release notes.")

    assert result == "Found skill."
    messages = model.calls[1]["messages"]
    assert any(
        message.get("role") == "tool"
        and "hidden-release-notes" in message.get("content", "")
        for message in messages
    )


def test_discoverable_skills_are_hidden_from_catalog_and_enable_search(tmp_path):
    skills_root = tmp_path / ".agents" / "skills"
    _write_skill(skills_root, name="alpha")
    _write_skill(skills_root, name="beta", discoverable=True)

    agent = Agent(
        name="agent",
        model=_ScriptedModel([]),
        skills=skills_root,
    )
    system_prompt = agent.get_system_prompt()

    assert "<name>alpha</name>" in system_prompt
    assert "<name>beta</name>" not in system_prompt
    assert "skill_search" in system_prompt
    assert "skill_search" in agent.tool_library.library


def test_skills_true_requires_explicit_default_skill_paths():
    with pytest.raises(TypeError, match="default_skill_paths"):
        AgentSkillManager(True)


def test_skills_config_validates_limits(tmp_path):
    with pytest.raises(ValueError, match="catalog_limit"):
        AgentSkillManager({"paths": tmp_path, "catalog_limit": -1})

    with pytest.raises(ValueError, match="search_top_k"):
        AgentSkillManager({"paths": tmp_path, "search_top_k": 0})


def test_default_skill_paths_helper_returns_common_locations():
    paths = [str(path) for path in mf.default_skill_paths()]

    assert any(".agents/skills" in path for path in paths)
    assert any(".codex/skills" in path for path in paths)
