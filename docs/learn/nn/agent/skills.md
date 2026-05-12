# Agent Skills

Agent Skills are folders with a `SKILL.md` file that package reusable
instructions, workflows, scripts, references, and assets for agents.

msgFlux follows the Agent Skills progressive disclosure model:

1. At startup, the agent discovers skill `name` and `description`.
2. The system prompt receives a compact skill catalog through the `agent_skills`
   template field.
3. When the model decides a skill is relevant, it calls `activate_skill(name)`
   to load the full `SKILL.md` body and resource list.

## Skill Directory

A skill is a directory containing `SKILL.md`:

```text
code-review/
├── SKILL.md
├── references/
│   └── checklist.md
└── scripts/
    └── inspect.py
```

Minimal `SKILL.md`:

```markdown
---
name: code-review
description: Review code changes for bugs, regressions, and missing tests.
---

# Code Review

Inspect correctness first, then tests and edge cases.
```

The required fields are:

- `name`: stable skill name.
- `description`: what the skill does and when to use it.

Optional fields such as `license`, `compatibility`, `metadata`, and
`allowed-tools` are parsed and stored by the runtime.

Skill visibility field:

- `discoverable`: keep the skill out of the initial system prompt catalog and
  make it available through `skill_search`. Defaults to `false`.

## Passing Skill Directories

Pass one directory:

```python
import msgflux as mf
import msgflux.nn as nn

agent = nn.Agent(
    name="developer_agent",
    model=mf.Model.chat_completion("openai/gpt-4.1-mini"),
    skills=".agents/skills",
)
```

Pass multiple directories:

```python
agent = nn.Agent(
    name="developer_agent",
    model=mf.Model.chat_completion("openai/gpt-4.1-mini"),
    skills=[
        ".agents/skills",
        ".codex/skills",
        "~/.agents/skills",
        "~/.codex/skills",
    ],
)
```

Pass glob patterns explicitly:

```python
agent = nn.Agent(
    name="developer_agent",
    model=mf.Model.chat_completion("openai/gpt-4.1-mini"),
    skills=[
        ".agents/skills",
        ".codex/skills",
        "*/skills",
    ],
)
```

Use the helper when you want common local locations explicitly:

```python
agent = nn.Agent(
    name="developer_agent",
    model=mf.Model.chat_completion("openai/gpt-4.1-mini"),
    skills=mf.default_skill_paths(),
)
```

msgFlux does not scan default skill paths implicitly. This avoids silently
loading instructions from a project or user directory. Pass the paths you want.

## Skill Config

For larger skill sets, pass a dict:

```python
agent = nn.Agent(
    name="developer_agent",
    model=mf.Model.chat_completion("openai/gpt-4.1-mini"),
    skills={
        "paths": [
            ".agents/skills",
            ".codex/skills",
            "*/skills",
        ],
        "catalog_limit": 20,
        "search_top_k": 5,
    },
)
```

Config keys:

- `paths`: directory, `SKILL.md` file, glob pattern, or list of those.
- `catalog_limit`: maximum number of non-discoverable skills included in the
  system prompt. Use `0` to keep the initial catalog empty.
- `search_top_k`: default number of results returned by `skill_search`.

## System Prompt Field

Skills are rendered by the system prompt template from the `agent_skills`
structured field.

The default template emits:

```xml
<agent_skills>
The following Agent Skills are available...
<available_skills>
  <skill>
    <name>code-review</name>
    <description>Review code changes...</description>
    <location>/repo/.agents/skills/code-review/SKILL.md</location>
  </skill>
</available_skills>
</agent_skills>
```

If you override `templates["system_prompt"]`, render `agent_skills` with Jinja
where you want the catalog to appear:

```python
agent = nn.Agent(
    name="developer_agent",
    model=model,
    skills=".agents/skills",
    templates={
        "system_prompt": """
        <system_note>
        {{ instructions }}
        {% if agent_skills %}
        <agent_skills>
        {% for skill in agent_skills %}
        <skill>
          <name>{{ skill.name }}</name>
          <description>{{ skill.description }}</description>
          <location>{{ skill.location }}</location>
        </skill>
        {% endfor %}
        </agent_skills>
        {% endif %}
        </system_note>
        """,
    },
)
```

## Activating A Skill

When skills exist, msgFlux registers an internal tool:

```python
activate_skill(name: str) -> str
```

The model calls it with the skill name. The tool returns wrapped content:

```xml
<skill_content name="code-review">
# Code Review

Inspect correctness first, then tests and edge cases.

Skill directory: /repo/.agents/skills/code-review
Relative paths in this skill are relative to the skill directory.
<skill_resources>
  <file>references/checklist.md</file>
  <file>scripts/inspect.py</file>
</skill_resources>
</skill_content>
```

Bundled resources are listed, not eagerly loaded. The skill instructions can
tell the agent when to read or execute those files with normal tools.

## Searching Skills

When at least one hidden discoverable skill exists, msgFlux registers another
internal tool:

```python
skill_search(query: str, top_k: int | None = None) -> str
```

A discoverable skill is intentionally omitted from the initial catalog and can
be found through `skill_search`.

```markdown
---
name: release-notes
description: Write concise release notes from merged changes.
discoverable: true
---

# Release Notes

Group changes by user-visible impact.
```

Search uses a small in-memory BM25 implementation over skill name, description,
and metadata. No external dependency is required.

If no hidden discoverable skill exists, `skill_search` is not registered.

## Runnable Example

Run the offline demo:

```bash
uv run python examples/agent_skills_demo.py
```

The example creates two temporary skill directories, passes both to an agent,
shows the generated skill catalog, and activates `code-review` through the
`activate_skill` tool.
