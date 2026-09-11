# Agent Skills

`SkillsExtension` discovers reusable `SKILL.md` workflows and exposes them with
progressive disclosure. The model initially sees names and descriptions; it
loads full instructions only when it calls `skill(name)`.

## Create A Skill

A skill is a directory containing `SKILL.md`:

```text
code-review/
├── SKILL.md
├── references/
│   └── checklist.md
└── scripts/
    └── inspect.py
```

```markdown
---
name: code-review
description: Review changes for bugs, regressions, and missing tests.
---

# Code Review

Inspect correctness first, then tests and edge cases.
```

`name` must use lowercase letters, numbers, and single hyphens and is limited
to 64 characters. `description` is required and limited to 1024 characters.
The optional `license`, `compatibility`, and `metadata` fields are retained.

Set `include_in_prompt: false` in the frontmatter when a skill should remain
discoverable but its name and description should be omitted from the initial
prompt.

## Install Skills

```python
import msgflux as mf
import msgflux.nn as nn

agent = nn.Agent(
    name="developer",
    model=mf.Model.chat_completion("openai/gpt-5.6-luna"),
    extensions=[
        mf.SkillsExtension(
            {
                "paths": [".agents/skills", ".codex/skills"],
                "catalog_limit": 20,
                "search_top_k": 5,
            }
        )
    ],
)
```

msgFlux does not scan conventional directories implicitly. Pass explicit paths
or opt in with `mf.default_skill_paths()`.

`skills={...}` on `Agent` remains a silent compatibility alias for
`extensions=[SkillsExtension(...)]` while the extension API stabilizes. New
code should use the extension directly.

## Configuration

| Key | Behavior |
| --- | --- |
| `paths` | A directory, `SKILL.md`, glob, or list of those values. |
| `allow` | Keep only the named skill or skills. Mutually exclusive with `block`. |
| `block` | Exclude the named skill or skills. |
| `preload` | Put selected full instructions in the initial system prompt. |
| `defer_loading` | When `true` (default), expose non-preloaded skills through `skill`. When `false`, preload all skills. |
| `catalog_limit` | Maximum descriptions in the initial prompt; `0` keeps it empty. |
| `discovery` | `"tool"` (default), a Markdown path, or `None`. Chooses how skills omitted from the prompt are discovered. |
| `search_top_k` | Default maximum results from `skill_search`. |

Preload stable instructions that every request needs:

```python
skills = mf.SkillsExtension(
    {
        "paths": ".agents/skills",
        "allow": ["code-review", "release-notes"],
        "preload": "code-review",
    }
)
```

To avoid all runtime skill tools, set `defer_loading=False`. Every discovered
skill that remains after `allow` or `block` is then inserted in the system
prompt.

## Runtime Tools

When deferred skills exist, the extension contributes one loader:

```text
skill(name: str) -> str
```

It returns only the validated skill body, wrapped as `skill_content`. The
frontmatter is not repeated. When the directory contains related files, the
result also identifies the skill directory so relative references work:

```xml
<skill_content name="code-review">
# Code Review

Inspect correctness first, then tests and edge cases.

Skill directory: /repo/.agents/skills/code-review
Relative paths in this skill are relative to the skill directory.
</skill_content>
```

The result enters the conversation as a normal tool result, so it remains
visible for the rest of the trajectory.

With `discovery="tool"`, the extension contributes `skill_search` when the
prompt omits any deferred skills:

```text
skill_search(query: str, top_k: int | None = None) -> str
```

Search uses the local BM25 retriever over skill names, descriptions, and
metadata.

### Filesystem Index

Pass a Markdown path to keep the complete discovery catalog out of the prompt
and avoid registering `skill_search`:

```python
from pathlib import Path

skills = mf.SkillsExtension(
    {
        "paths": ".agents/skills",
        "discovery": Path(".msgflux/skills/index.md"),
    }
)
```

On registration, the extension creates or refreshes the parent directories and
`index.md`. The deterministic file contains each deferred skill's name and
description. The system prompt includes only its resolved location and tells
the model to search the index with an available filesystem tool, then call
`skill(name)` to load the selected workflow. The extension does not repeat
frontmatter in the index or expose the full skill body there.

Use `discovery=None` only when the application supplies another discovery
mechanism. Deferred skills still have the `skill` loader, but omitted names
will not be discoverable through msgFlux.

## Remove Skills

Because skills are an extension, their prompt section and tools have one owner:

```python
handle = agent.register_extension(
    "skills",
    mf.SkillsExtension({"paths": ".agents/skills"}),
)

handle.remove()
```

New runs stop seeing the catalog and tools immediately. Runs already in flight
retain their starting snapshot, which keeps concurrent threads isolated.
