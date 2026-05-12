from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence, Union

SkillPath = Union[str, Path]
SkillPaths = Union[SkillPath, Sequence[SkillPath]]


def default_skill_paths() -> list[Path]:
    """Return conventional local Agent Skill directories."""
    cwd = Path.cwd()
    home = Path.home()
    return [
        cwd / ".agents" / "skills",
        cwd / ".codex" / "skills",
        cwd / "codex" / "skills",
        home / ".agents" / "skills",
        home / ".codex" / "skills",
    ]


@dataclass(frozen=True)
class AgentSkill:
    """Discovered Agent Skill metadata and activation payload."""

    name: str
    description: str
    path: Path
    body: str
    license: Optional[str] = None
    compatibility: Optional[str] = None
    allowed_tools: Optional[str] = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    @property
    def directory(self) -> Path:
        return self.path.parent


def _strip_scalar(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _parse_frontmatter(frontmatter: str) -> dict[str, Any]:
    """Parse the small YAML subset used by SKILL.md frontmatter."""
    data: dict[str, Any] = {}
    current_map_key: str | None = None

    for raw_line in frontmatter.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue

        if raw_line.startswith((" ", "\t")):
            if current_map_key is None or ":" not in raw_line:
                continue
            key, value = raw_line.strip().split(":", 1)
            nested = data.setdefault(current_map_key, {})
            if isinstance(nested, dict):
                nested[key.strip()] = _strip_scalar(value)
            continue

        current_map_key = None
        if ":" not in raw_line:
            continue

        key, value = raw_line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not value:
            data[key] = {}
            current_map_key = key
        else:
            data[key] = _strip_scalar(value)

    return data


def parse_skill_file(path: SkillPath) -> AgentSkill:
    """Parse a SKILL.md file using the Agent Skills frontmatter format."""
    skill_path = Path(path).expanduser().resolve()
    text = skill_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError(f"`{skill_path}` must start with YAML frontmatter.")

    end_index = None
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            end_index = index
            break
    if end_index is None:
        raise ValueError(f"`{skill_path}` frontmatter is not closed.")

    frontmatter = "\n".join(lines[1:end_index])
    body = "\n".join(lines[end_index + 1 :]).strip()
    metadata = _parse_frontmatter(frontmatter)

    name = metadata.get("name")
    description = metadata.get("description")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"`{skill_path}` is missing required `name`.")
    if not isinstance(description, str) or not description.strip():
        raise ValueError(f"`{skill_path}` is missing required `description`.")

    skill_metadata = metadata.get("metadata", {})
    if not isinstance(skill_metadata, Mapping):
        skill_metadata = {}

    return AgentSkill(
        name=name.strip(),
        description=description.strip(),
        path=skill_path,
        body=body,
        license=metadata.get("license")
        if isinstance(metadata.get("license"), str)
        else None,
        compatibility=metadata.get("compatibility")
        if isinstance(metadata.get("compatibility"), str)
        else None,
        allowed_tools=metadata.get("allowed-tools")
        if isinstance(metadata.get("allowed-tools"), str)
        else None,
        metadata={str(k): str(v) for k, v in skill_metadata.items()},
    )


class AgentSkillManager:
    """Discover and activate Agent Skills from local directories."""

    def __init__(
        self,
        paths: Optional[SkillPaths] = None,
    ) -> None:
        self.paths = self._normalize_paths(paths)
        self.skills: dict[str, AgentSkill] = {}
        self.diagnostics: list[str] = []
        self.discover()

    def _normalize_paths(
        self,
        paths: Optional[SkillPaths],
    ) -> list[Path]:
        if paths is None:
            return []
        if isinstance(paths, bool):
            raise TypeError(
                "`skills` must be a path or list of paths. Use "
                "`msgflux.default_skill_paths()` to opt into conventional paths."
            )
        if isinstance(paths, (str, Path)):
            paths = [paths]
        return [Path(path).expanduser().resolve() for path in paths]

    def discover(self) -> None:
        """Discover skills under configured paths."""
        self.skills.clear()
        self.diagnostics.clear()
        for root in self.paths:
            for skill_file in self._iter_skill_files(root):
                try:
                    skill = parse_skill_file(skill_file)
                except Exception as exc:
                    self.diagnostics.append(f"{skill_file}: {exc}")
                    continue
                if skill.name in self.skills:
                    self.diagnostics.append(
                        f"{skill_file}: skill `{skill.name}` shadowed by "
                        f"`{self.skills[skill.name].path}`."
                    )
                    continue
                self.skills[skill.name] = skill

    def _iter_skill_files(self, root: Path) -> Iterable[Path]:
        if not root.exists():
            self.diagnostics.append(f"{root}: directory does not exist.")
            return
        if root.is_file():
            if root.name == "SKILL.md":
                yield root
            return
        direct_skill = root / "SKILL.md"
        if direct_skill.exists():
            yield direct_skill
            return
        for child in sorted(root.iterdir()):
            if child.is_dir():
                skill_file = child / "SKILL.md"
                if skill_file.exists():
                    yield skill_file

    def has_skills(self) -> bool:
        return bool(self.skills)

    def names(self) -> list[str]:
        return sorted(self.skills)

    def get(self, name: str) -> AgentSkill:
        try:
            return self.skills[name]
        except KeyError as exc:
            available = ", ".join(self.names()) or "none"
            raise ValueError(
                f"Unknown skill `{name}`. Available skills: {available}."
            ) from exc

    def render_catalog(self) -> str:
        if not self.skills:
            return ""

        entries = []
        for skill in sorted(self.skills.values(), key=lambda item: item.name):
            entries.append(
                "\n".join(
                    [
                        "  <skill>",
                        f"    <name>{skill.name}</name>",
                        f"    <description>{skill.description}</description>",
                        f"    <location>{skill.path}</location>",
                        "  </skill>",
                    ]
                )
            )
        return (
            "The following Agent Skills are available. Use them when the task "
            "matches a skill description. To activate a skill, call "
            "`activate_skill` with the skill name before following that skill's "
            "workflow. When a skill references relative paths, resolve them "
            "against the skill directory returned by `activate_skill`.\n"
            "<available_skills>\n" + "\n".join(entries) + "\n</available_skills>"
        )

    def activate(self, name: str) -> str:
        skill = self.get(name)
        resource_lines = self._resource_lines(skill)
        resources = (
            "\n<skill_resources>\n" + "\n".join(resource_lines) + "\n</skill_resources>"
            if resource_lines
            else ""
        )
        return (
            f'<skill_content name="{skill.name}">\n'
            f"{skill.body}\n\n"
            f"Skill directory: {skill.directory}\n"
            "Relative paths in this skill are relative to the skill directory."
            f"{resources}\n"
            "</skill_content>"
        )

    def _resource_lines(self, skill: AgentSkill, *, limit: int = 50) -> list[str]:
        lines = []
        for path in sorted(skill.directory.rglob("*")):
            if not path.is_file() or path.name == "SKILL.md":
                continue
            relative = path.relative_to(skill.directory)
            lines.append(f"  <file>{relative}</file>")
            if len(lines) >= limit:
                lines.append("  <file>...</file>")
                break
        return lines
