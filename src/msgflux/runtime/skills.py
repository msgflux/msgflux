from dataclasses import dataclass, field
from glob import glob
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence, Union

import yaml

from msgflux.data.retrievers.providers.bm25 import BM25LexicalRetriever

SkillPath = Union[str, Path]
SkillPaths = Union[SkillPath, Sequence[SkillPath]]
SkillsConfig = Mapping[str, Any]


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
    catalog: bool = True
    metadata: Mapping[str, str] = field(default_factory=dict)

    @property
    def directory(self) -> Path:
        return self.path.parent


def _parse_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1", "on"}:
            return True
        if normalized in {"false", "no", "0", "off"}:
            return False
    return default


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
    metadata = yaml.safe_load(frontmatter) or {}
    if not isinstance(metadata, Mapping):
        raise ValueError(f"`{skill_path}` frontmatter must be a YAML mapping.")

    name = metadata.get("name")
    description = metadata.get("description")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"`{skill_path}` is missing required `name`.")
    if not isinstance(description, str) or not description.strip():
        raise ValueError(f"`{skill_path}` is missing required `description`.")

    skill_metadata = metadata.get("metadata", {})
    if not isinstance(skill_metadata, Mapping):
        skill_metadata = {}

    catalog = metadata.get("catalog")
    catalog = _parse_bool(catalog, default=True)

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
        catalog=catalog,
        metadata={str(k): str(v) for k, v in skill_metadata.items()},
    )


class AgentSkillManager:
    """Discover and activate Agent Skills from local directories."""

    def __init__(
        self,
        config: Optional[SkillsConfig] = None,
    ) -> None:
        config = self._normalize_config(config)
        self.paths = self._normalize_paths(config.get("paths"))
        self.catalog_limit = config["catalog_limit"]
        self.search_top_k = config["search_top_k"]
        self.skills: dict[str, AgentSkill] = {}
        self.diagnostics: list[str] = []
        self.discover()

    def _normalize_config(self, config: Optional[SkillsConfig]) -> dict[str, Any]:
        if config is None:
            return {"paths": None, "catalog_limit": None, "search_top_k": 5}
        if not isinstance(config, Mapping):
            raise TypeError(
                "`skills` must be a dict with `paths`, `catalog_limit`, "
                "and `search_top_k` keys."
            )
        allowed_keys = {"paths", "catalog_limit", "search_top_k"}
        invalid_keys = set(config) - allowed_keys
        if invalid_keys:
            raise ValueError(
                f"Invalid skills config keys: {invalid_keys}. "
                f"Valid keys are: {allowed_keys}"
            )
        catalog_limit = self._normalize_optional_int(
            config.get("catalog_limit"),
            name="catalog_limit",
            minimum=0,
        )
        search_top_k = self._normalize_optional_int(
            config.get("search_top_k", 5),
            name="search_top_k",
            minimum=1,
        )
        return {
            "paths": config.get("paths"),
            "catalog_limit": catalog_limit,
            "search_top_k": search_top_k,
        }

    def _normalize_optional_int(
        self,
        value: Any,
        *,
        name: str,
        minimum: int,
    ) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, bool):
            raise TypeError(
                f"`{name}` must be an integer greater than or equal to {minimum}."
            )
        try:
            normalized = int(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"`{name}` must be an integer greater than or equal to {minimum}."
            ) from exc
        if normalized < minimum:
            raise ValueError(f"`{name}` must be greater than or equal to {minimum}.")
        return normalized

    def _normalize_paths(
        self,
        paths: Optional[SkillPaths],
    ) -> list[Path]:
        if paths is None:
            return []
        if isinstance(paths, bool):
            raise TypeError(
                "`skills['paths']` must be a path or list of paths. Use "
                "`msgflux.default_skill_paths()` to opt into conventional paths."
            )
        if isinstance(paths, (str, Path)):
            paths = [paths]
        resolved_paths = []
        seen = set()
        for path in paths:
            expanded = self._expand_path(path)
            for candidate in expanded:
                resolved = candidate.expanduser().resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                resolved_paths.append(resolved)
        return resolved_paths

    def _expand_path(self, path: SkillPath) -> list[Path]:
        path_str = str(Path(path).expanduser())
        if not any(char in path_str for char in "*?[]"):
            return [Path(path)]
        matches = glob(path_str, recursive=True)
        return [Path(match) for match in sorted(matches)]

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

    def has_searchable_skills(self) -> bool:
        return bool(self.searchable_skills())

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

    def catalog_skills(self) -> list[AgentSkill]:
        skills = [
            skill
            for skill in sorted(self.skills.values(), key=lambda item: item.name)
            if skill.catalog
        ]
        if self.catalog_limit is None:
            return skills
        return skills[: max(int(self.catalog_limit), 0)]

    def searchable_skills(self) -> list[AgentSkill]:
        cataloged = {skill.name for skill in self.catalog_skills()}
        return [
            skill
            for skill in sorted(self.skills.values(), key=lambda item: item.name)
            if skill.name not in cataloged
        ]

    def catalog(self) -> list[dict[str, str]]:
        return [
            {
                "name": skill.name,
                "description": skill.description,
            }
            for skill in self.catalog_skills()
        ]

    def search(self, query: str, *, top_k: Optional[int] = None) -> str:
        results = self.search_results(query, top_k=top_k)
        if not results:
            return "No matching skills found."
        lines = ["<skill_search_results>"]
        for skill, score in results:
            lines.extend(
                [
                    "<skill>",
                    f"name: {skill.name}",
                    f"description: {skill.description}",
                    f"score: {score:.4f}",
                    "</skill>",
                ]
            )
        lines.append("</skill_search_results>")
        return "\n".join(lines)

    def search_results(
        self,
        query: str,
        *,
        top_k: Optional[int] = None,
    ) -> list[tuple[AgentSkill, float]]:
        candidates = self.searchable_skills()
        if not candidates:
            return []
        if not query.strip():
            return []

        documents = [
            " ".join([skill.name, skill.description, " ".join(skill.metadata.values())])
            for skill in candidates
        ]
        if not any(document.strip() for document in documents):
            return []

        skills_by_document = dict(zip(documents, candidates))
        retriever = BM25LexicalRetriever()
        retriever.add(documents)
        response = retriever(
            query,
            top_k=top_k or self.search_top_k,
            threshold=0.0,
            return_score=True,
        )
        scored = []
        for result in response.data[0].results:
            skill = skills_by_document.get(result.data)
            if skill is None or result.score <= 0:
                continue
            scored.append((skill, result.score))
        return scored

    def activate(self, name: str) -> str:
        skill = self.get(name)
        return (
            f'<skill_content name="{skill.name}">\n'
            f"{skill.body}\n\n"
            f"Skill directory: {skill.directory}\n"
            "Relative paths in this skill are relative to the skill directory."
            "\n"
            "</skill_content>"
        )
