"""Opt-in project instructions from the live workspace's current directory."""

from msgflux.nn.extensions.base import AgentExtension
from msgflux.nn.extensions.prompt import _append_section
from msgflux.nn.hooks import Hook, ModelContext
from msgflux.runtime.context import get_execution_scope

_FILENAMES = ("AGENTS.md", "Agents.md", "agents.md")


class AgentsFileExtension(AgentExtension):
    """Append one AGENTS.md variant without modifying conversation history.

    The first existing variant in AGENTS.md, Agents.md, agents.md order wins.
    Reads use the current workspace and its permissions on every model request;
    no workspace, missing files or blank content leaves the prompt unchanged.
    """

    def __init__(self, *, max_bytes: int = 65_536) -> None:
        super().__init__("agents_file")
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError("max_bytes must be a positive integer")
        self.max_bytes = max_bytes

    def hooks(self):
        return (Hook(event="transform_system_prompt", handler=self._add_instructions),)

    def _add_instructions(self, ctx: ModelContext) -> ModelContext:
        workspace = get_execution_scope().workspace
        if workspace is None:
            return ctx
        for filename in _FILENAMES:
            try:
                content = workspace.read_text(
                    filename, max_bytes=self.max_bytes, encoding="utf-8-sig"
                )
            except FileNotFoundError:
                continue
            if not content.strip():
                return ctx
            return _append_section(
                ctx, f'<agents_file name="{filename}">\n{content}\n</agents_file>'
            )
        return ctx
