from typing import Optional

from msgflux.runtime.skills import AgentSkillManager


class ActivateSkillTool:
    """Activate an Agent Skill and return its full instructions."""

    name = "activate_skill"
    display_name = "Skill"
    description = "Activate an Agent Skill and return its full instructions."

    def __init__(self, manager: AgentSkillManager):
        self.manager = manager

    def __call__(self, name: str) -> str:
        """Activate an Agent Skill and return its full instructions.

        Args:
            name: Name of the skill to activate.
        """
        return self.manager.activate(name)

    async def acall(self, name: str) -> str:
        return self(name)


class SkillSearchTool:
    """Search Agent Skills that are not listed in the initial catalog."""

    name = "skill_search"
    display_name = "Skill Search"
    description = "Search Agent Skills that are not listed in the initial catalog."

    def __init__(self, manager: AgentSkillManager):
        self.manager = manager

    def __call__(self, query: str, top_k: Optional[int] = None) -> str:
        """Search Agent Skills that are not listed in the initial catalog.

        Args:
            query: Search query describing the needed skill.
            top_k: Maximum number of skill results to return.
        """
        return self.manager.search(query, top_k=top_k)

    async def acall(self, query: str, top_k: Optional[int] = None) -> str:
        return self(query, top_k=top_k)
