from typing import Optional

from msgflux.runtime.skills import AgentSkillManager


class ActivateSkill:
    """Activate an Agent Skill and return its full instructions."""

    name = "activate_skill"
    display_name = "Skill"

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


class SkillSearch:
    """Search discoverable Agent Skills by name and description."""

    name = "skill_search"
    display_name = "Skill Search"

    def __init__(self, manager: AgentSkillManager):
        self.manager = manager

    def __call__(self, query: str, top_k: Optional[int] = None) -> str:
        """Search discoverable Agent Skills.

        Args:
            query: Search query describing the needed skill.
            top_k: Maximum number of skill results to return.
        """
        return self.manager.search(query, top_k=top_k)

    async def acall(self, query: str, top_k: Optional[int] = None) -> str:
        return self(query, top_k=top_k)
