from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Sequence

from msgflux.runtime.context import (
    ExecutionScope,
    get_execution_context,
    new_thread_id,
)
from msgflux.tools.dataclasses import ToolMetadata
from msgflux.tools.types import ToolBucket

if TYPE_CHECKING:
    from msgflux.nn.modules.agent import Agent


class AgentTool(ToolBucket):
    """Dispatch a message to one of the configured agents."""

    capture_kind = "agent"
    task_resume_params = ("name",)
    task_checkpoint_namespace_param = "name"
    name = "agent"
    display_name = "Agent"
    tool_config = {
        "inject_messages": True,
        "inject_vars": True,
    }
    description = (
        "Send a message to one of the configured agents. Use this when a "
        "specialized agent is better suited for the task."
    )
    annotations = {"name": str, "message": str, "return": str}

    def __init__(self, agents: Sequence[Agent] = ()):
        self._base_description = type(self).description
        for agent in agents:
            agent_name = self._get_agent_name(agent)
            self.add(
                ToolMetadata(
                    name=agent_name,
                    description=self._get_agent_description(agent) or "",
                    annotations={},
                    tool_config={
                        **dict(getattr(agent, "tool_config", {}) or {}),
                        "tool_kind": "agent",
                    },
                    impl=agent,
                    display_name=agent_name,
                    usage_guidance=self._get_agent_usage_guidance(agent),
                )
            )

        self.refresh()

    def refresh(self) -> None:
        self.description = self._build_description()
        self.usage_guidance = self._build_usage_guidance()

    def __call__(
        self,
        name: str,
        message: str,
        *,
        messages: Sequence[Mapping[str, Any]] | None = None,
        vars: Mapping[str, Any] | None = None,  # noqa: A002
        scope: ExecutionScope | None = None,
    ) -> str:
        selected = self.resolve_agent(name)
        return selected(
            message,
            **self._build_agent_kwargs(
                name,
                selected,
                messages=messages,
                runtime_vars=vars,
                scope=scope,
            ),
        )

    async def acall(
        self,
        name: str,
        message: str,
        *,
        messages: Sequence[Mapping[str, Any]] | None = None,
        vars: Mapping[str, Any] | None = None,  # noqa: A002
        scope: ExecutionScope | None = None,
    ) -> str:
        selected = self.resolve_agent(name)
        return await selected.acall(
            message,
            **self._build_agent_kwargs(
                name,
                selected,
                messages=messages,
                runtime_vars=vars,
                scope=scope,
            ),
        )

    def resolve_agent(self, name: str) -> Agent:
        metadata = self.tools.get(name)
        if metadata is not None:
            return metadata.impl
        available = ", ".join(sorted(self.tools))
        raise ValueError(f"Agent `{name}` not found. Available agents: {available}.")

    def _build_agent_kwargs(
        self,
        agent_name: str,
        agent: Agent,
        *,
        messages: Sequence[Mapping[str, Any]] | None,
        runtime_vars: Mapping[str, Any] | None,
        scope: ExecutionScope | None,
    ) -> Dict[str, Any]:
        metadata = self.tools.get(agent_name)
        config = metadata.tool_config if metadata is not None else {}
        kwargs: Dict[str, Any] = {"scope": scope or self._build_scope(agent)}

        if config.get("inject_messages", False):
            kwargs["messages"] = deepcopy(messages) if messages is not None else None

        inject_vars = config.get("inject_vars", False)
        if inject_vars is True:
            kwargs["vars"] = runtime_vars or {}
        elif isinstance(inject_vars, list) and inject_vars:
            available_vars = runtime_vars or {}
            missing = [key for key in inject_vars if key not in available_vars]
            if missing:
                missing_names = ", ".join(f"`{key}`" for key in missing)
                raise ValueError(
                    f"The agent `{agent_name}` requires the injected parameter "
                    f"{missing_names}, but it was not found."
                )
            kwargs["vars"] = {key: available_vars[key] for key in inject_vars}

        return kwargs

    def _build_scope(self, agent: Agent) -> ExecutionScope:
        context = get_execution_context()
        parent_scope = context["scope"]
        thread_id = context.get("thread_id")
        task_handle = context.get("task_handle")
        run_id = getattr(task_handle, "task_id", None) or context.get("run_id")
        parent_run_id = context.get("parent_run_id")
        if parent_run_id is None and context.get("run_id") != run_id:
            parent_run_id = context.get("run_id")

        return parent_scope.with_overrides(
            namespace=self._get_agent_runtime_name(agent),
            thread_id=thread_id if isinstance(thread_id, str) else new_thread_id(),
            run_id=run_id if isinstance(run_id, str) else None,
            parent_run_id=parent_run_id if isinstance(parent_run_id, str) else None,
            root_run_id=(
                context.get("root_run_id")
                if isinstance(context.get("root_run_id"), str)
                else None
            ),
        )

    def _build_description(self) -> str:
        agent_lines: List[str] = []
        for agent_name, metadata in sorted(self.tools.items()):
            description = metadata.description
            if description:
                agent_lines.append(f"- {agent_name}: {description}")
            else:
                agent_lines.append(f"- {agent_name}")
        if not agent_lines:
            agent_lines.append("- none")
        return f"{self._base_description}\n\nAvailable agents:\n" + "\n".join(
            agent_lines
        )

    def _build_usage_guidance(self) -> str | None:
        guidance_lines: List[str] = []
        for agent_name, metadata in sorted(self.tools.items()):
            guidance = metadata.usage_guidance
            if isinstance(guidance, str) and guidance.strip():
                guidance_lines.append(f"- {agent_name}: {' '.join(guidance.split())}")
        if not guidance_lines:
            return None
        return "Agent-specific guidance:\n" + "\n".join(guidance_lines)

    @staticmethod
    def _get_agent_name(agent: Agent) -> str:
        if hasattr(agent, "get_module_name"):
            name = agent.get_module_name()
        else:
            name = getattr(agent, "name", None) or getattr(agent, "__name__", None)
        if not isinstance(name, str) or not name:
            raise ValueError("Each agent must provide a non-empty name.")
        return name

    @staticmethod
    def _get_agent_runtime_name(agent: Agent) -> str:
        if hasattr(agent, "get_module_name"):
            return agent.get_module_name()
        name = getattr(agent, "name", None) or getattr(agent, "__name__", None)
        return name if isinstance(name, str) and name else "agent"

    @staticmethod
    def _get_agent_description(agent: Agent) -> str | None:
        if hasattr(agent, "get_module_description"):
            description = agent.get_module_description()
        else:
            description = getattr(agent, "description", None)
        if not isinstance(description, str) or not description.strip():
            return None
        return " ".join(description.split())

    @staticmethod
    def _get_agent_usage_guidance(agent: Agent) -> str | None:
        guidance = getattr(agent, "usage_guidance", None)
        if isinstance(guidance, str):
            return guidance
        config = getattr(agent, "tool_config", None)
        if isinstance(config, dict):
            value = config.get("usage_guidance")
        elif config is not None:
            value = getattr(config, "usage_guidance", None)
        else:
            value = None
        return value if isinstance(value, str) else None
