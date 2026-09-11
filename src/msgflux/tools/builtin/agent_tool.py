from __future__ import annotations

from typing import Any, List, Mapping, Optional

from msgflux.runtime.context import (
    ExecutionScope,
    get_execution_context,
    new_run_id,
    new_thread_id,
)
from msgflux.tools.handles import ToolBucketHandle
from msgflux.tools.types import ToolBucket, ToolBucketEntry, ToolLibraryOperator


class AgentTool(ToolBucket, ToolLibraryOperator):
    """Dispatch a message to one of the configured agents."""

    capture = {"tool_kind": "agent", "defer_loading": False}
    task_resume_params = ("name", "model")
    task_checkpoint_namespace_param = "name"
    name = "agent"
    display_name = "Agent"
    tool_config = {
        "runtime_inputs": ("handle",),
    }
    description = "Available agents:"
    annotations = {"name": str, "message": str, "return": str}

    def __init__(self):
        self._base_description = type(self).description
        self._entries: tuple[ToolBucketEntry, ...] = ()
        self.refresh()

    def refresh(self, entries: tuple[ToolBucketEntry, ...] = ()) -> None:
        self._entries = entries
        self.description = self._build_description()
        self.usage_guidance = self._build_usage_guidance()

    def patch_schema_annotations(
        self,
        annotations: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        annotations = dict(annotations)
        if self._has_gateway_agents():
            annotations["model"] = Optional[str]
        else:
            annotations.pop("model", None)
        return annotations

    def __call__(
        self,
        name: str,
        message: str,
        model: str | None = None,
        *,
        handle: ToolBucketHandle,
        scope: ExecutionScope | None = None,
    ) -> str:
        self._validate_agent(name, handle)
        model_preference = self._resolve_model_preference(name, model, handle)
        namespace = handle.get_execution_namespace(name)
        runtime_arguments = {
            "scope": scope
            or self._build_scope(
                namespace,
                tool_call_id=handle.get_tool_call_id(),
            )
        }
        if model_preference is not None:
            runtime_arguments["model_preference"] = model_preference
        return handle(
            name,
            message=message,
            _runtime_arguments=runtime_arguments,
        )

    async def acall(
        self,
        name: str,
        message: str,
        model: str | None = None,
        *,
        handle: ToolBucketHandle,
        scope: ExecutionScope | None = None,
    ) -> str:
        self._validate_agent(name, handle)
        model_preference = self._resolve_model_preference(name, model, handle)
        namespace = handle.get_execution_namespace(name)
        runtime_arguments = {
            "scope": scope
            or self._build_scope(
                namespace,
                tool_call_id=handle.get_tool_call_id(),
            )
        }
        if model_preference is not None:
            runtime_arguments["model_preference"] = model_preference
        return await handle.acall(
            name,
            message=message,
            _runtime_arguments=runtime_arguments,
        )

    @staticmethod
    def _validate_agent(name: str, handle: ToolBucketHandle) -> None:
        if handle.has_tool(name):
            return
        available = ", ".join(handle.list_captured_tools()) or "none"
        raise ValueError(f"Agent `{name}` not found. Available agents: {available}.")

    def _resolve_model_preference(
        self,
        agent_name: str,
        model: str | None,
        handle: ToolBucketHandle,
    ) -> str | None:
        if model is None:
            return None
        deployments = handle.get_entry(agent_name).metadata.get("models", ())
        if not deployments:
            return None
        available = tuple(deployment["name"] for deployment in deployments)
        if not isinstance(model, str) or not model:
            raise TypeError("`model` must be a non-empty string")
        if model not in available:
            choices = ", ".join(available)
            raise ValueError(f"Unknown model `{model}`. Available models: {choices}.")
        return model

    def _build_scope(
        self,
        agent_name: str,
        *,
        tool_call_id: str | None,
    ) -> ExecutionScope:
        context = get_execution_context()
        parent_scope = context["scope"]
        thread_id = context.get("thread_id")
        task_handle = context.get("task_handle")
        current_run_id = context.get("run_id")
        task_id = getattr(task_handle, "task_id", None)
        if isinstance(task_id, str):
            run_id = task_id
        elif isinstance(current_run_id, str) and isinstance(tool_call_id, str):
            run_id = f"{current_run_id}:{tool_call_id}:{agent_name}"
        else:
            run_id = new_run_id()
        parent_run_id = (
            context.get("parent_run_id") if isinstance(task_id, str) else current_run_id
        )
        root_run_id = context.get("root_run_id") or current_run_id

        return parent_scope.with_overrides(
            namespace=agent_name,
            thread_id=thread_id if isinstance(thread_id, str) else new_thread_id(),
            run_id=run_id,
            parent_run_id=parent_run_id if isinstance(parent_run_id, str) else None,
            root_run_id=root_run_id if isinstance(root_run_id, str) else None,
        )

    def _build_description(self) -> str:
        agent_lines: List[str] = []
        for entry in sorted(self._entries, key=lambda item: item.name):
            agent_name = entry.name
            description = entry.description
            if description:
                agent_lines.append(f"- {agent_name}: {description}")
            else:
                agent_lines.append(f"- {agent_name}")
            deployments = entry.metadata.get("models", ())
            if deployments:
                agent_lines.append("  Models:")
                for deployment in deployments:
                    model_name = deployment["name"]
                    model_description = deployment.get("description")
                    if not isinstance(model_description, str) or not model_description:
                        raise ValueError(
                            f"Agent `{agent_name}` uses ModelGateway model "
                            f"`{model_name}` without a description. Add a non-empty "
                            "`description` to every selectable deployment."
                        )
                    agent_lines.append(f"  - {model_name}: {model_description}")
        if not agent_lines:
            agent_lines.append("- none")
        return f"{self._base_description}\n" + "\n".join(agent_lines)

    def _has_gateway_agents(self) -> bool:
        return any(entry.metadata.get("models") for entry in self._entries)

    def _build_usage_guidance(self) -> str | None:
        guidance_lines: List[str] = []
        for entry in sorted(self._entries, key=lambda item: item.name):
            agent_name = entry.name
            guidance = entry.usage_guidance
            if isinstance(guidance, str) and guidance.strip():
                guidance_lines.append(f"- {agent_name}: {' '.join(guidance.split())}")
        if guidance_lines:
            return "Agent-specific guidance:\n" + "\n".join(guidance_lines)
        return None

    def compose_usage_guidance(self, declared: str | None) -> str | None:
        sections = []
        if isinstance(declared, str) and declared.strip():
            sections.append(" ".join(declared.split()))
        if isinstance(self.usage_guidance, str) and self.usage_guidance.strip():
            sections.append(self.usage_guidance)
        return "\n\n".join(sections) or None
