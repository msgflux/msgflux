"""Safe metadata projection and display for registered service agents."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from msgflux.runtime.service.records import AgentConfiguration, AgentModelDeployment
from msgflux.utils.msgspec import msgspec_dumps

if TYPE_CHECKING:
    from msgflux.runtime.service.local.manifest import ServiceManifest


def manifest_agent_configuration(
    manifest: ServiceManifest, agent_id: str
) -> AgentConfiguration:
    declared = manifest.agents[agent_id]
    source = {
        "config_path": str(manifest.path),
        "config_digest": manifest.digest,
        "extensions": tuple(declared.extensions),
    }
    if declared.factory is not None:
        return AgentConfiguration(factory=declared.factory, **source)
    resolved = manifest.coding_catalog.agents[agent_id]
    return AgentConfiguration(
        harness="coding",
        supports_model_override=True,
        model=resolved.model,
        reasoning_effort=resolved.reasoning_effort,
        models=tuple(
            AgentModelDeployment(
                model_name=deployment.model_name,
                model=deployment.model,
                description=deployment.description,
                reasoning_effort=deployment.reasoning_effort,
                account_name=deployment.account_name,
                is_default=index == 0,
            )
            for index, deployment in enumerate(resolved.models)
        ),
        model_fallback=resolved.model_fallback,
        permissions=resolved.permissions,
        tools_active=resolved.tools_active,
        tools_deferred=resolved.tools_deferred,
        subagents=resolved.subagents,
        preset=declared.preset,
        prompt_source="file"
        if resolved.prompt_file is not None
        else "preset"
        if resolved.instructions is not None
        else "default",
        prompt_file=str(resolved.prompt_file)
        if resolved.prompt_file is not None
        else None,
        **source,
    )


async def inspect_registered_agent(client, args) -> None:
    result = await client.inspect_agent(args.agent_id)
    if args.json:
        sys.stdout.write(msgspec_dumps(result) + "\n")
        return
    sys.stdout.write(
        f"Agent: {result.agent_id}\nDefault: {'yes' if result.is_default else 'no'}\n"
        f"Account (default): {result.account_name}\n"
    )
    config = result.configuration
    if config is None:
        sys.stdout.write(
            "Configuration: not supplied by the host factory registration.\n"
        )
        return
    sys.stdout.write("Configuration: loaded registration defaults\n")
    for label, value in (
        ("Harness", config.harness),
        ("Factory", config.factory),
        ("Model", config.model),
        ("Reasoning", config.reasoning_effort),
        ("Permissions", config.permissions),
        ("Tools (active defaults)", config.tools_active),
        ("Tools (deferred defaults)", config.tools_deferred),
        ("Subagents", config.subagents),
        ("Extensions", config.extensions),
        ("Preset", config.preset),
        ("Prompt source", config.prompt_source),
        ("Prompt file", config.prompt_file),
        ("Config", config.config_path),
        ("Config digest", config.config_digest),
    ):
        if value is not None:
            text = ", ".join(value) or "none" if isinstance(value, tuple) else value
            sys.stdout.write(f"{label}: {text}\n")
    if config.models:
        fallback = config.model_fallback if config.model_fallback is not None else False
        sys.stdout.write(f"Model fallback: {'yes' if fallback else 'no'}\n")
        sys.stdout.write("Models:\n")
        for deployment in config.models:
            default = " (default)" if deployment.is_default else ""
            reasoning = (
                f"; reasoning={deployment.reasoning_effort}"
                if deployment.reasoning_effort is not None
                else ""
            )
            account = (
                f"; account={deployment.account_name}"
                if deployment.account_name is not None
                else "; account=inherited execution account"
            )
            sys.stdout.write(
                f"  - {deployment.model_name}: {deployment.model}{default}; "
                f"{deployment.description}{reasoning}{account}\n"
            )
    if config.harness == "coding":
        sys.stdout.write(
            "Runtime tools and thread/run overrides depend on the session.\n"
        )
