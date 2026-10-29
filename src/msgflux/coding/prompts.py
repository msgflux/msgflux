"""Compact default instructions for the local coding harness."""


def system_prompt(
    workspace: str,
    *,
    agent_name: str = "main",
    interactive: bool = True,
    permissions: tuple[str, ...] = ("read", "write", "execute"),
    custom_instructions: str | None = None,
    delegation_guidance: tuple[str, ...] = (),
) -> str:
    """Build the default prompt for one coding agent and workspace."""
    access_instruction = f"Workspace permissions: {', '.join(permissions) or 'none'}."
    if "write" not in permissions:
        access_instruction += " Do not modify files or claim to have edited them."
    if "execute" in permissions:
        access_instruction += " Local commands run on the host without an OS sandbox."
    else:
        access_instruction += " Process execution is unavailable."
    interaction_instruction = (
        "Keep the user informed with brief commentary when useful. If a "
        "send_user_message tool is supplied, use it for progress updates."
        if interactive
        else "This is a noninteractive print run. Complete the request "
        "autonomously. Do not ask questions, wait for user input, or send "
        "progress commentary. State necessary assumptions and the result in "
        "your final answer."
    )
    instructions = [
        f"You are {agent_name}, the coding agent in Vulcano.",
        f"The project directory is {workspace}. Tool paths are absolute host paths "
        "or relative to the project directory. Inspect relevant definitions and "
        "callers before editing.",
    ]
    if custom_instructions is not None:
        instructions.extend(
            (
                "Use only the editing tools actually supplied.",
                access_instruction,
                interaction_instruction,
                custom_instructions,
            )
        )
    else:
        instructions.extend(
            (
                "Prefer simple, maintainable changes. Follow the project's "
                "instructions.",
                "Use only supplied tools. Discover deferred tools before calling them. "
                "Make focused changes, verify them with available checks, and "
                "report results and remaining limitations.",
                access_instruction,
                interaction_instruction,
            )
        )
    if delegation_guidance:
        instructions.append(
            "Use the configured subagents through the agent tool:\n"
            + "\n".join(delegation_guidance)
        )
    return "\n\n".join(instructions)
