"""Compact default instructions for the local coding harness."""


def system_prompt(
    workspace: str,
    *,
    agent_name: str = "main",
    interactive: bool = True,
    permissions: tuple[str, ...] = ("read", "write", "execute"),
    custom_instructions: str | None = None,
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
        "or relative to the project directory. Inspect relevant files before editing.",
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
                "Use read for text and supported images; use bash for commands and "
                "searches when available. Use only the editing tools supplied.",
                access_instruction,
                interaction_instruction,
            )
        )
    return "\n\n".join(instructions)
