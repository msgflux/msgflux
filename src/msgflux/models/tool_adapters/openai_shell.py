"""Local Responses shell wire helpers; no execution or authority here."""

from copy import deepcopy
from typing import Mapping

import msgspec

from msgflux.tools.shell import ShellCommandResult, ShellResult
from msgflux.utils.msgspec import msgspec_dumps


def shell_arguments(item):
    action = item.get("action")
    if not isinstance(action, Mapping):
        raise ValueError("Shell call requires an action")
    commands = action.get("commands")
    if (
        not isinstance(commands, list)
        or not commands
        or any(
            not isinstance(command, str) or not command.strip() or "\0" in command
            for command in commands
        )
    ):
        raise ValueError("Shell action requires non-empty commands")
    arguments = {"command": deepcopy(commands)}
    for name in ("timeout_ms", "max_output_length"):
        value = action.get(name)
        if value is not None:
            if type(value) is not int or value <= 0:
                raise ValueError("Shell limits must be positive integers")
            if name == "timeout_ms":
                arguments[name] = value
    return arguments


def shell_output(
    call_id, result, *, error=None, max_output_length=None, command_count=None
):
    if error is not None:
        output = [
            {"stdout": "", "stderr": error, "outcome": {"type": "exit", "exit_code": 1}}
        ] * (command_count or 1)
    else:
        if not isinstance(result, ShellResult):
            result = msgspec.convert(result, type=ShellResult, strict=True)
        output = [
            {
                "stdout": part.stdout,
                "stderr": part.stderr,
                "outcome": {"type": "timeout"}
                if part.status == "timed_out"
                else {
                    "type": "exit",
                    "exit_code": part.returncode if part.status == "exited" else 1,
                },
            }
            for part in result.results
        ]
        if command_count is not None and len(output) != command_count:
            raise ValueError("Shell output must include every command")
    item = {"type": "shell_call_output", "call_id": call_id, "output": output}
    if max_output_length is not None:
        item["max_output_length"] = max_output_length
    return item


class OpenAIShellAdapter:
    provider = "openai"
    api_mode = "responses"
    codec = "openai.responses.shell"
    version = 1
    kind = "shell"
    item_type = "shell_call"
    output_type = "shell_call_output"

    def declaration(self):
        return {"type": "shell", "environment": {"type": "local"}}

    def supports(self, entry):
        schema = (
            getattr(entry, "input_schema", None)
            or getattr(entry, "parameters", None)
            or {}
        )
        parameters = set(schema.get("properties", {}))
        # Preserve runtime selectors and custom inputs via function calling.
        return "command" in parameters and parameters <= {
            "command",
            "timeout_ms",
        }

    def validate_metadata(self, metadata):
        if (
            type(metadata.get("command_count")) is not int
            or metadata["command_count"] <= 0
        ):
            raise ValueError("Tool transport requires a positive command count")
        limit = metadata.get("max_output_length")
        if limit is not None and (type(limit) is not int or limit <= 0):
            raise ValueError("Invalid tool transport output limit")

    def decode(self, item, name):
        arguments = shell_arguments(item)
        metadata = {
            "codec": self.codec,
            "version": self.version,
            "name": name,
            "command_count": len(arguments["command"]),
            "max_output_length": item["action"].get("max_output_length"),
        }
        return arguments, metadata

    def render(self, call_id, result, metadata, *, error=None):
        return shell_output(
            call_id,
            result,
            error=error,
            command_count=metadata["command_count"],
            max_output_length=metadata.get("max_output_length"),
        )

    def project_history(self, item):
        if item["type"] == self.item_type:
            metadata = item.get("metadata", {}).get("tool_transport", {})
            return {
                "type": "function_call",
                "call_id": item["call_id"],
                "name": metadata.get("name", "bash_tool"),
                "arguments": msgspec_dumps(shell_arguments(item)),
            }
        results = []
        for part in item["output"]:
            outcome = part["outcome"]
            results.append(
                ShellCommandResult(
                    status="timed_out" if outcome["type"] == "timeout" else "exited",
                    returncode=outcome.get("exit_code"),
                    stdout=part["stdout"],
                    stderr=part["stderr"],
                )
            )
        return {
            "type": "function_call_output",
            "call_id": item["call_id"],
            "output": msgspec.to_builtins(ShellResult(results=tuple(results))),
        }

    def interrupted(self, item, reason):
        metadata = item.get("metadata", {}).get("tool_transport")
        if metadata is None:
            _, metadata = self.decode(item, "bash_tool")
        return self.render(item["call_id"], None, metadata, error=reason)
