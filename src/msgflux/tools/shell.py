"""Provider-independent shell tool results shared with transport adapters."""

from typing import Literal

import msgspec

from msgflux._private.tool_result_reference import ToolResultRef


class ShellCommandResult(msgspec.Struct, frozen=True, kw_only=True):
    status: Literal["exited", "timed_out", "not_executed"]
    stdout: str = ""
    stderr: str = ""
    returncode: int | None = None

    def __post_init__(self):
        if self.status not in {"exited", "timed_out", "not_executed"}:
            raise ValueError("Invalid shell command status")
        if not isinstance(self.stdout, str) or not isinstance(self.stderr, str):
            raise ValueError("Shell stdout/stderr must be strings")
        if self.status == "exited":
            if type(self.returncode) is not int:
                raise ValueError("Exited command requires an integer returncode")
        elif self.returncode is not None:
            raise ValueError("Unfinished command cannot have a returncode")


class ShellResult(msgspec.Struct, frozen=True, kw_only=True, omit_defaults=True):
    results: tuple[ShellCommandResult, ...]
    output_reference: ToolResultRef | None = None

    def __post_init__(self):
        if self.output_reference is not None and not isinstance(
            self.output_reference, ToolResultRef
        ):
            raise TypeError("output_reference must be ToolResultRef or None")
        if (
            not isinstance(self.results, tuple)
            or not self.results
            or any(
                not isinstance(result, ShellCommandResult) for result in self.results
            )
        ):
            raise ValueError("Shell result requires command results")
