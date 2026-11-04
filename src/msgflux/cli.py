"""Unified service administration and noninteractive coding command line."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Sequence

from msgflux._private.cli_output import write_observation
from msgflux.utils.errors import exception_message
from msgflux.utils.msgspec import msgspec_dumps


def _prompt_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="msgflux",
        description=(
            "Run a coding task, or manage the backend with "
            "'msgflux service', 'msgflux init', and 'msgflux accounts'."
        ),
        epilog=(
            "Commands: msgflux --version; msgflux version --check; msgflux update; "
            "msgflux init --model PROVIDER/MODEL; "
            "msgflux accounts login PROVIDER --account-name NAME; "
            "msgflux service; msgflux service status; msgflux service stop; "
            "msgflux service inspect; "
            "msgflux runs inspect/resume/activity THREAD_ID RUN_ID; "
            "msgflux runs ls --active; msgflux runs result THREAD_ID RUN_ID; "
            "msgflux threads result THREAD_ID [--wait]; "
            "msgflux threads interrupt/stop THREAD_ID; "
            "msgflux agents ls; msgflux agents inspect AGENT_ID; "
            "msgflux threads ls; msgflux threads watch THREAD_ID; "
            "msgflux threads steer THREAD_ID -m MESSAGE; "
            "msgflux config get/set/unset/validate; "
            "msgflux extensions ls; "
            "msgflux extensions inspect NAME [--options]; "
            "msgflux extensions add/remove NAME --agent ID. "
            "Prompt text: msgflux -p TEXT or "
            "msgflux --prompt-file PATH (- reads stdin)."
        ),
    )
    from msgflux.version import __version__  # noqa: PLC0415

    parser.add_argument("--version", action="version", version=f"msgflux {__version__}")
    prompt = parser.add_mutually_exclusive_group(required=True)
    prompt.add_argument("-p", "--prompt", help="noninteractive task")
    prompt.add_argument(
        "--prompt-file",
        metavar="PATH",
        help="read the task from a UTF-8 file, or from stdin with -",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="JSON output; NDJSON when combined with --stream",
    )
    execution = parser.add_mutually_exclusive_group()
    execution.add_argument(
        "--stream",
        action="store_true",
        help="observe live events before the final result",
    )
    execution.add_argument(
        "--detach",
        action="store_true",
        help="return execution IDs after admission and leave the service run active",
    )
    parser.add_argument(
        "--detach-on-interrupt",
        action="store_true",
        help="Ctrl+C stops observation and leaves the service run active",
    )
    parser.add_argument(
        "--agent", help="configured agent ID; defaults to defaults.agent"
    )
    parser.add_argument(
        "--account-name", help="credential bundle; defaults to defaults.account"
    )
    parser.add_argument(
        "--model",
        metavar="PROVIDER/MODEL",
        help="bind a model to a new thread; defaults to the selected agent's model",
    )
    parser.add_argument("--thread", help="continue an existing noninteractive thread")
    parser.add_argument("--config", type=Path, help="backend TOML configuration")
    parser.add_argument(
        "--runtime-dir", type=Path, help="local service runtime directory"
    )
    parser.add_argument(
        "--cwd",
        type=Path,
        help="project directory; defaults to the current directory for new threads",
    )
    return parser


def _read_stdin_prompt() -> str:
    """Decode stdin strictly; surrogate escapes are not acceptable task text."""
    buffer = getattr(sys.stdin, "buffer", None)
    if buffer is not None:
        return buffer.read().decode("utf-8")
    # Text-mode streams without a buffer may already hold surrogate escapes.
    return sys.stdin.read().encode("utf-8", "strict").decode("utf-8")


def _read_prompt_file(source: str, parser: argparse.ArgumentParser) -> str:
    try:
        if source == "-":
            return _read_stdin_prompt()
        return Path(source).expanduser().read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        parser.error(
            f"--prompt-file {source} is not readable UTF-8 text: "
            f"{exception_message(error)}"
        )
    raise AssertionError("unreachable")  # pragma: no cover - parser.error exits


def _prompt_text(options, parser: argparse.ArgumentParser) -> str:
    """Resolve the task text before any service or configuration is touched."""
    if options.prompt is not None:
        prompt, source = options.prompt, "--prompt"
    else:
        prompt = _read_prompt_file(options.prompt_file, parser)
        source = f"--prompt-file {options.prompt_file}"
    if not prompt.strip():
        parser.error(f"{source} must not be empty")
    return prompt


def _prompt_model(model: str | None, parser: argparse.ArgumentParser) -> str | None:
    if model is not None:
        from msgflux.coding.config import validate_model_path  # noqa: PLC0415

        try:
            validate_model_path(model)
        except ValueError as error:
            parser.error(str(error))
    return model


def _write_result(result, *, as_json: bool, streaming: bool = False) -> None:
    if as_json and streaming:
        write_observation("result", result, as_json=True)
    elif as_json:
        sys.stdout.write(msgspec_dumps(result) + "\n")
    elif result.status == "completed":
        sys.stdout.write((result.output or "") + "\n")
    else:
        sys.stderr.write(f"{result.error or result.status}\n")


def _write_identities(result) -> None:
    identities = " ".join(
        f"{name}={getattr(result, name)}"
        for name in ("thread_id", "agent_id", "model", "request_id", "run_id")
        if getattr(result, name, None) is not None
    )
    if identities:
        sys.stderr.write(identities + "\n")
        sys.stderr.flush()


class _PromptOutput:
    def __init__(self, options) -> None:
        self.options = options
        self.identities_written = False

    def admitted(self, result) -> None:
        if not self.options.json:
            _write_identities(result)
            self.identities_written = True

    def finish(self, result) -> None:
        if not self.options.json and not self.identities_written:
            _write_identities(result)
        if (
            self.options.detach
            and not self.options.json
            and result.status in {"accepted", "running", "completed"}
        ):
            sys.stdout.write(f"status={result.status}\n")
        else:
            _write_result(
                result, as_json=self.options.json, streaming=self.options.stream
            )


def _dispatch_administration(args: list[str]) -> bool:
    if args and args[0] in {"version", "update"}:
        from msgflux._private.updates import main as updates_main  # noqa: PLC0415

        updates_main(args[1:], command=args[0])
        return True
    if args and args[0] == "accounts":
        from msgflux.accounts.cli import main as accounts_main  # noqa: PLC0415

        accounts_main(args[1:])
        return True
    if args and args[0] == "extensions":
        from msgflux.runtime.service.local.extensions_cli import (  # noqa: PLC0415
            main as extensions_main,
        )

        extensions_main(args[1:])
        return True
    if args and args[0] == "config":
        from msgflux.runtime.service.local.config_cli import (  # noqa: PLC0415
            main as config_main,
        )

        config_main(args[1:])
        return True
    if args and args[0] in {"runs", "threads", "agents"}:
        from msgflux.runtime.service.local.cli import (  # noqa: PLC0415
            main as service_main,
        )

        service_main(args[1:], resource=args[0])
        return True
    if args and args[0] in {"init", "service"}:
        from msgflux.runtime.service.local.cli import (  # noqa: PLC0415
            main as service_main,
        )

        service_main(args if args[0] == "init" else args[1:])
        return True
    return False


def main(argv: Sequence[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and _dispatch_administration(args):
        return
    parser = _prompt_parser()
    options = parser.parse_args(args)
    # Resolve the task text before discovery so bad input never starts a service.
    prompt = _prompt_text(options, parser)
    model = _prompt_model(options.model, parser)
    from msgflux.coding.prompt import PromptExecution  # noqa: PLC0415

    execution = PromptExecution()
    output = _PromptOutput(options)

    def observe(kind, record):
        write_observation(kind, record, as_json=options.json)

    try:
        cwd = options.cwd.expanduser().resolve() if options.cwd is not None else None
        result = asyncio.run(
            execution.run(
                prompt,
                config=options.config,
                runtime_dir=options.runtime_dir,
                cwd=cwd,
                agent_id=options.agent,
                model=model,
                account_name=options.account_name,
                thread_id=options.thread,
                observer=observe if options.stream else None,
                interrupt_on_cancel=not (options.detach_on_interrupt or options.detach),
                detach=options.detach,
                on_admitted=output.admitted,
            )
        )
    except KeyboardInterrupt:
        result = execution.result
        if result.error is None and result.status == "failed":
            detached = options.detach_on_interrupt or options.detach
            result.status = "detached" if detached else "interrupted"
            result.error = (
                "Observation interrupted."
                if detached
                else "Interrupted before a run was admitted."
            )
        output.finish(result)
        raise SystemExit(130) from None
    except Exception as error:
        result = execution.result
        result.status = "failed"
        result.error = exception_message(error)
    output.finish(result)
    successful = (
        {"accepted", "running", "completed"} if options.detach else {"completed"}
    )
    if result.status not in successful:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
