"""Unified service administration and noninteractive coding command line."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Sequence

from msgflux._cli_output import write_observation
from msgflux.utils.msgspec import msgspec_dumps


def _prompt_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="msgflux",
        description=(
            "Run a coding task, or manage the backend with "
            "'msgflux service', 'msgflux init', and 'msgflux accounts'."
        ),
        epilog=(
            "Commands: msgflux init --model PROVIDER/MODEL; "
            "msgflux accounts login PROVIDER --account-name NAME; "
            "msgflux service; msgflux service status; msgflux service stop; "
            "msgflux service inspect; msgflux runs inspect/resume THREAD_ID RUN_ID; "
            "msgflux runs ls --active; msgflux runs result THREAD_ID RUN_ID; "
            "msgflux threads result THREAD_ID [--wait]; "
            "msgflux threads interrupt/stop THREAD_ID; "
            "msgflux agents ls; msgflux agents inspect AGENT_ID; "
            "msgflux threads ls; msgflux threads watch THREAD_ID; "
            "msgflux threads steer THREAD_ID -m MESSAGE; "
            "msgflux config validate; msgflux config cors ls/add/remove; "
            "msgflux extensions ls; "
            "msgflux extensions inspect NAME [--options]; "
            "msgflux extensions add/remove NAME --agent ID."
        ),
    )
    parser.add_argument("-p", "--prompt", required=True, help="noninteractive task")
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
        for name in ("thread_id", "agent_id", "request_id", "run_id")
        if getattr(result, name) is not None
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
    options = _prompt_parser().parse_args(args)
    if not options.prompt.strip():
        _prompt_parser().error("--prompt must not be empty")
    from msgflux.coding.prompt import PromptExecution  # noqa: PLC0415

    execution = PromptExecution()
    output = _PromptOutput(options)

    def observe(kind, record):
        write_observation(kind, record, as_json=options.json)

    try:
        result = asyncio.run(
            execution.run(
                options.prompt,
                config=options.config,
                runtime_dir=options.runtime_dir,
                cwd=options.cwd,
                agent_id=options.agent,
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
        result.error = str(error)
    output.finish(result)
    successful = (
        {"accepted", "running", "completed"} if options.detach else {"completed"}
    )
    if result.status not in successful:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
