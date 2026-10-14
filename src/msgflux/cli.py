"""Unified service administration and noninteractive coding command line."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Sequence

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
            "msgflux service; msgflux service status; msgflux service restart."
        ),
    )
    parser.add_argument("-p", "--prompt", required=True, help="noninteractive task")
    parser.add_argument(
        "--json",
        action="store_true",
        help="JSON output; NDJSON when combined with --stream",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="observe live events before the final result",
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


def _write_observation(kind: str, record, *, as_json: bool) -> None:
    if as_json:
        line = msgspec_dumps({"type": kind, kind: record, "version": 1})
    elif kind == "event":
        content = record.data.get("delta")
        if not isinstance(content, str):
            content = msgspec_dumps(record.data)
        source = "/".join(record.source_path)
        label = f"{record.type} {source}" if source else record.type
        line = f"[{label}] {content}"
    else:
        line = f"[snapshot] thread_id={record.thread_id}"
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def _write_result(result, *, as_json: bool, streaming: bool = False) -> None:
    if as_json and streaming:
        _write_observation("result", result, as_json=True)
    elif as_json:
        sys.stdout.write(msgspec_dumps(result) + "\n")
    elif result.status == "completed":
        sys.stdout.write((result.output or "") + "\n")
    else:
        identities = " ".join(
            f"{name}={getattr(result, name)}"
            for name in ("thread_id", "run_id", "request_id")
            if getattr(result, name) is not None
        )
        sys.stderr.write(f"{result.error or result.status}\n{identities}\n")


def main(argv: Sequence[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "accounts":
        from msgflux.accounts.cli import main as accounts_main  # noqa: PLC0415

        accounts_main(args[1:])
        return
    if args and args[0] in {"init", "service"}:
        from msgflux.runtime.service.local.cli import (  # noqa: PLC0415
            main as service_main,
        )

        service_main(args if args[0] == "init" else args[1:])
        return
    options = _prompt_parser().parse_args(args)
    if not options.prompt.strip():
        _prompt_parser().error("--prompt must not be empty")
    from msgflux.coding.prompt import PromptExecution  # noqa: PLC0415

    execution = PromptExecution()

    def observe(kind, record):
        _write_observation(kind, record, as_json=options.json)

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
                interrupt_on_cancel=not options.detach_on_interrupt,
            )
        )
    except KeyboardInterrupt:
        result = execution.result
        if result.error is None and result.status == "failed":
            result.status = "detached" if options.detach_on_interrupt else "interrupted"
            result.error = (
                "Observation interrupted."
                if options.detach_on_interrupt
                else "Interrupted before a run was admitted."
            )
        _write_result(result, as_json=options.json, streaming=options.stream)
        raise SystemExit(130) from None
    except Exception as error:
        result = execution.result
        result.status = "failed"
        result.error = str(error)
    _write_result(result, as_json=options.json, streaming=options.stream)
    if result.status != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
