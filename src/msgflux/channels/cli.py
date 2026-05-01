import argparse
from typing import Optional, Sequence

from msgflux.channels.http.cli import run_server


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="msgflux")
    subparsers = parser.add_subparsers(dest="command", required=True)

    server_parser = subparsers.add_parser(
        "server",
        help="Serve agents through an OpenAI-compatible HTTP server",
    )
    server_parser.add_argument(
        "target",
        help="Python target containing a ChannelRegistry, e.g. app.py:registry",
    )
    server_parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help=(
            "Allow downloading and executing a remote Python file when `target` "
            "is an http(s) URL."
        ),
    )
    server_parser.add_argument("--host", default="127.0.0.1")
    server_parser.add_argument("--port", default=8010, type=int)
    server_parser.add_argument("--title", default="msgflux")
    server_parser.add_argument("--log-level", default="info")

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "server":
        return run_server(args)

    parser.error("Unsupported command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
