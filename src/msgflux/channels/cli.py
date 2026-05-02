import argparse
from typing import Optional, Sequence

from msgflux.channels.http.cli import run_server
from msgflux.channels.telegram_cli import run_telegram


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
    server_parser.add_argument("--title")
    server_parser.add_argument("--description")
    server_parser.add_argument("--log-level", default="info")
    server_parser.add_argument(
        "--env-file",
        default=".env",
        help="Load environment variables from this file before importing target.",
    )

    telegram_parser = subparsers.add_parser(
        "telegram",
        help="Manage Telegram social channel webhooks",
    )
    telegram_parser.add_argument(
        "--env-file",
        default=".env",
        help="Load TELEGRAM_BOT_TOKEN and TELEGRAM_WEBHOOK_SECRET from this file.",
    )
    telegram_parser.add_argument("--bot-token")
    telegram_parser.add_argument("--bot-token-env", default="TELEGRAM_BOT_TOKEN")
    telegram_parser.add_argument("--secret-token")
    telegram_parser.add_argument(
        "--secret-token-env",
        default="TELEGRAM_WEBHOOK_SECRET",
    )
    telegram_subparsers = telegram_parser.add_subparsers(
        dest="telegram_action",
        required=True,
    )
    set_webhook_parser = telegram_subparsers.add_parser(
        "set-webhook",
        help="Point Telegram updates to a public msgFlux webhook URL",
    )
    set_webhook_parser.add_argument(
        "url",
        help="Public HTTPS URL, e.g. https://example.com/social/telegram/webhook",
    )
    set_webhook_parser.add_argument(
        "--drop-pending-updates",
        action="store_true",
        default=None,
    )
    set_webhook_parser.add_argument(
        "--allowed-updates",
        nargs="+",
        help="Telegram update types to receive, e.g. message edited_message.",
    )

    delete_webhook_parser = telegram_subparsers.add_parser(
        "delete-webhook",
        help="Remove the Telegram webhook",
    )
    delete_webhook_parser.add_argument(
        "--drop-pending-updates",
        action="store_true",
        default=None,
    )

    telegram_subparsers.add_parser(
        "webhook-info",
        help="Show Telegram webhook status",
    )

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "server":
        return run_server(args)
    if args.command == "telegram":
        return run_telegram(args)

    parser.error("Unsupported command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
