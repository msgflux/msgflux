import asyncio
import json
import sys
from argparse import Namespace
from typing import Any, Dict

from msgflux.channels.env import load_env_file
from msgflux.channels.social import TelegramAdapter


def run_telegram(args: Namespace) -> int:
    load_env_file(getattr(args, "env_file", None))

    adapter = TelegramAdapter(
        bot_token=getattr(args, "bot_token", None),
        bot_token_env=getattr(args, "bot_token_env", None),
        secret_token=getattr(args, "secret_token", None),
        secret_token_env=getattr(args, "secret_token_env", None),
    )
    result = asyncio.run(_run_telegram_action(adapter, args))
    sys.stdout.write(f"{json.dumps(result, indent=2, sort_keys=True)}\n")
    return 0


async def _run_telegram_action(
    adapter: TelegramAdapter,
    args: Namespace,
) -> Dict[str, Any]:
    action = args.telegram_action
    if action == "set-webhook":
        return await adapter.set_webhook(
            args.url,
            secret_token=getattr(args, "secret_token", None),
            drop_pending_updates=getattr(args, "drop_pending_updates", None),
            allowed_updates=getattr(args, "allowed_updates", None),
        )
    if action == "delete-webhook":
        return await adapter.delete_webhook(
            drop_pending_updates=getattr(args, "drop_pending_updates", None),
        )
    if action == "webhook-info":
        return await adapter.get_webhook_info()
    raise ValueError(f"Unsupported Telegram action `{action}`")
