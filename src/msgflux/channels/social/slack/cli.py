import asyncio
import json
import os
import sys
from argparse import Namespace
from typing import Any, Dict

from msgflux.channels.env import load_env_file
from msgflux.channels.exceptions import ChannelError
from msgflux.channels.social.slack.adapter import (
    DEFAULT_SLACK_BOT_TOKEN_ENV,
    _post_slack_api,
)


def run_slack(args: Namespace) -> int:
    load_env_file(getattr(args, "env_file", None))

    result = asyncio.run(_run_slack_action(args))
    sys.stdout.write(f"{json.dumps(result, indent=2, sort_keys=True)}\n")
    return 0


async def _run_slack_action(args: Namespace) -> Dict[str, Any]:
    action = args.slack_action
    if action == "auth-info":
        return await asyncio.to_thread(
            _slack_auth_info,
            _slack_bot_token(args),
            getattr(args, "timeout_s", 10.0),
        )
    raise ValueError(f"Unsupported Slack action `{action}`")


def _slack_auth_info(token: str, timeout_s: float) -> Dict[str, Any]:
    result = _post_slack_api(token, "auth.test", {}, timeout_s)
    if result.get("ok") is False:
        raise ChannelError(f"Slack auth.test failed: {result.get('error')}")
    return {
        "ok": result.get("ok"),
        "url": result.get("url"),
        "team": result.get("team"),
        "team_id": result.get("team_id"),
        "user": result.get("user"),
        "user_id": result.get("user_id"),
        "bot_id": result.get("bot_id"),
    }


def _slack_bot_token(args: Namespace) -> str:
    token = getattr(args, "bot_token", None)
    token_env = getattr(args, "bot_token_env", None) or DEFAULT_SLACK_BOT_TOKEN_ENV
    token = token or os.getenv(token_env, "")
    if not token:
        raise ChannelError("Slack bot token is not configured")
    return token
