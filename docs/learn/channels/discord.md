# Discord Social Channel

The Discord adapter lets a msgFlux server receive Discord Interactions, route
slash-command input to registered Agents, and send one or more follow-up messages
back to Discord.

The MVP uses Discord **Interactions Endpoint URL**, not the Gateway. This keeps
local and production deployments webhook-based, like Telegram and Slack.

The flow is:

1. Discord sends an Interaction webhook to your public URL.
2. msgFlux verifies Discord's Ed25519 request signature when `DISCORD_PUBLIC_KEY`
   is configured.
3. The adapter answers Discord `PING` validation with `type: 1`.
4. Slash commands are acknowledged immediately with a deferred response
   `type: 5`.
5. The Social Boundary routes the command to an Agent in the background.
6. The Discord adapter sends Agent output as follow-up webhook messages.

## 1. **Create a Discord App**

Create a Discord application and store the public key in `.env`:

```bash
DISCORD_PUBLIC_KEY=your-application-public-key
DISCORD_BOT_TOKEN=your-bot-token
```

`DISCORD_PUBLIC_KEY` verifies that inbound interaction requests came from
Discord. The adapter checks `X-Signature-Ed25519` and `X-Signature-Timestamp`
before decoding the interaction.

`DISCORD_BOT_TOKEN` is optional for interaction follow-up webhooks, but keeping it
configured is useful if your deployment later adds bot-authenticated Discord API
calls.

??? example "Discord app setup walkthrough"

    Discord's developer UI changes over time. Treat this as a practical
    walkthrough and use Discord's official documentation as the source of truth
    if labels move.

    1. Open [Discord Developer Portal](https://discord.com/developers/applications).
    2. Click **New Application**.
    3. Choose a name, such as `msgFlux Dev Bot`.
    4. Open **General Information**.
    5. Copy **Public Key** into `DISCORD_PUBLIC_KEY`.
    6. Open **Bot**.
    7. Create or reset the bot token and copy it into `DISCORD_BOT_TOKEN`.
    8. Start the msgFlux server locally.
    9. Expose it with a tunnel.
    10. Open **General Information** again.
    11. Set **Interactions Endpoint URL** to
        `https://your-public-url.example/social/discord/webhook`.
    12. Save changes.

    Discord validates the URL by sending a signed `PING` interaction. The msgFlux
    Discord adapter responds with `{"type": 1}` automatically through
    `SocialWebhookResponse`.

    If Discord refuses to save the endpoint, check the public URL, HTTPS tunnel,
    server logs, and `DISCORD_PUBLIC_KEY`. Discord rejects endpoints that do not
    validate request signatures or do not answer the `PING` payload correctly.

## 2. **Create a Slash Command**

Create an application command that sends text to the Agent. A minimal command is
`/ask` with one string option named `prompt`.

The adapter reads command option values in this priority order:

```text
prompt, message, text, query, input
```

If none of those option names exist, it joins all option values into a text block.

Attachments can be supported by adding Discord attachment options to the slash
command. Discord sends attachment metadata under `data.resolved.attachments`, and
msgFlux preserves it as `message.attachments`.

## 3. **Register the Adapter**

```python
import msgflux as mf
import msgflux.nn as nn
from msgflux.channels import DiscordInteractionsAdapter

registry = mf.ChannelRegistry()
registry.social_adapter(
    "discord",
    DiscordInteractionsAdapter(
        public_key_env="DISCORD_PUBLIC_KEY",
        bot_token_env="DISCORD_BOT_TOKEN",
    ),
)

@registry.social_route(channel="discord")
def route_discord(message, context):
    return "support"

@registry.agent(name="support")
class SupportAgent(nn.Agent):
    """Support agent for Discord slash commands."""

    model = "openai/gpt-4.1-mini"
    system_message = "You are a concise Discord support assistant."
```

The server exposes the adapter at:

```text
POST /social/discord/webhook
```

## 4. **Run Locally**

Start the msgFlux server:

```bash
uv run --with 'msgflux[server,openai]' msgflux server server.py --host 127.0.0.1 --port 8010
```

The `msgflux server` command loads `.env` by default without overriding
already-exported environment variables. Use `--env-file` to point at a different
file.

## 5. **Choose a Tunnel**

Discord's Interactions Endpoint URL needs a public HTTPS endpoint. For local
development, expose your local server through a tunnel and paste the generated
URL into Discord.

`localtunnel` is a low-friction option for local tests:

```bash
npx localtunnel --port 8010 --local-host 127.0.0.1
```

Other common choices:

| Tunnel | Good for | Command |
| --- | --- | --- |
| `localtunnel` | Fast local setup, no account needed for random URLs. | `npx localtunnel --port 8010 --local-host 127.0.0.1` |
| `cloudflared` | Stable Cloudflare-managed tunnels and team environments. | `cloudflared tunnel --url http://127.0.0.1:8010` |
| `ngrok` | Debug UI, reserved domains, and webhook inspection. | `ngrok http 8010` |

Use a managed domain or ingress in production. Free local tunnels are best for
development, demos, and webhook tests.

## 6. **Runtime Metadata**

The Discord adapter decodes application command interactions into
`SocialMessage` and sets:

```text
session_id = "discord:{guild_id}:{channel_id}:{user_id}"
conversation_id = "{channel_id}"
sender_id = "{user_id}"
```

For DMs, `guild_id` becomes `dm`. For guild channels, the `session_id` scopes a
conversation by guild, channel, and user. That keeps each user's slash-command
thread independent inside the Social Boundary.

The Agent receives a normal one-turn chat message. Registry defaults still
populate `vars`, and pre-processors can mutate the run if needed, but Discord
metadata is not mixed into `vars` automatically.

Route functions and hooks can read `message.session_id`,
`message.conversation_id`, `message.sender_id`, and Discord-specific values in
`message.metadata`.

## 7. **Restrict Access**

For internal bots, restrict access by Discord `sender_id`, channel, or guild:

```python
import os

ALLOWED_DISCORD_USERS = {
    user.strip()
    for user in os.getenv("DISCORD_ALLOWED_USER_IDS", "").split(",")
    if user.strip()
}

@registry.social_route(channel="discord")
def route_discord(message, context):
    if ALLOWED_DISCORD_USERS and message.sender_id not in ALLOWED_DISCORD_USERS:
        return None
    return "support"
```

If you already use `registry.auth` for HTTP, you can reuse it for social
channels. Social auth receives `http_request=None`, `request=SocialMessage`, and
a `ChannelContext` whose `channel` is `social:discord`:

```python
@registry.auth
def authenticate(http_request, request, context):
    if context.channel == "http":
        token = http_request.headers.get("authorization")
        return authenticate_api_token(token)

    if context.channel == "social:discord":
        allowed = os.getenv("DISCORD_ALLOWED_USER_IDS", "").split(",")
        if request.sender_id not in {item.strip() for item in allowed}:
            return False
        return {
            "provider": "discord",
            "api_key": f"discord:{request.sender_id}",
            "sender_id": request.sender_id,
            "conversation_id": request.conversation_id,
            "tenant": request.metadata.get("guild_id"),
        }

    return False
```

The public key proves the request came through Discord. Registry auth decides
whether the Discord user, channel, guild, or tenant is allowed to use a registered
agent.

## 8. **Rate Limits**

Registry rate limits also apply to Discord social channels. Prefer stable
Discord identities over IP-based buckets: webhook requests come from Discord, not
from the end user's device.

Limit by authenticated principal:

```python
@registry.auth
def authenticate(http_request, request, context):
    if context.channel == "social:discord":
        return {"api_key": f"discord:{request.sender_id}"}
    ...

registry.rate_limit(
    name="discord-user-minute",
    agent="support",
    requests=20,
    window_s=60,
    by="api_key",
)
```

Or use a callable bucket key:

```python
registry.rate_limit(
    name="discord-session-minute",
    agent="support",
    requests=30,
    window_s=60,
    by=lambda message, context: message.session_id,
)
```

Use `"service"` for a global bot-wide cap and `"tenant"` when your auth handler
maps Discord guilds or users to tenants.

When a social rate limit rejects a message, msgFlux sends
`social_rate_limit_message` if configured. The default is
`"Too many requests. Try again later."`. Set it to `None` to drop rate-limited
social events silently.

## 9. **Commands**

Handle strong commands before the Agent. The model should not decide what
`/start`, `/stop`, or `/cancel` means.

Commands run after social auth and before route. This prevents unauthenticated
senders from calling `/cancel` or custom command handlers.

Use `@registry.social_command` for command-specific behavior. Commands can be
scoped by social channel:

```python
@registry.social_command("/start", channel="discord")
def start_command(message, context):
    return "Send `/ask prompt:...` and I will route it to an agent."

@registry.social_command(["/cancel", "/stop"], channel="discord")
def cancel_command(message, context):
    cancelled = context.boundary.cancel_session(message.session_id)
    return "Cancelled." if cancelled else "Nothing is running."

@registry.social_route(channel="discord")
def route_discord(message, context):
    return "support"
```

Discord slash commands arrive without the leading slash in `message.text`; for
example, `/ask prompt:hello` usually becomes `message.text == "hello"` and
`message.metadata["command_name"] == "ask"`. Use social commands for text-style
commands that users send as the prompt value, and use `social_route` when routing
by Discord command name.

## 10. **Responses**

Discord requires an initial interaction response quickly. The adapter therefore
returns a deferred response immediately:

```json
{"type": 5}
```

The Agent runs after that acknowledgement, and the final output is sent as a
follow-up webhook message.

The social boundary supports multiple outbound messages. Commands may return a
list, and hooks/processors can call `SocialContext.send(...)`:

```python
@registry.social_command("/help", channel="discord")
def help_command(message, context):
    return [
        "First, run `/ask` with your question.",
        "Then I will route it to the right agent.",
    ]
```

```python
@registry.post("support")
async def notify_progress(output, context, run):
    social_context = context.state["social_context"]
    await social_context.send("Formatting the answer...")
    return output
```

Any callable that receives the channel `context` can access
`context.state["social_context"]` and send social messages. That includes hooks,
auth handlers, authorizers, rate-limit bucket callables, pre-processors, and
post-processors. Use this sparingly in auth/rate-limit paths; for those cases,
prefer the configured social error messages unless you need custom behavior.

## 11. **Multimodal Input**

Discord slash-command attachment options arrive as attachment metadata. The
adapter preserves those objects as `message.attachments` and does not download
files automatically.

For images/files sent by the user, resolve media in a pre-processor and replace
`run.messages` with the multimodal message you want the Agent to receive:

```python
@registry.pre("support")
def discord_files_to_messages(message, context, run):
    image_urls = [
        attachment.payload["url"]
        for attachment in message.attachments
        if attachment.type == "image"
    ]

    if image_urls:
        content = [{"type": "text", "text": message.text or "Analyze the image."}]
        content.extend(
            {"type": "image_url", "image_url": {"url": url}}
            for url in image_urls
        )
        run.messages = [{"role": "user", "content": content}]

    return run
```

Discord CDN URLs should still go through your application policy before model
use. Enforce size limits, MIME allowlists, tenant access, token handling, and
retention rules before passing media to a model.

## 12. **Gateway Scope**

This adapter intentionally does not consume free-form Discord channel messages.
Free-form messages require a Gateway session and, for message content, the
`MESSAGE_CONTENT` privileged intent. The HTTP Interactions approach is simpler
for local development and easier to operate as a stateless webhook service.
