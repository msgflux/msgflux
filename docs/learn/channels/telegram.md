# Telegram Social Channel

The Telegram adapter lets a local or hosted msgFlux server receive Telegram
webhook updates, route each message to a registered Agent, and send the final
answer back to the same chat.

The flow is:

1. Telegram sends a webhook update to your public URL.
2. msgFlux acknowledges the webhook quickly and publishes an internal event.
3. The Social Boundary routes the event to an Agent.
4. The Telegram adapter sends the Agent's final response with `sendMessage`.

## 1. **Create a Bot**

Create a bot with BotFather and store the token in `.env`:

```bash
TELEGRAM_BOT_TOKEN=123456:your-bot-token
TELEGRAM_WEBHOOK_SECRET=replace-with-a-random-secret
```

`TELEGRAM_BOT_TOKEN` authenticates msgFlux when it calls the Telegram Bot API.

`TELEGRAM_WEBHOOK_SECRET` is your application secret. Telegram only knows it
because you pass it when configuring the webhook; after that Telegram includes
the same value in `X-Telegram-Bot-Api-Secret-Token` on every webhook request.

## 2. **Register the Adapter**

```python
import msgflux as mf
import msgflux.nn as nn
from msgflux.channels import TelegramAdapter

registry = mf.ChannelRegistry()
registry.social_adapter(
    "telegram",
    TelegramAdapter(
        bot_token_env="TELEGRAM_BOT_TOKEN",
        secret_token_env="TELEGRAM_WEBHOOK_SECRET",
    ),
)

@registry.social_route(channel="telegram")
def route_telegram(message, context):
    text = (message.text or "").strip().lower()
    if text.startswith("/sales"):
        return "sales"
    if text.startswith("/support"):
        return "support"
    return "support"

@registry.agent(name="support")
class SupportAgent(nn.Agent):
    model = "openai/gpt-4.1-mini"
    system_message = "You are a concise Telegram support assistant."

@registry.agent(name="sales")
class SalesAgent(nn.Agent):
    model = "openai/gpt-4.1-mini"
    system_message = "You answer product and pricing questions."
```

The server exposes the adapter at:

```text
POST /social/telegram/webhook
```

## 3. **Run Locally**

Start the msgFlux server:

```bash
uv run --with 'msgflux[server,openai]' msgflux server server.py --host 127.0.0.1 --port 8010
```

The `msgflux server` command loads `.env` by default without overriding
already-exported environment variables. Use `--env-file` to point at a
different file.

## 4. **Choose a Tunnel**

Telegram's hosted Bot API needs a public HTTPS webhook URL. For local
development, expose your local server through a tunnel and pass the generated
URL to Telegram.

`localtunnel` is the lowest-friction option for most local tests:

```bash
npx localtunnel --port 8010 --local-host 127.0.0.1
```

If you prefer installing it globally:

```bash
npm install -g localtunnel
lt --port 8010 --local-host 127.0.0.1
```

If you start msgFlux with `--port 8000`, then use
`lt --port 8000 --local-host 127.0.0.1`. The tunnel port must match the local
server port.

Other common choices:

| Tunnel | Good for | Command |
| --- | --- | --- |
| `localtunnel` | Fastest local setup, no account needed for random URLs. | `npx localtunnel --port 8010 --local-host 127.0.0.1` |
| `cloudflared` | Stable Cloudflare-managed tunnels and team environments. | `cloudflared tunnel --url http://127.0.0.1:8010` |
| `ngrok` | Debug UI, reserved domains, and webhook inspection. | `ngrok http 8010` |

Use a managed domain or ingress in production. Free local tunnels are best for
development, demos, and webhook tests.

## 5. **Configure the Webhook**

Copy the public HTTPS URL printed by your tunnel and point Telegram at the
msgFlux webhook path:

```bash
uv run --with 'msgflux[server]' msgflux telegram set-webhook \
  https://your-public-url.example/social/telegram/webhook
```

Equivalent raw Bot API call:

```bash
curl -X POST "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/setWebhook" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://your-public-url.example/social/telegram/webhook",
    "secret_token": "'"$TELEGRAM_WEBHOOK_SECRET"'"
  }'
```

Inspect or remove the webhook with:

```bash
uv run --with 'msgflux[server]' msgflux telegram webhook-info
uv run --with 'msgflux[server]' msgflux telegram delete-webhook
```

## 6. **Runtime Metadata**

The Telegram adapter decodes updates into `SocialMessage` and sets:

```text
session_id = "telegram:{chat_id}"
conversation_id = "{chat_id}"
sender_id = "{from.id}"
```

The Agent receives a normal one-turn chat message. Registry defaults still
populate `vars`, and pre-processors can mutate the run if needed, but social
metadata is not mixed into `vars` automatically.

Route functions and hooks can read `message.session_id`,
`message.conversation_id`, `message.sender_id`, and the same values in
`context.state`.

When the future checkpointer lands, it should use `session_id` to load and trim
conversation history. The Social Boundary intentionally does not persist
history by itself.

## 7. **Restrict Access**

For a personal bot, restrict access by `sender_id`. Telegram does not expose
your `sender_id` to a bot before you interact with that bot. The practical flow
is to send `/start`, read `message.sender_id` from the first webhook update, and
persist that value in your application config.

In private chats, `sender_id` and `conversation_id` are usually the same value.
In groups, use `sender_id` when the permission is about the person, and
`conversation_id` when the permission is about the chat.

The simplest place to enforce a personal allowlist is the social route:

```python
import os

ALLOWED_SENDERS = {
    sender.strip()
    for sender in os.getenv("TELEGRAM_ALLOWED_SENDER_IDS", "").split(",")
    if sender.strip()
}

@registry.social_route(channel="telegram")
def route_telegram(message, context):
    if message.metadata.get("chat_type") != "private":
        return None
    if ALLOWED_SENDERS and message.sender_id not in ALLOWED_SENDERS:
        return None
    return "support"
```

After the first inbound message, the server log or your route hook can reveal
the Telegram `sender_id`. Store it in `.env`:

```bash
TELEGRAM_ALLOWED_SENDER_IDS=949670859
```

The webhook secret proves the request came through Telegram. The `sender_id`
then identifies which Telegram user sent the message.

If you want to know your id before talking to your own bot, use a helper bot
such as `@userinfobot` or `@RawDataBot`. That gives you a value to put in
`TELEGRAM_ALLOWED_SENDER_IDS` before exposing your webhook. Your own bot still
only receives and persists that id after the user sends it an update.

If you already use `registry.auth` for HTTP, you can reuse it for social
channels. Social auth receives `http_request=None`, `request=SocialMessage`,
and a `ChannelContext` whose `channel` is `social:telegram`:

```python
@registry.auth
def authenticate(http_request, request, context):
    if context.channel == "http":
        token = http_request.headers.get("authorization")
        return authenticate_api_token(token)

    if context.channel == "social:telegram":
        allowed = os.getenv("TELEGRAM_ALLOWED_SENDER_IDS", "").split(",")
        if request.sender_id not in {item.strip() for item in allowed}:
            return False
        return {
            "provider": "telegram",
            "sender_id": request.sender_id,
            "conversation_id": request.conversation_id,
        }

    return False

@registry.authorize(agent="support")
def authorize_support(request, context, principal):
    if context.channel == "social:telegram":
        return principal["sender_id"] == context.state["sender_id"]
    return principal["tenant"] == request.run_config["vars"]["tenant"]
```

`context.state["social_message"]` contains the original `SocialMessage` for
hooks and shared auth logic.

## 8. **Rate Limits**

Registry rate limits also apply to social channels. Prefer stable social
identities over IP-based buckets: Telegram webhook requests come from Telegram,
not from the end user's device.

Limit by authenticated principal:

```python
@registry.auth
def authenticate(http_request, request, context):
    if context.channel == "social:telegram":
        return {"api_key": f"telegram:{request.sender_id}"}
    ...

registry.rate_limit(
    name="telegram-sender-minute",
    agent="support",
    requests=20,
    window_s=60,
    by="api_key",
)
```

Or use a callable bucket key:

```python
registry.rate_limit(
    name="telegram-sender-minute",
    agent="support",
    requests=20,
    window_s=60,
    by=lambda message, context: context.state["sender_id"],
)
```

Use `"service"` for a global bot-wide cap and `"tenant"` when your auth handler
maps Telegram users or chats to tenants.

## 9. **Commands**

Handle strong commands before the Agent. The model should not decide what
`/start`, `/stop`, or `/cancel` means.

Use `@registry.social_command` for command-specific behavior. Commands can be
scoped by social channel. Returning a string sends that text back to the same
conversation and does not call the Agent. Returning `None` consumes the command
without a reply. Returning `False` lets the message fall through to
`social_route`.

```python
@registry.social_command("/start", channel="telegram")
def start_command(message, context):
    # Store sender/chat metadata if this is an onboarding flow.
    return "Send a support question and I will route it to an agent."

@registry.social_command("/cancel", channel="telegram")
def cancel_command(message, context):
    # Override the built-in cancellation response if you need custom behavior.
    cancelled = context.boundary.cancel_session(message.session_id)
    return "Cancelled." if cancelled else "Nothing is running."

@registry.social_route(channel="telegram")
def route_telegram(message, context):
    return "support"
```

If you need full control over the outbound payload, return
`OutboundSocialMessage`. `from_context(...)` fills `channel` and
`conversation_id` from the inbound message:

```python
from msgflux.channels import OutboundSocialMessage

@registry.social_command("/help", channel="telegram")
def help_command(message, context):
    return OutboundSocialMessage.from_context(
        context,
        "Use /support for support or /sales for sales.",
        metadata={"command": "/help"},
    )
```

If no custom handler is registered, `/cancel` and `/stop` are built in. They
cancel the active Agent task for `message.session_id`. For Telegram private
chats, this is the private conversation with the bot. For groups, this is the
group conversation, so `/cancel` stops the active request for that group.

`session_id` is the user-facing conversation/thread identity. Future
checkpointing can use the same value to load message history. A `run_id`, when
present, remains an internal durability/retry identifier and is not required for
basic cancellation.

## 10. **Multimodal Input**

The default Telegram adapter keeps raw Telegram media metadata in
`message.attachments`. Applications can decide how and when to download files.

If an adapter downloads or resolves media, it can set `SocialMessage.content`
to a chat-completion content list:

```python
message.content = [
    {"type": "text", "text": message.text or ""},
    {"type": "image_url", "image_url": {"url": image_url}},
]
```

The Social Boundary forwards `message.content` directly as the user message
content when it is present.

## 11. **Responses**

By default, social replies send only the final response text. Reasoning remains
internal unless your Agent or post-processor explicitly maps it into the
outbound text.

In the future, a `stream_events` interface can map tool calls, progress updates,
and cancellation signals into multiple platform messages. For now, keep
Telegram responses as a single final message and route special commands like
`/cancel` in application code when needed.
