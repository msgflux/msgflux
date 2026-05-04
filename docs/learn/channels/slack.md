# Slack Social Channel

The Slack adapter lets a msgFlux server receive Slack Events API webhooks, route
message events to registered Agents, and reply in the same Slack channel or
thread.

The flow is:

1. Slack sends an Events API webhook to your public URL.
2. msgFlux verifies the Slack request signature.
3. The adapter answers Slack's `url_verification` challenge when Slack validates
   the endpoint.
4. Regular message events are published to the Social Boundary and routed to an
   Agent.
5. The Slack adapter sends the Agent's final response with `chat.postMessage`.

## 1. **Create a Slack App**

Create a Slack app, install it to your workspace, and store the bot token and
signing secret in `.env`:

```bash
SLACK_BOT_TOKEN=xoxb-your-bot-token
SLACK_SIGNING_SECRET=your-signing-secret
```

`SLACK_BOT_TOKEN` authenticates msgFlux when it calls Slack Web API methods such
as `chat.postMessage`.

`SLACK_SIGNING_SECRET` verifies that inbound webhook requests came from Slack.
The adapter checks `X-Slack-Request-Timestamp` and `X-Slack-Signature` before it
decodes the event.

??? example "Slack app setup walkthrough"

    Slack's app UI changes over time. Treat this as a practical walkthrough and
    use the official Slack documentation as the source of truth if labels move.

    1. Open [Slack API Apps](https://api.slack.com/apps).
    2. Click **Create New App**.
    3. Choose **From scratch**.
    4. Pick a name, such as `msgFlux Dev Bot`, and choose the workspace.
    5. Open **OAuth & Permissions**.
    6. Under **Bot Token Scopes**, add `chat:write`.
    7. Add the message-read scopes you need:

    | Scope | Use case |
    | --- | --- |
    | `im:history` | Direct messages with the bot. |
    | `channels:history` | Public channels where the bot is a member. |
    | `groups:history` | Private channels where the bot is a member. |
    | `mpim:history` | Multi-person direct messages. |

    For a minimal local test, start with `chat:write` and `im:history`.

    1. Click **Install to Workspace**.
    2. Copy **Bot User OAuth Token** into `SLACK_BOT_TOKEN`.
    3. Open **Basic Information**.
    4. Copy **Signing Secret** into `SLACK_SIGNING_SECRET`.
    5. Start the msgFlux server locally.
    6. Expose it with a tunnel.
    7. Open **Event Subscriptions**.
    8. Enable events and set the Request URL to
       `https://your-public-url.example/social/slack/webhook`.
    9. Under **Subscribe to bot events**, add `message.im` for direct messages.
    10. Save changes and reinstall the app if Slack asks.

    Slack verifies the Request URL by sending `url_verification`. The msgFlux
    Slack adapter responds with the challenge automatically through
    `SocialWebhookResponse`.

## 2. **Register the Adapter**

```python
import msgflux as mf
import msgflux.nn as nn
from msgflux.channels import SlackAdapter

registry = mf.ChannelRegistry()
registry.social_adapter(
    "slack",
    SlackAdapter(
        bot_token_env="SLACK_BOT_TOKEN",
        signing_secret_env="SLACK_SIGNING_SECRET",
    ),
)

@registry.social_route(channel="slack")
def route_slack(message, context):
    text = (message.text or "").strip().lower()
    if text.startswith("sales"):
        return "sales"
    return "support"

@registry.agent(name="support")
class SupportAgent(nn.Agent):
    """Support agent for Slack conversations."""

    model = "openai/gpt-4.1-mini"
    system_message = "You are a concise Slack support assistant."

@registry.agent(name="sales")
class SalesAgent(nn.Agent):
    """Sales agent for product and pricing questions."""

    model = "openai/gpt-4.1-mini"
    system_message = "You answer product and pricing questions."
```

The server exposes the adapter at:

```text
POST /social/slack/webhook
```

## 3. **Run Locally**

Start the msgFlux server:

```bash
uv run --with 'msgflux[server,openai]' msgflux server server.py --host 127.0.0.1 --port 8010
```

The `msgflux server` command loads `.env` by default without overriding
already-exported environment variables. Use `--env-file` to point at a different
file.

## 4. **Choose a Tunnel**

Slack's Events API needs a public HTTPS Request URL. For local development,
expose your local server through a tunnel and pass the generated URL to Slack.

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

## 5. **Configure Events API**

In the Slack app settings, open **Event Subscriptions**, enable events, and set
the Request URL to:

```text
https://your-public-url.example/social/slack/webhook
```

Slack sends a `url_verification` payload when you save the URL. msgFlux responds
with Slack's challenge through `SocialWebhookResponse`, so the endpoint can be
verified before normal events are delivered.

Subscribe the bot to message events that match your use case, for example:

| Event | Use case |
| --- | --- |
| `message.im` | Direct messages to the bot. |
| `message.channels` | Public channels where the bot is a member. |
| `message.groups` | Private channels where the bot is a member. |
| `message.mpim` | Multi-person direct messages. |

Grant the bot a send-message scope such as `chat:write`, then reinstall the app
if Slack asks you to apply scope changes.

## 6. **Runtime Metadata**

The Slack adapter decodes message events into `SocialMessage` and sets:

```text
session_id = "slack:{team_id}:{channel_id}:{thread_ts}"
conversation_id = "{channel_id}"
sender_id = "{user_id}"
```

For root messages, `thread_ts` is the message `ts`. For replies inside a Slack
thread, `thread_ts` is the parent thread timestamp. That means each Slack thread
maps to a separate msgFlux social session.

The Agent receives a normal one-turn chat message. Registry defaults still
populate `vars`, and pre-processors can mutate the run if needed, but social
metadata is not mixed into `vars` automatically.

Route functions and hooks can read `message.session_id`,
`message.conversation_id`, `message.sender_id`, and the same values in
`context.state`.

## 7. **Restrict Access**

For internal bots, restrict access by Slack `sender_id`, channel, or team:

```python
import os

ALLOWED_SLACK_USERS = {
    user.strip()
    for user in os.getenv("SLACK_ALLOWED_USER_IDS", "").split(",")
    if user.strip()
}

@registry.social_route(channel="slack")
def route_slack(message, context):
    if ALLOWED_SLACK_USERS and message.sender_id not in ALLOWED_SLACK_USERS:
        return None
    return "support"
```

If you already use `registry.auth` for HTTP, you can reuse it for social
channels. Social auth receives `http_request=None`, `request=SocialMessage`, and
a `ChannelContext` whose `channel` is `social:slack`:

```python
@registry.auth
def authenticate(http_request, request, context):
    if context.channel == "http":
        token = http_request.headers.get("authorization")
        return authenticate_api_token(token)

    if context.channel == "social:slack":
        allowed = os.getenv("SLACK_ALLOWED_USER_IDS", "").split(",")
        if request.sender_id not in {item.strip() for item in allowed}:
            return False
        return {
            "provider": "slack",
            "api_key": f"slack:{request.sender_id}",
            "sender_id": request.sender_id,
            "conversation_id": request.conversation_id,
        }

    return False
```

The signing secret proves the request came through Slack. The `sender_id`,
`conversation_id`, and `team_id` metadata identify who sent the message and where
it came from.

For shared channels, keep the decision in `social_route`. Slack may deliver
channel messages where the bot is a member; the route should return `None` unless
the bot was explicitly mentioned:

```python
SLACK_BOT_USER_ID = "U012ABCDEF"

@registry.social_route(channel="slack")
def route_slack(message, context):
    text = message.text or ""
    channel_id = message.metadata.get("channel_id", "")

    if not channel_id.startswith("D"):
        mentioned = f"<@{SLACK_BOT_USER_ID}>" in text
        if not mentioned:
            return None

    return "support"
```

Use `sender_id` for user allowlists, `conversation_id` for channel allowlists,
and `team_id` for workspace/tenant allowlists. The mention check prevents normal
channel traffic from becoming agent work.

## 8. **Rate Limits**

Registry rate limits also apply to Slack social channels. Prefer stable Slack
identities over IP-based buckets: webhook requests come from Slack, not from the
end user's device.

Limit by authenticated principal:

```python
@registry.auth
def authenticate(http_request, request, context):
    if context.channel == "social:slack":
        return {"api_key": f"slack:{request.sender_id}"}
    ...

registry.rate_limit(
    name="slack-user-minute",
    agent="support",
    requests=20,
    window_s=60,
    by="api_key",
)
```

Or use a callable bucket key:

```python
registry.rate_limit(
    name="slack-thread-minute",
    agent="support",
    requests=30,
    window_s=60,
    by=lambda message, context: message.session_id,
)
```

Use `"service"` for a global bot-wide cap and `"tenant"` when your auth handler
maps Slack users, channels, or teams to tenants.

## 9. **Commands**

Handle strong commands before the Agent. The model should not decide what
`/start`, `/stop`, or `/cancel` means.

Use `@registry.social_command` for command-specific behavior. Commands can be
scoped by social channel:

```python
@registry.social_command("/start", channel="slack")
def start_command(message, context):
    return "Send a support question and I will route it to an agent."

@registry.social_command("/cancel", channel="slack")
def cancel_command(message, context):
    cancelled = context.boundary.cancel_session(message.session_id)
    return "Cancelled." if cancelled else "Nothing is running."

@registry.social_route(channel="slack")
def route_slack(message, context):
    return "support"
```

If no custom handler is registered, `/cancel` and `/stop` are built in. They
cancel the active Agent task for `message.session_id`. For Slack, this means the
active request for the current Slack thread.

## 10. **Responses**

By default, Slack replies send only the final response text. Reasoning remains
internal unless your Agent or post-processor explicitly maps it into the
outbound text.

The adapter posts responses to `conversation_id` and preserves the inbound
`thread_ts`, so threaded messages receive threaded replies.
