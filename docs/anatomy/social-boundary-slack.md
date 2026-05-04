# Social Boundary And Slack

The Slack adapter extends the Social Boundary with Slack-specific webhook
verification, Events API challenge handling, message decoding, and threaded
responses.

This page documents the Slack-specific decisions. The shared queueing, routing,
debounce, auth, rate-limit, cancellation, and agent-run behavior is documented in
[Social Boundary And Telegram](social-boundary-telegram.md), because those pieces
are platform independent.

## Why Slack Needs A Webhook Response Hook

Telegram webhooks can be acknowledged with a generic accepted payload after the
body is decoded. Slack has an extra endpoint verification step.

When a Slack app Request URL is configured, Slack sends a payload like:

```json
{
  "type": "url_verification",
  "challenge": "..."
}
```

The server must return the challenge as the HTTP response. This is not a normal
social message and should not be routed to an Agent.

That requirement introduced `SocialWebhookResponse` and an optional adapter hook:

```text
adapter.webhook_response(body, http_request)
  -> None, for normal event processing
  -> SocialWebhookResponse, for platform-specific immediate responses
  -> mapping, converted to SocialWebhookResponse(payload=mapping)
```

The Social Boundary calls this hook after signature verification and before
`decode(...)`.

```text
adapter.verify(...)
  -> adapter.webhook_response(...)
     -> return immediately when not None
  -> adapter.decode(...)
  -> publish SocialEvent
```

This keeps Slack's challenge out of the generic social message contract while
still letting other adapters ignore the hook.

## Slack Request Verification

Slack signs each request with:

- `X-Slack-Request-Timestamp`
- `X-Slack-Signature`

The adapter lowercases headers before lookup so tests, FastAPI, and other request
objects do not need identical header casing.

Verification flow:

```text
read signing secret
  -> allow request if no secret configured
  -> read timestamp and signature headers
  -> reject missing or malformed timestamp
  -> reject timestamp older than tolerance
  -> compute HMAC-SHA256 over "v0:{timestamp}:{body}"
  -> compare with v0={digest} using hmac.compare_digest
```

The default timestamp tolerance is five minutes. That protects against replayed
webhooks while allowing modest clock skew.

Skipping the signing secret is useful for local experiments but should not be
used for production Slack apps.

## Slack App Surface Requirements

Slack Events API delivery and Slack's app conversation UI are configured in
different places.

The webhook can be valid, signed, and subscribed to `message.im`, while the Slack
client still refuses direct messages with errors such as `Sending messages to
this app has been turned off` or `This is still a work in progress`.

For direct messages from the app conversation, the Slack app must also have:

- **App Home** enabled as needed for the app.
- **Messages Tab** enabled.
- **Allow users to send Slash commands and messages from the messages tab**
  checked.

After changing these settings or OAuth scopes, the app may need to be reinstalled
to the workspace and the Slack client refreshed.

This is intentionally not represented in `SlackAdapter`. The adapter receives
only webhooks that Slack decides to deliver; app-surface availability remains an
external Slack app configuration prerequisite.

## Webhook Response Path

`SlackAdapter.webhook_response(...)` decodes the JSON body and handles only
`type == "url_verification"`.

If the challenge is valid, it returns:

```python
SocialWebhookResponse(payload={"challenge": challenge})
```

The HTTP route then serializes that payload with the configured status code.
Normal event callbacks return `None`, which tells the Social Boundary to continue
into `decode(...)`.

The hook is intentionally adapter-level. The Social Boundary should not know what
`url_verification` means; it only knows that some platforms require an immediate
custom webhook response.

## Event Decode

The Slack adapter currently decodes Events API `event_callback` payloads where
the inner event is a user `message`.

It ignores:

- non-`event_callback` payloads
- non-message events
- `bot_message` and `message_deleted` subtypes
- events with `bot_id`
- events without a user id

Ignoring bot messages is important because the adapter sends responses with
`chat.postMessage`. Without this guard the bot could consume its own replies.

The adapter preserves Slack message `files` as `SocialAttachment` records. It
does not download private file URLs. If a Slack event has no `text` but includes
files, the event is still decoded and routed with `message.text=None`.

Resolving media into model input should happen in application code, usually in a
pre-processor, because downloads require policy for file size, MIME type,
retention, tenant access, and bot token usage.

## Slack Identity Mapping

Slack event identity is mapped into `SocialMessage` like this:

```text
id = "slack:{event_id}"
channel = "slack"
session_id = "slack:{team_id}:{channel_id}:{thread_ts}"
conversation_id = "{channel_id}"
sender_id = "{user_id}"
```

`thread_ts` is the parent thread timestamp when the message is inside a Slack
thread. For root messages, it falls back to the message `ts`.

That means the Social Boundary concurrency unit is a Slack thread:

- messages in the same thread share a `session_id`
- `/cancel` cancels the active run for that thread
- debounce batches messages in the same thread
- different threads in the same channel can run independently

This is a better fit than using only `channel_id`, because Slack channels often
host multiple concurrent conversations.

## Runtime Metadata

The adapter stores useful Slack fields in `message.metadata`:

```text
team_id
channel_id
user_id
ts
thread_ts
event_type
file_ids
```

The Social Boundary copies the generic social fields into `ChannelContext.state`.
Applications can read the Slack-specific metadata from
`context.state["social_message"].metadata` in hooks, auth handlers, authorizers,
pre-processors, and rate-limit bucket callables.

The adapter does not inject Slack metadata into `vars`. If an application wants
Slack team/channel/user data inside the Agent's runtime variables, it should do
that explicitly in a pre-processor.

## Multimodal Status

Slack multimodal inbound is metadata-first:

Current behavior:

- plain Slack text messages are routed to the Agent
- Slack messages without text are routed when they include files
- Slack file/image/audio/video metadata is exposed as `SocialAttachment`
- file payloads are not downloaded automatically
- outbound responses are text-only through `chat.postMessage`

Use a pre-processor to resolve Slack files and replace `run.messages` with a
multimodal message:

```python
@registry.pre("support")
def slack_files_to_messages(message, context, run):
    image_urls = []

    for attachment in message.attachments:
        if attachment.type == "image":
            # Slack file URLs are usually private. Resolve or proxy them here
            # with your own size/type/access policy.
            image_urls.append(resolve_slack_file_url(attachment.payload))

    if image_urls:
        content = [{"type": "text", "text": message.text or "Analyze the image."}]
        content.extend(
            {"type": "image_url", "image_url": {"url": url}}
            for url in image_urls
        )
        run.messages = [{"role": "user", "content": content}]

    return run
```

This matches the shared Social Boundary rule: social multimodal should converge
into `run.messages`. `task_multimodal` remains an Agent-level option, but avoid
mixing it with `messages` unless you also control the full Agent call shape.

## Sending Responses

`SlackAdapter.send(...)` maps `OutboundSocialMessage` to Slack `chat.postMessage`.

The outbound payload includes:

```text
channel = outbound.conversation_id
text = outbound.text
thread_ts = outbound.metadata["thread_ts"] or inbound message.metadata["thread_ts"]
```

Preserving `thread_ts` makes replies stay inside the same Slack thread by
default.

Tests can inject a custom `sender` callable. Production usage defaults to a direct
Slack Web API call using `SLACK_BOT_TOKEN` or the configured token env var.

## Rate Limits And Auth

Slack webhooks originate from Slack infrastructure, so IP-based rate limits do
not identify the end user. Prefer stable identities such as:

- `sender_id`, for per-user limits
- `message.session_id`, for per-thread limits
- `metadata["team_id"]`, for workspace limits
- authenticated tenant mapping, for multi-tenant applications

A typical auth handler maps Slack identity into a principal:

```python
@registry.auth
def authenticate(http_request, request, context):
    if context.channel == "social:slack":
        return {
            "provider": "slack",
            "api_key": f"slack:{request.sender_id}",
            "sender_id": request.sender_id,
            "team_id": request.metadata["team_id"],
        }
```

The signing secret proves the webhook came from Slack. Registry auth decides
whether the Slack user, team, channel, or tenant is allowed to use the registered
agent.

## Boundary Interaction Summary

The complete Slack path is:

```text
Slack Events API
  -> POST /social/slack/webhook
  -> SlackAdapter.verify(...)
  -> SlackAdapter.webhook_response(...)
     -> url_verification returns challenge immediately
  -> SlackAdapter.decode(...)
  -> social dedupe by message id
  -> SocialEvent(channel="slack", message=SocialMessage(...))
  -> SocialBoundary auth/command/route/debounce/authorize/rate-limit/run
  -> SlackAdapter.send(...)
  -> Slack chat.postMessage
```

Slack therefore exercises one extension that Telegram did not need:
`SocialWebhookResponse`. The rest of the processing is intentionally shared with
other social channels.
