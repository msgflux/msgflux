# Social Boundary And Discord

The Discord adapter extends the Social Boundary with Discord Interactions
verification, `PING` handshake handling, deferred interaction responses, slash
command decoding, and follow-up messages.

This page documents Discord-specific decisions. Shared queueing, routing,
debounce, auth, rate-limit, cancellation, and agent-run behavior is documented in
[Social Boundary And Telegram](social-boundary-telegram.md), because those pieces
are platform independent.

## Why Discord Uses Interactions

Discord has two practical inbound paths:

- Gateway events, for full bot event streams.
- Interactions Endpoint URL, for HTTP webhook delivery of slash commands and
  interaction callbacks.

The adapter uses Interactions because it matches msgFlux social channels better:

- the server remains webhook based
- local tunneling works like Telegram and Slack
- Discord validates the endpoint with a signed request
- slash commands provide a structured user input surface
- no long-running Gateway connection is needed for the MVP

Free-form channel messages are intentionally out of scope for this adapter. They
require Gateway handling and message-content permissions that are operationally
heavier than slash commands.

## Request Verification

Discord signs interaction requests with:

- `X-Signature-Ed25519`
- `X-Signature-Timestamp`

Verification flow:

```text
read public key
  -> allow request if no key configured
  -> read signature and timestamp headers
  -> verify Ed25519 signature over timestamp + body
  -> reject malformed or invalid signatures
```

`cryptography` is imported lazily. This keeps importing `msgflux.channels` from
requiring Discord verification dependencies unless a Discord public key is
actually configured.

Skipping the public key is useful for local unit tests but should not be used for
production Discord apps.

## Webhook Response Path

Discord requires platform-specific immediate responses before normal background
Agent work can continue.

For endpoint validation, Discord sends an interaction with `type == 1`. The
adapter returns:

```python
SocialWebhookResponse(payload={"type": 1})
```

For application commands, Discord expects a quick acknowledgement. The adapter
returns:

```python
SocialWebhookResponse(
    payload={"type": 5},
    continue_processing=True,
)
```

`continue_processing=True` tells the Social Boundary to return the webhook
response to Discord and still decode/publish the interaction in the background.
That is the key difference from Slack `url_verification`, where the response ends
processing immediately.

```text
adapter.verify(...)
  -> adapter.webhook_response(...)
     -> PING returns PONG and stops
     -> command returns deferred response and continues
  -> adapter.decode(...)
  -> publish SocialEvent
```

## Interaction Decode

The adapter currently decodes application command interactions only.

It ignores:

- non-application-command interactions
- payloads without a command `data` object

The command text is extracted from option values in this priority order:

```text
prompt, message, text, query, input
```

If none of those names exist, all option values are joined into a text block.
This keeps the `/ask prompt:...` case simple while still allowing custom command
option names.

## Discord Identity Mapping

Discord interaction identity is mapped into `SocialMessage` like this:

```text
id = "discord:{interaction_id}"
channel = "discord"
session_id = "discord:{guild_id}:{channel_id}:{user_id}"
conversation_id = "{channel_id}"
sender_id = "{user_id}"
```

For DMs, `guild_id` becomes `dm`. For guild channels, this makes the Social
Boundary concurrency unit a user inside a Discord channel:

- repeated commands from the same user/channel share a `session_id`
- `/cancel` cancels that user's active run in that channel
- debounce batches that user's command input in that channel
- different users in the same channel can run independently

## Runtime Metadata

The adapter stores useful Discord fields in `message.metadata`:

```text
application_id
interaction_id
interaction_token
guild_id
channel_id
user_id
command_name
option_values
```

`application_id` and `interaction_token` are required for follow-up webhook
messages. `command_name` and `option_values` let applications route by slash
command shape without reparsing the raw payload.

The adapter does not inject Discord metadata into `vars`. If an application wants
Discord guild/channel/user data inside Agent runtime variables, it should do that
explicitly in a pre-processor.

## Multimodal Status

Discord multimodal inbound is metadata-first:

Current behavior:

- slash-command text options are routed to the Agent
- Discord attachment option metadata is exposed as `SocialAttachment`
- file payloads are not downloaded automatically
- outbound responses are text-only follow-up webhook messages

Use a pre-processor to resolve Discord attachments and replace `run.messages`
with a multimodal message:

```python
@registry.pre("support")
def discord_files_to_messages(message, context, run):
    image_urls = []

    for attachment in message.attachments:
        if attachment.type == "image":
            image_urls.append(attachment.payload["url"])

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

`DiscordInteractionsAdapter.send(...)` maps `OutboundSocialMessage` to a Discord
follow-up webhook call:

```text
POST /webhooks/{application_id}/{interaction_token}
{"content": outbound.text}
```

The adapter gets `application_id` and `interaction_token` from inbound message
metadata plus outbound metadata. Custom sender callables can override the network
send path in tests or custom deployments.

The social boundary supports multiple outbound messages. Each call to
`SocialContext.send(...)` creates another follow-up message. This is important
for progress notices today and for event-stream style delivery later.

## Rate Limits And Auth

Discord webhooks originate from Discord infrastructure, so IP-based rate limits
do not identify the end user. Prefer stable identities such as:

- `sender_id`, for per-user limits
- `message.session_id`, for per-user-per-channel limits
- `metadata["guild_id"]`, for guild limits
- authenticated tenant mapping, for multi-tenant applications

A typical auth handler maps Discord identity into a principal:

```python
@registry.auth
def authenticate(http_request, request, context):
    if context.channel == "social:discord":
        return {
            "provider": "discord",
            "api_key": f"discord:{request.sender_id}",
            "sender_id": request.sender_id,
            "guild_id": request.metadata["guild_id"],
        }
```

The public key proves the webhook came from Discord. Registry auth decides
whether the Discord user, guild, channel, or tenant is allowed to use the
registered agent.

## Boundary Interaction Summary

The complete Discord path is:

```text
Discord Interactions
  -> POST /social/discord/webhook
  -> DiscordInteractionsAdapter.verify(...)
  -> DiscordInteractionsAdapter.webhook_response(...)
     -> PING returns {"type": 1} immediately
     -> command returns {"type": 5} and continues
  -> DiscordInteractionsAdapter.decode(...)
  -> social dedupe by message id
  -> SocialEvent(channel="discord", message=SocialMessage(...))
  -> SocialBoundary auth/command/route/debounce/authorize/rate-limit/run
  -> DiscordInteractionsAdapter.send(...)
  -> Discord follow-up webhook
```

Discord therefore exercises the `SocialWebhookResponse.continue_processing` path:
the HTTP response is immediate, while social event processing continues in the
background.
