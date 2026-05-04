# Social Boundary And Telegram

The social channel boundary turns webhook-oriented chat applications into normal
msgFlux agent runs.

This page documents the shared social boundary and the Telegram adapter. It is
about internal contracts and design decisions, not end-user setup commands.

## Design Goal

Social platforms are not OpenAI-compatible chat-completion clients. They deliver
webhooks, expect fast acknowledgements, identify users and chats in platform
specific ways, and require the application to send the final answer back through
a platform API.

The Social Boundary keeps those concerns out of `Agent`:

```text
platform webhook
  -> adapter.verify(...)
  -> adapter.decode(...)
  -> SocialMessage
  -> SocialBoundary event queue
  -> registry.auth
  -> registry.social_command / registry.social_route
  -> registry.authorize / registry.rate_limit
  -> Agent.acall(...)
  -> adapter.send(...)
```

The registered agent still receives an ordinary one-turn chat message. Platform
metadata stays in `SocialMessage`, `SocialContext`, and `ChannelContext.state`.
It is not injected into `vars` automatically.

## Core Types

`SocialMessage` is the normalized inbound message:

```text
id              stable platform event/message id
channel         normalized channel name, for example "telegram"
session_id      user-facing conversation/thread identity
conversation_id platform destination for replies
sender_id       platform user id
text            text or caption when available
content         optional chat-completion style multimodal content
attachments     unresolved platform media metadata
metadata        normalized adapter metadata
raw             original webhook payload
```

`SocialContext` is the runtime context passed to route and command handlers:

```text
channel
adapter
message
boundary
agent_name
state
```

`SocialEvent` is the internal queue item. It keeps the decoded message, channel,
and adapter together so the consumer can process webhooks asynchronously after the
HTTP route has acknowledged the platform.

`OutboundSocialMessage` is the normalized reply contract. Adapters decide how to
turn it into platform API calls.

## Adapter Contract

A social adapter is intentionally small. The boundary expects these methods:

```text
verify(http_request, body) -> bool
decode(body, http_request) -> list[SocialMessage]
send(outbound, context) -> None
```

`verify` authenticates the platform webhook. `decode` converts one webhook body
into zero or more `SocialMessage` objects. `send` publishes the final response
back to the platform.

The adapter owns platform details. The boundary owns routing, auth integration,
rate limits, debounce, cancellation, and agent execution.

## Webhook Handling

`SocialBoundary.handle_webhook(...)` is called by the FastAPI route registered at
`/social/{channel}/webhook`.

The flow is:

```text
normalize channel
  -> find adapter
  -> adapter.verify(...)
  -> adapter.webhook_response(...) when available
  -> adapter.decode(...)
  -> dedupe each SocialMessage by "{channel}:{message.id}"
  -> publish SocialEvent for each decoded SocialMessage
  -> return accepted event count
```

The webhook route returns quickly. Actual agent work happens in the boundary
consumer task. This matters because social platforms generally expect a fast HTTP
acknowledgement and may retry when webhook responses are slow.

`registry.settings(social_dedup_ttl_s=...)` controls the dedupe window. The
default is `300` seconds. Set it to `0` or `None` to disable dedupe. The default
store is `InMemorySocialDedupStore`, which is suitable for one Python process.
Production deployments with multiple workers should inject a shared store through
`registry.social_dedup_store(...)`.

## Event Consumer

`SocialBoundary.start()` creates a consumer task when at least one adapter exists.
The consumer reads from `InMemorySocialEventBus` and calls `process_event(...)`.

`process_event(...)` does the synchronous application decisions before starting a
run:

```text
build SocialContext
  -> registry.auth
  -> handle command
  -> reject if a run is already active for session_id
  -> optional debounce
  -> route to agent
  -> registry.authorize(agent_name)
  -> registry.check_rate_limits(...)
  -> create active task for the session
```

The active task map is keyed by `message.session_id`. This is the unit of
cancellation and concurrency protection.

## Security Pipeline

Social channels have two different authentication layers:

- `adapter.verify(...)` proves that the webhook came from the platform.
- `registry.auth` decides whether the platform sender/chat/team/tenant may use
  this application.

The security order is:

```text
adapter.verify(...)
  -> adapter.decode(...)
  -> social dedupe
  -> registry.auth
  -> registry.social_command(...)
  -> registry.social_route(...)
  -> registry.authorize(agent_name)
  -> registry.check_rate_limits(...)
  -> Agent.acall(...)
```

This order is intentional:

- commands cannot bypass auth
- `/cancel` cannot be executed by an unauthenticated sender
- route runs only after the sender is identified
- agent-specific authorization and rate limits run only after `agent_name` is
  known

Command handlers are checked against global rate-limit policies
(`agent=None`). Agent-specific rate limits apply after routing. For social
channels, prefer buckets based on stable social identity rather than IP address,
because webhook requests originate from platform infrastructure.

Social error responses are configurable:

```python
registry.settings(
    social_unauthorized_message=None,
    social_forbidden_message=None,
    social_rate_limit_message="Too many requests. Try again later.",
)
```

The default keeps unauthorized and forbidden events silent. Rate-limited events
send a short user-facing message by default because the sender has already been
identified.

## Commands Before Agents

Commands are deliberately handled before routing. The model should not decide
what `/start`, `/cancel`, or `/stop` means.

Commands still run after `registry.auth`. They are "before agents", not before
the security boundary.

`registry.social_command(...)` registers handlers per channel or globally. A
command may be a single string or a list of aliases:

```python
@registry.social_command(["/cancel", "/stop"], channel="telegram")
def cancel_command(message, context):
    cancelled = context.boundary.cancel_session(message.session_id)
    return "Cancelled." if cancelled else "Nothing is running."
```

Command return values are part of the boundary contract:

- `str` sends a text response and consumes the command.
- `OutboundSocialMessage` sends a custom outbound payload and consumes it.
- `None` consumes the command without a response.
- `False` lets the message fall through to `social_route`.

If there is no custom handler, `/cancel` and `/stop` are built in. They cancel
both active runs and pending debounced messages for the same `session_id`.

## Routing To Agents

Routes map a `SocialMessage` to an agent name:

```python
@registry.social_route(channel="telegram")
def route_telegram(message, context):
    if message.text and message.text.startswith("/sales"):
        return "sales"
    return "support"
```

The boundary checks channel-specific routes first, then global routes. Returning
`None` or any falsey value drops the event.

This keeps multi-agent selection in application code instead of encoding it in
Telegram-specific adapter logic.

## Debounce Before Run Start

`registry.settings(social_debounce_s=...)` enables short message coalescing per
`session_id`.

The behavior is:

```text
message arrives
  -> no command
  -> no active task
  -> append to pending_events[session_id]
  -> start debounce timer

another message arrives before timer expires
  -> append to the same pending list
  -> cancel old timer
  -> start a new timer

timer expires
  -> merge messages
  -> route once
  -> run agent once
```

The merge strategy is intentionally simple:

- text parts are joined with newlines
- attachments are concatenated
- `content` is cleared so the merged text is used unless a pre-processor builds a
  richer multimodal payload
- metadata receives `batched`, `batch_size`, and `batch_message_ids`
- `raw` stores the list of original raw payloads

## Active Session And Cancellation

Only one active agent task is allowed per `session_id`.

If a non-command message arrives while a run is active, the boundary replies:

```text
A request is already running for this session. Send /cancel to stop it.
```

`cancel_session(session_id)` cancels two things:

- any pending debounce timer and buffered messages
- any active agent task

This gives `/cancel` useful behavior even before the run has started.

## Social Auth And Rate Limits

Social auth reuses the same registry hooks as HTTP, but with social context:

```text
context.channel = "social:telegram"
http_request = None
request = SocialMessage
context.state["social_message"] = message
```

The adapter-level webhook secret proves the request came through Telegram. The
registry auth handler decides whether this sender/chat/tenant may use the
application.

For rate limits, stable social identities are usually better than IP buckets.
Telegram webhook requests come from Telegram infrastructure, not from the end
user device.

## Agent Run Mapping

After routing and auth, the boundary prepares an `AgentRun`:

```text
messages = [{"role": "user", "content": social_message_content}]
vars = defaults.vars
stream = False
model_preference = defaults.model_preference
tool_filter = defaults.tool_filter
kwargs = defaults.kwargs
policies = defaults policies
```

Pre-processors can mutate this run. That is the intended place for application
specific transformations such as:

- mapping social metadata into `vars` when the application wants that
- converting Telegram attachments into `task_multimodal`
- applying per-tenant tool filters or model preferences

The boundary itself does not persist history and does not load previous messages.
Future checkpointing should use `session_id` to decide which conversation/thread
history belongs to the run.

## Multimodal Input Boundary

The social boundary uses `run.messages` as the canonical inbound shape. That is
the same shape checkpointing will eventually persist and replay.

For text-only social messages, the default run is:

```python
run.messages = [{"role": "user", "content": message.text or ""}]
```

For user-originated multimodal input, adapters should first preserve platform
media metadata in `message.attachments`. Application code can then resolve those
attachments in a pre-processor and replace `run.messages` with a multimodal
ChatML message:

```python
@registry.pre("support")
def telegram_media_to_messages(message, context, run):
    image_urls = []
    file_urls = []

    for attachment in message.attachments:
        if attachment.type == "photo":
            image_urls.append(resolve_telegram_file_url(attachment.payload))
        elif attachment.type == "document":
            file_urls.append(resolve_telegram_file_url(attachment.payload))

    if image_urls or file_urls:
        content = [{"type": "text", "text": message.text or "Analyze the media."}]
        content.extend(
            {"type": "image_url", "image_url": {"url": url}}
            for url in image_urls
        )
        # Add file blocks here when the target provider supports them.
        run.messages = [{"role": "user", "content": content}]

    return run
```

This produces an Agent call shaped like:

```python
await agent.acall(
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Analyze the media."},
                {"type": "image_url", "image_url": {"url": "https://..."}},
            ],
        }
    ],
)
```

`task` and `task_multimodal` are still valid Agent-level inputs, but they should
not be mixed accidentally with `messages`. If application code chooses that path,
it must clear `run.messages`, extract the text/caption into `run.kwargs["task"]`,
and set `run.kwargs["task_multimodal"]`.

The important invariant is: each social run should use either `messages` or
`task` plus `task_multimodal`, not `messages` plus `task_multimodal` without
`task`.

`SocialMessage.content` remains an adapter escape hatch for already-normalized
content blocks. When it is present, the boundary prepares the default user
message with:

```text
content = message.content if present else message.text or ""
```

The boundary will pass that list directly as the user message content:

```python
{"role": "user", "content": message.content}
```

The current Telegram adapter does not download media. It preserves Telegram
`photo`, `document`, `audio`, `voice`, `video`, and `sticker` payloads as
`SocialAttachment` records. Application code can inspect those attachments in a
pre-processor, download or resolve the file through the platform API, and set
`run.kwargs["task_multimodal"]`.

For media-only messages, if no pre-processor converts attachments into
`task_multimodal`, `message.content`, or `run.messages`, the Agent receives an
empty user content string. This is deliberate: downloading files requires
application policy for size limits, storage, allowed media types, retention, and
credentials.

## Telegram Adapter

`TelegramAdapter` implements the generic social adapter contract.

### Verification

Telegram webhook verification uses `X-Telegram-Bot-Api-Secret-Token`. The secret
is configured in the application and passed to Telegram when setting the webhook.
After that, Telegram includes the same value on each webhook request.

If no secret is configured, verification returns `True`. That is useful for local
experiments but should not be used for production webhooks.

### Decode

The adapter accepts `message` and `edited_message` updates. It extracts text from
`text` or `caption`, keeps media metadata in `attachments`, and ignores updates
that cannot produce a useful social message.

Telegram identity mapping:

```text
id = "telegram:{update_id}:{message_id}"
channel = "telegram"
session_id = "telegram:{chat_id}"
conversation_id = "{chat_id}"
sender_id = "{from.id}" or chat_id fallback
```

For private chats, `session_id` identifies the private conversation with the bot.
For groups, `session_id` identifies the group chat. This is why group `/cancel`
stops the active request for that group, not just for the individual sender.

### Send

Outbound text is sent with Telegram `sendMessage`. Telegram has a message length
limit, so the adapter splits text into chunks of 4096 characters.

The default sender calls Telegram's Bot API directly. Tests and custom
integrations can pass `sender=...` to intercept outbound messages without network
calls.

## Why The Boundary Does Not Own History

The Social Boundary currently treats each run as one inbound message or one
merged debounce batch. It does not store conversation history.

That is intentional for this branch. History has different lifecycle rules:

- it needs persistence or a checkpointer
- it needs trimming and TTL policies
- it should include tool calls, assistant messages, and user messages
- it must decide how much context to load for each `session_id`

Those concerns belong in the future checkpoint/history feature, not in the
platform adapter.
