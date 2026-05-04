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
  -> registry.social_command / registry.social_route
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
  -> adapter.decode(...)
  -> publish SocialEvent for each decoded SocialMessage
  -> return accepted event count
```

The webhook route returns quickly. Actual agent work happens in the boundary
consumer task. This matters because social platforms generally expect a fast HTTP
acknowledgement and may retry when webhook responses are slow.

## Event Consumer

`SocialBoundary.start()` creates a consumer task when at least one adapter exists.
The consumer reads from `InMemorySocialEventBus` and calls `process_event(...)`.

`process_event(...)` does the synchronous application decisions before starting a
run:

```text
build SocialContext
  -> handle command
  -> reject if a run is already active for session_id
  -> optional debounce
  -> route to agent
  -> create active task for the session
```

The active task map is keyed by `message.session_id`. This is the unit of
cancellation and concurrency protection.

## Commands Before Agents

Commands are deliberately handled before routing. The model should not decide
what `/start`, `/cancel`, or `/stop` means.

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

Debounce does not conflict with future notifications. Debounce is pre-run
coalescing. Notifications are for messages that arrive while an agent is already
working.

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

The boundary runs:

```text
registry.auth_handler()
  -> registry.authorizers(agent_name)
  -> registry.check_rate_limits(...)
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
- converting Telegram attachments into chat-completion multimodal content
- applying per-tenant tool filters or model preferences

The boundary itself does not persist history and does not load previous messages.
Future checkpointing should use `session_id` to decide which conversation/thread
history belongs to the run.

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
