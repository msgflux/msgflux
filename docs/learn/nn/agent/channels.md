# Agent Channels

`AgentChannel` connects an application's input boundary to an
[`AgentService`](service.md). It provides authorization, ordered preprocessing,
registered commands, and output formatting. The service continues to own the
Agent, its executions, and conversation history.

This API is independent of HTTP and platform SDKs. An adapter verifies an
incoming token or webhook signature, identifies the sender, selects a thread,
and calls the channel. Telegram, Slack, and Discord adapters are not included.

## Connect A Channel To A Service

```python
import asyncio

import msgflux as mf
from msgflux.channels import AgentChannel, ChannelRequest
from msgflux.data.stores import InMemoryCheckpointStore
from msgflux.nn import Agent
from msgflux.runtime.service import AgentService, AgentSession, SQLiteServiceStore


async def main():
    journal = SQLiteServiceStore()
    service = AgentService(store=journal)

    def create_session(thread_id):
        model = mf.Model.chat_completion("openai/gpt-6-luna")
        agent = Agent(
            name="main",
            model=model,
            checkpoint_store=InMemoryCheckpointStore(),
        )
        return AgentSession(agent, on_close=model.aclose)

    service.register("main", create_session)

    def authorize(context, request):
        return (
            context.principal == "alice"
            and request.agent_id == "main"
            and request.thread_id == "alice-project"
        )

    channel = AgentChannel(service, name="web", authorize=authorize)
    try:
        admission = await channel.prompt(
            ChannelRequest("main", "alice-project", "Explain the project briefly."),
            principal="alice",       # Established by the application's authentication
            request_id="message-1",  # Stable source message ID, reused on retry
        )
        receipt = admission.receipt
        settled = await service.wait(receipt.thread_id, receipt.request_id)
        print(settled.status)
        snapshot = await service.snapshot(receipt.thread_id)
        print(snapshot.messages)
    finally:
        await service.aclose()
        journal.close()


asyncio.run(main())
```

Set `OPENAI_API_KEY` before running. The example admits one message and waits
for its execution; the initial receipt normally has status `accepted`.
Memory-backed stores in this example do not survive a process restart.
Persistent journal and checkpoint stores remain configured by the service host.

The channel borrows the service. It does not close it or create a second worker,
conversation store, execution scope, or credential store. `AgentServiceClient`
from the [native HTTP adapter](service-http.md) can also be passed as `service`:
the channel uses the same `open_thread()` and `prompt()` contracts.

## Authorization And Origin Identity

Authorization is required and must return a `bool`, synchronously or
asynchronously. Returning `False` raises `ChannelPermissionError`; any other
return type raises `TypeError`. The callback runs before preprocessing and
again with the final request, before commands, thread creation, or admission.
This also checks a destination changed by preprocessing.

`ChannelContext`, `ChannelRequest`, `ChannelAdmission`, and `ChannelReply` are
immutable `msgspec.Struct` records:

| Record | Fields |
| --- | --- |
| `ChannelContext` | `channel`, authenticated `principal`, source `request_id` |
| `ChannelRequest` | `agent_id`, explicit `thread_id`, `prompt` |
| `ChannelAdmission` | `context`, effective `request`, service `receipt` |
| `ChannelReply` | `context`, effective `request`, formatted `content` |

The context contains non-secret origin identifiers. It is passed only to
registered host callbacks; it is not injected into Agent tools. The service's
trusted `AgentSession` factory and workspace permissions still determine what
the Agent may do. A channel's permission to send a prompt does not grant tool
permissions or change the Agent's execution principal.

An adapter must authenticate before supplying `principal`. Do not take it from
an unverified request body. Authorization must cover the selected agent and
thread, including shared conversation access. Native service credentials are
host credentials; keep the native HTTP endpoint private if an application has
multiple users. That endpoint does not automatically apply a channel's policy.

## Preprocess Requests

```python
import msgspec


@channel.register_preprocessor
def normalize(context, request):
    return msgspec.structs.replace(request, prompt=request.prompt.strip())
```

Processors run in registration order and must return a `ChannelRequest`.
Both sync and async functions are supported. They may change the prompt or
destination, but cannot replace origin identity. Requests and callback results
are validated before admission. A pipeline captures its registrations before
its first await; registrations added during that request affect later requests.

Keep preprocessing and routing deterministic for a given source identity.
Processors execute on retries too; durable Agent admission does not deduplicate
arbitrary callback side effects.

## Register Commands

```python
def help_command(context, request, arguments):
    return "Send a message to work on the project."


channel.register_command("help", help_command)

reply = await channel.prompt(
    ChannelRequest("main", "alice-project", "/help"),
    principal="alice",
    request_id="help-1",
)
print(reply.content)
```

Commands are registered by their exact name, without the leading slash.
The first token of the processed prompt selects a command; the remaining text
is passed as `arguments`. A command returns a string, optionally asynchronously.
Its result runs through postprocessors and becomes a `ChannelReply`, without
creating a thread or an Agent admission. Duplicate names are rejected.

Only registered commands are intercepted. An unregistered slash token remains
ordinary prompt text, so file paths and other text beginning with `/` remain
usable. There are no default commands. Commands with side effects need their
own idempotency strategy; they are not covered by the admission journal.
Interface commands such as copy or quit belong in their respective frontend.

## Format Output And Observe Executions

```python
@channel.register_postprocessor
def presentation(context, text):
    return f"Project assistant: {text}"


# `admission` is returned by channel.prompt() for an Agent input.
# `final_text` is the final answer collected by the application's observer.
reply = await channel.format(admission, final_text)
await adapter.send(reply.content)
```

Postprocessors run in order and must return a string, synchronously or
asynchronously. `format()` transforms presentation text only; it does not modify
messages or checkpoints. It rejects an admission from another channel.
Whole-answer formatting is separate from incremental event formatting.
The `adapter` and `final_text` in this fragment represent application code.

Use the receipt's `thread_id`, `request_id`, and `run_id` with the service's
existing wait, receipt, watch, interrupt, or steer APIs. When handling both input
types, branch on `isinstance(result, ChannelReply)` before accessing a receipt.
An observer can detach and reconnect for a fresh snapshot while execution
continues. See [HTTP/SSE observation](service-http.md) for remote clients.
The channel does not retry executions or deliver messages itself.

Applications must authorize observation and control endpoints as well as input.
Filter events by the intended run and root source; child Agents may emit events
too. Commentary, tool activity, and the final answer remain distinct native
events for the frontend to present.

## Shared Conversations And Retries

Channels share a conversation by selecting the same explicit service thread.
The thread remains bound to one agent. A second channel uses the same checkpoint
history and execution controls; it does not copy history into a new Agent.

Source message identities are scoped by `(channel, principal, request_id)` and
encoded into the service receipt's request ID. Use the receipt's ID for service
lookups, not the source ID. Two platforms can each deliver `message-1` without
colliding. Repeating the same identity and processed prompt in the same thread
returns the same run; changing the prompt raises `ServiceConflictError`.

Idempotency remains scoped to a thread. Changing routing on retry can select a
different admission: persist platform conversation bindings and keep routing
stable. The channel does not infer thread IDs from text or generate temporary
conversation bindings. Only one foreground run per thread is admitted at once.

SQLite admissions can survive a service restart when the host reopens the same
journal. Recovering unfinished execution still follows the service's explicit
recovery rules. A repeated delivery does not establish worker quiescence or
automatically resume an uncertain run.

Sending a platform response is a separate effect. Use a durable delivery outbox
when duplicate sends or delivery failures must be reconciled; do not start a
new model run merely because delivery failed. No outbox is included in this API.
