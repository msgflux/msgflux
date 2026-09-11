# Hooks & Guards

Hooks are the primary mechanism for intercepting and validating data flowing through Modules. The `Hook` base class provides the interface, and `Guard` is a built-in hook for input/output validation. Hooks can target a stable lifecycle event, the module execution boundary (`forward`), or a specific method on the module.

## Lifecycle Hooks

Lifecycle hooks attach to a named execution boundary instead of a Python method.
This keeps extensions independent from internal method names and allows the same
hook to work in synchronous and asynchronous execution.

```python
from dataclasses import replace

from msgflux.nn.hooks import Hook


def expand_references(ctx):
    return replace(
        ctx,
        output=ctx.output.replace("artifact://report", "the complete report"),
    )


output_hook = Hook(
    event="transform_output",
    handler=expand_references,
)
```

Pass the hook through the module's `hooks` argument. When the module reaches
`transform_output`, the handler receives the current payload. Returning a value
replaces the payload for the next handler; returning `None` leaves it unchanged.
Handlers run in registration order.

The Agent currently exposes these lifecycle boundaries:

| Event | Payload | Replacement behavior |
| --- | --- | --- |
| `before_run` | `BeforeRun` | May replace the message or call arguments before a fresh run is prepared. |
| `before_resume` | `BeforeResume` | May replace restored messages, model preference, or non-identity scope fields before execution resumes. |
| `transform_context` | `ConversationContext` | May replace the model-visible messages for the current request. |
| `before_compaction` | `BeforeCompaction` | May approve or skip automatic compaction after token estimation and before the Model creates a complete context view. |
| `transform_notifications` | `NotificationContext` | May filter or replace non-control notifications before they enter model context. |
| `transform_tool_catalog` | `ToolCatalogContext` | May replace the logical tool catalog before prompt rendering and provider compilation. |
| `transform_system_prompt` | `ModelContext` | May replace the rendered prompt; includes `scope`, `vars`, and a read-only view of the active tool catalog. |
| `before_request` | `ModelRequestContext` | May replace provider-neutral request parameters immediately before the LM call. |
| `after_response` | `ModelResponseContext` | May replace a settled, non-streaming response before it enters history. |
| `before_tool` | `BeforeTool` | May replace model-visible arguments or block local execution. |
| `before_dispatch` | `BeforeToolDispatch` | May block a validated call or reduce background/detached dispatch to foreground. |
| `after_tool` | `AfterTool` | May replace the result or error before it becomes a tool result. |
| `resolve_tool_feedback` | `ToolFeedbackContext` | May continue the model loop or return an Agent result after a batch of tool outcomes. The first return decision stops this hook chain. |
| `before_run_end` | `RunEndContext` | May inspect or replace the terminal outcome immediately before the final checkpoint. |
| `after_run_end` | `RunEndContext` | Runs after the final checkpoint is committed. |
| `transform_output` | `OutputContext` for Agents; settled value for other Modules | May replace only the value presented to the caller. |

`before_compaction` is inactive unless an extension or application hook is
registered for it. The built-in `CompactionExtension` uses this boundary to
apply `CompactionPolicy`; returning `action="skip"` prevents both the
compaction call and its execution events. The hook controls the decision, while
the Model remains responsible for counting tokens and producing the compacted
view. See [Conversation Compaction](compaction.md).

`before_run_end` participates in the terminal commit and may still turn a
successful run into a failure. `after_run_end` runs after that commit. If its
handler raises, the committed output and outcome remain primary and the hook
failure produces a runtime warning. The run is not executed or checkpointed a
second time. A completed turn therefore cannot be silently rewritten as failed
by an observational hook.

Import typed payloads from `msgflux.nn.hooks`. Dataclass replacement keeps a
handler explicit and preserves fields added to the contract later:

```python
from dataclasses import replace

from msgflux.nn.hooks import BeforeTool, Hook


def authorize_tool(event: BeforeTool):
    if event.tool_name == "delete_record":
        return replace(event, block="Deletion requires approval")
    return event


agent = Agent(
    name="operator",
    model=model,
    tools=[delete_record],
    hooks=[Hook(event="before_tool", handler=authorize_tool)],
)
```

`before_tool` runs after the tool has been resolved and its arguments prepared.
It receives only model-visible arguments. Runtime injections such as
`runtime_inputs` bindings remain attached to the
call and are not exposed to or removed by the hook. If a `before_tool` handler
raises or returns an invalid payload, execution fails closed and the tool is not
called. An `after_tool` handler failure leaves the original outcome unchanged
and emits a `handler.error` execution event.

Handlers are sequential for each call. When a `before_tool` or
`before_dispatch` handler sets `block`, later handlers for that event and that
call do not run. Other tool calls from the same model response continue through
their own hook chains. The reason becomes the tool error returned to the model,
and the runtime emits `tool.blocked`; it does not emit `tool.start`, `tool.end`,
or call `after_tool` because execution never began.

`before_dispatch` runs after `ToolExecutionPlan` selects `foreground`,
`background`, or `detached`. Its payload exposes public
arguments and normalized tool config, but not runtime injections. A handler may
keep the selected mode, reduce `background` or `detached` to `foreground`, or
block. It cannot redirect a foreground call into detached execution.

Use `before_resume` to restore extension-owned resources or transform the
conversation loaded from a checkpoint. A replacement must remain a
`BeforeResume` instance. It may change `messages`, `model_preference`, and
non-identity scope fields such as the abort signal, but it cannot change the
restored `thread_id`, `namespace`, or `run_id`.

The tool catalog on `ModelContext` lets prompt extensions describe the tools
available to the current request. It is contextual and read-only at this
boundary. Use `transform_tool_catalog` when an extension needs to change the
request tool surface; the transformed catalog is used both for prompt guidance
and provider compilation.

`ToolCatalogContext.catalog` is an immutable `ToolCatalogView`. Filter its
regular entries with `with_tools(...)` instead of rebuilding schemas:

```python
from dataclasses import replace

from msgflux.nn.hooks import Hook
from msgflux.tools import ToolCatalogView


def expose_read_only_tools(ctx):
    assert isinstance(ctx.catalog, ToolCatalogView)
    allowed = {"search_orders", "lookup_customer"}
    names = (
        entry.name
        for entry in ctx.catalog.tool_entries()
        if entry.name in allowed
    )
    return replace(ctx, catalog=ctx.catalog.with_tools(names))


catalog_hook = Hook(
    event="transform_tool_catalog",
    handler=expose_read_only_tools,
)
```

The returned view preserves stable tool references, native bindings, loading
state, and thread identity. `with_choice(...)` can apply a provider-neutral
selection after filtering. The Agent sends the view unchanged to the Model;
the concrete provider adapter compiles it to its wire protocol.

`transform_notifications` receives only ordinary progress and task
notifications. Control messages such as pause and interrupt are handled before
the hook and cannot be suppressed by an extension.

`before_run_end` and `after_run_end` receive the same terminal context for
completed, failed, interrupted, and paused non-streaming runs. The final
checkpoint is written between these boundaries. A direct `ModelStreamResponse`
settles later in its stream finalizer, so these hooks do not run when that
stream is handed directly to the caller; use `stream_events()` when lifecycle
observation must include the complete streamed execution.

```python
agent = Agent(
    name="reporter",
    model=model,
    hooks=[output_hook],
)
```

An async handler is awaited during async execution:

```python
async def load_artifacts(ctx):
    expanded = await artifact_store.expand(ctx.output, tenant=ctx.vars["tenant"])
    return replace(ctx, output=expanded)


output_hook = Hook(
    event="transform_output",
    handler=load_artifacts,
)
```

Lifecycle hooks observe the active `AbortSignal` before and after synchronous
handlers. During async execution, aborting the signal also cancels an in-flight
async handler. This lets hooks perform network or LM work without delaying
execution cancellation. Prefer `agent.acall(...)` or `agent.stream_events(...)`
for extensions that use async handlers; synchronous execution rejects an
awaitable handler.

For a removable package containing several hooks or tools, use an
[Agent Extension](extensions.md). The `hooks=` argument remains the direct API
for one-off interception and guards.

### Canonical Responses vs. Presented Output

`after_response` and `transform_output` serve different purposes:

| Event | Changes checkpoint history | Visible to the model later | Changes user output |
| --- | --- | --- | --- |
| `after_response` | Yes | Yes | Yes |
| `transform_output` | No | No | Yes |

Use `transform_output` for presentation work such as replacing an artifact
reference with the file contents. The Agent commits the original response to
`ChatMessages` first, then transforms the value returned to the caller:

```python
import asyncio

import msgflux as mf
from msgflux.chat_messages import ChatMessages
from msgflux.nn import Agent
from msgflux.nn.hooks import Hook


REFERENCE = "artifact://incident-report"
REPORT = "Incident report: scanner recovered; reconciliation pending."


def expand_artifact_reference(ctx):
    return replace(ctx, output=ctx.output.replace(REFERENCE, REPORT))


async def main():
    history = ChatMessages()
    agent = Agent(
        name="reporter",
        model=mf.Model.chat_completion(
            "openai/gpt-5.6-luna",
            api_mode="responses",
            store=False,
        ),
        system_prompt=f"Reply with exactly {REFERENCE} and nothing else.",
        hooks=[Hook(event="transform_output", handler=expand_artifact_reference)],
        config={"stream": True},
    )

    events = [
        event
        async for event in agent.stream_events(
            "Return the incident report reference.",
            messages=history,
        )
    ]

    message_end = next(event for event in events if event.type == "message.end")
    run_end = next(event for event in events if event.type == "run.end")
    assert message_end.data["content"] == REPORT
    assert run_end.data == {"outcome": "completed"}

    assistant = [
        item for item in history.to_chatml() if item["role"] == "assistant"
    ][-1]
    assert assistant["content"] == REFERENCE


asyncio.run(main())
```

When the async `stream_events()` iterator consumes a provider stream, a
registered `transform_output` hook puts assistant content in buffered mode.
Tool and reasoning events remain live, raw `message.delta` events are withheld,
and the transformed complete value is emitted once in `message.end`.
`to_chatml()` is used above only to make the assertion provider-independent;
the canonical timeline may store Responses content as typed content blocks.

For direct token streaming through `ModelStreamResponse`, no complete-output
transformation is applied. Use the execution event stream when a hook requires
the fully accumulated assistant output.

`event=...` cannot be combined with `on=...` or `method=...`. Forward and
method hooks remain available as lower-level extension points, while lifecycle
events are the recommended contract for agent execution boundaries.

## Guard

A `Guard` validates inputs and/or outputs of a Module. Each Guard wraps a **validator** callable and defines:

- **`on`** — when to run: `"pre"` (before execution) or `"post"` (after execution).
- **`message`** — controls the reaction when `safe=False`:
    - **With `message`** — short-circuits the pipeline and returns the message as the response (the model is never called).
    - **Without `message`** (default) — raises `UnsafeUserInputError` (pre) or `UnsafeModelResponseError` (post).
- **`target`** — submodule to register on. Defaults to `"generator"` for forward hooks and `None` for method hooks.
- **`method`** — optional method name. When omitted, the guard runs around `forward`. When set, it runs around that method instead.
- **`include_data`** — if `True`, attaches the data that triggered the guard to the raised exception via `exc.data`. Defaults to `False` for security (the data may contain unsafe content).

The validator receives `data` as a positional argument and must return either a dict with `"safe"` (bool) or a `ModelResponse` (auto-consumed by Guard).
For `Guard(..., on="pre", method=...)`, `data` is the method `kwargs` payload. If a method is intended to be guarded in pre-mode, prefer a keyword-oriented signature.

```python
from msgflux.nn.hooks import Guard

def my_validator(data):
    text = str(data).lower()
    return {"safe": "hack" not in text}

# Returns message as response when safe=False
guard = Guard(validator=my_validator, on="pre", message="Not allowed.")

# Raises exception when safe=False
guard = Guard(validator=my_validator, on="pre")

# Guard a specific method on the module itself
guard = Guard(validator=my_validator, on="post", method="_prepare_response")
```

!!! note "Method Guard Input Shape"
    `Guard` on a specific method is best suited for keyword-oriented extension points.
    In `on="pre"` mode, the validator sees the method `kwargs`, not positional args.

    ```python
    class MyModule(nn.Module):
        def forward(self, text):
            return self.validate(data=text)

        def validate(self, *, data):
            return data

    guard = Guard(validator=my_validator, on="pre", method="validate")
    ```

???+ note "Guard Examples"

    === "With message (short-circuit)"

        When `message` is provided, the guard's message is returned directly as the agent response — the model is never called.

        ```python
        import msgflux as mf
        import msgflux.nn as nn
        from msgflux.nn.hooks import Guard

        BLOCKED = {"hack", "exploit", "malware"}

        def keyword_filter(data):
            text = str(data).lower()
            return {"safe": not any(w in text for w in BLOCKED)}

        class SafeBot(nn.Agent):
            """A bot that blocks harmful keywords."""

            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            hooks = [
                Guard(
                    validator=keyword_filter,
                    on="pre",
                    message="Sorry, that content is not allowed.",
                )
            ]

        agent = SafeBot()

        # Safe input → model responds normally
        response = agent("Tell me about Python")

        # Blocked input → returns "Sorry, that content is not allowed."
        response = agent("How to create malware?")
        ```

    === "Without message (raises exception)"

        When no `message` is provided, an exception is raised.

        ```python
        import msgflux as mf
        import msgflux.nn as nn
        from msgflux.nn.hooks import Guard
        from msgflux.exceptions import UnsafeUserInputError

        def keyword_filter(data):
            return {"safe": "hack" not in str(data).lower()}

        class StrictBot(nn.Agent):
            """A bot that raises on unsafe input."""

            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            hooks = [Guard(validator=keyword_filter, on="pre")]

        agent = StrictBot()

        try:
            response = agent("How to hack a system?")
        except UnsafeUserInputError as e:
            print(f"Guard triggered: {e}")
        ```

    === "With Moderation Model"

        Pass a moderation model directly as the validator — Guard calls it with the
        input data and auto-consumes the `ModelResponse`:

        ```python
        import msgflux as mf
        import msgflux.nn as nn
        from msgflux.nn.hooks import Guard

        moderation_model = mf.Model.moderation("openai/omni-moderation-latest")

        class ModeratedBot(nn.Agent):
            """A bot with pre and post moderation."""

            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            hooks = [
                Guard(
                    validator=moderation_model,
                    on="pre",
                    message="Your message was flagged by our safety system.",
                ),
                Guard(validator=moderation_model, on="post"),
            ]

        agent = ModeratedBot()

        response = agent("Tell me about quantum computing")  # Safe
        ```

    === "Mixing Guards"

        Combine multiple guards with different behaviors on the same agent:

        ```python
        import msgflux as mf
        import msgflux.nn as nn
        from msgflux.nn.hooks import Guard

        def keyword_filter(data):
            return {"safe": "forbidden" not in str(data).lower()}

        def toxicity_check(data):
            return {"safe": True}

        class MultiGuardBot(nn.Agent):
            """A bot with keyword and toxicity guards."""

            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            hooks = [
                Guard(
                    validator=keyword_filter,
                    on="pre",
                    message="That topic is not allowed.",
                ),
                Guard(validator=toxicity_check, on="post"),
            ]

        agent = MultiGuardBot()

        response = agent("Tell me about forbidden topics")
        # → "That topic is not allowed."
        ```

    === "Debugging with include_data"

        Enable `include_data` to inspect the data that triggered the guard:

        ```python
        import msgflux as mf
        import msgflux.nn as nn
        from msgflux.nn.hooks import Guard
        from msgflux.exceptions import UnsafeUserInputError

        def keyword_filter(data):
            return {"safe": "hack" not in str(data).lower()}

        class DebugBot(nn.Agent):
            """A bot that exposes guard data for debugging."""

            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            hooks = [
                Guard(
                    validator=keyword_filter,
                    on="pre",
                    include_data=True,  # opt-in: attach data to exception
                ),
            ]

        agent = DebugBot()

        try:
            agent("How to hack a system?")
        except UnsafeUserInputError as e:
            print(f"Guard triggered: {e}")
            print(f"Offending data: {e.data}")  # available only with include_data=True
        ```

## Custom Hooks

The `Hook` base class allows creating custom hooks beyond guards. Implement `__call__` (sync) and optionally override `acall` (async). By default, `acall` runs `__call__` in an executor.

### Hook Interface

| Attribute | Description |
|---|---|
| `on` | `"pre"` (before execution) or `"post"` (after execution) |
| `target` | Submodule name to register on. `None` = the module itself |
| `method` | Method name to register on. `None` = the module execution boundary (`forward`) |
| `processor_key` | Key for processor matching in `_set_hooks`. `None` = no processor |
| `__call__` | Sync hook — called by the sync hook dispatcher |
| `acall` | Async hook — called by the async hook dispatcher |

### Hook Signatures

```python
# Pre hook — receives module, args and kwargs before execution
def __call__(self, module, args, kwargs, output=None): ...

# Post hook — receives module, args, kwargs and the output
def __call__(self, module, args, kwargs, output=None): ...
```

Both pre and post hooks share the same signature. For pre hooks, `output` is always `None`. The same signature is used for forward hooks and method hooks.

???+ note "Custom Hook Examples"

    === "Logging Hook"

        ```python
        from msgflux.nn.hooks import Hook

        class LoggingHook(Hook):
            """Logs every call to the module."""

            def __init__(self):
                super().__init__(on="pre", target=None)

            def __call__(self, module, args, kwargs, output=None):
                print(f"[{module.__class__.__name__}] called with {len(kwargs)} kwargs")
        ```

    === "Timing Hook"

        ```python
        import time
        from msgflux.nn.hooks import Hook

        class TimingHook(Hook):
            """Measures execution time of the generator."""

            def __init__(self):
                super().__init__(on="post", target="generator")
                self.start_time = None

            def __call__(self, module, args, kwargs, output=None):
                elapsed = time.time() - self.start_time
                print(f"Generator took {elapsed:.2f}s")
        ```

    === "Hooking an Internal Method"

        ```python
        from msgflux.nn.hooks import Hook

        class PrepareResponseHook(Hook):
            """Intercept Agent._prepare_response."""

            def __init__(self):
                super().__init__(on="post", method="_prepare_response")

            def __call__(self, module, args, kwargs, output=None):
                print("response_type:", kwargs["response_type"])
                return output
        ```

    === "Token Counter Hook"

        ```python
        from msgflux.nn.hooks import Hook

        class TokenCounterHook(Hook):
            """Tracks cumulative token usage from model responses."""

            def __init__(self):
                super().__init__(on="post", target="generator")
                self.total_tokens = 0

            def __call__(self, module, args, kwargs, output=None):
                if hasattr(output, "usage"):
                    self.total_tokens += output.usage.get("total_tokens", 0)
        ```

    === "Async Custom Hook"

        ```python
        import httpx2
        from msgflux.nn.hooks import Hook

        class AsyncWebhookHook(Hook):
            """Sends a webhook notification after every call."""

            def __init__(self, webhook_url):
                super().__init__(on="post", target="generator")
                self.webhook_url = webhook_url

            def __call__(self, module, args, kwargs, output=None):
                pass  # sync fallback — no-op

            async def acall(self, module, args, kwargs, output=None):                
                async with httpx2.AsyncClient() as client:
                    await client.post(self.webhook_url, json={"status": "ok"})
        ```

### Hook Registration

All `nn.Module` subclasses support the `hooks` class attribute. Each hook declares where it registers:

- `target="generator"` (default for Guard) — registers on the internal `Generator` wrapper
- `target=None` — registers on the module itself
- `method="_prepare_response"` — registers on that specific method instead of `forward`

```python
import msgflux.nn as nn

class Bot(nn.Agent):
    model = model
    hooks = [input_guard, output_guard, logging_hook]

agent = Bot()
```

You can also register hooks manually via `hook.register()`:

```python
import msgflux as mf
import msgflux.nn as nn
from msgflux.nn.hooks import Guard

class Bot(nn.Agent):
    model = model

agent = Bot()

def my_validator(data):
    return {"safe": "blocked" not in str(data).lower()}

guard = Guard(validator=my_validator, on="pre", message="Nope.")
handle = guard.register(agent.generator)  # returns RemovableHandle
handle.remove()  # unregister when done
```

You can also register declarative hooks for a specific method via `hooks=`:

```python
from msgflux.nn.hooks import Hook

class PrepareResponseHook(Hook):
    def __init__(self):
        super().__init__(on="post", method="_prepare_response")

    def __call__(self, module, args, kwargs, output=None):
        print("prepared response")
        return output

class Bot(nn.Agent):
    model = model
    hooks = [PrepareResponseHook()]
```

### Using Hooks with `nn.Module` Directly

The hook system is built into `nn.Module`. You can register hooks on any module via PyTorch-style methods for `forward`, or via method hooks for other methods:

```python
import msgflux.nn as nn

class Bot(nn.Agent):
    model = model

module = Bot()

# Register a plain function as a pre-hook
def my_pre_hook(module, args, kwargs):
    print("About to call forward")

handle = module.register_forward_pre_hook(my_pre_hook)

# Register a plain function as a post-hook
def my_post_hook(module, args, kwargs, output):
    print(f"Forward returned: {type(output)}")

handle = module.register_forward_hook(my_post_hook)
```

### Registering Hooks on Arbitrary Methods

Use `register_method_pre_hook()` and `register_method_hook()` when the extension point is an internal method rather than `forward`:

```python
import msgflux.nn as nn

class MyModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, text):
        return self._normalize(text)

    def _normalize(self, text):
        return text.strip().lower()

module = MyModule()

def normalize_pre_hook(module, args, kwargs):
    return (args[0] + "  ",), kwargs

def normalize_post_hook(module, args, kwargs, output):
    return f"[{output}]"

pre_handle = module.register_method_pre_hook("_normalize", normalize_pre_hook)
post_handle = module.register_method_hook("_normalize", normalize_post_hook)

result = module(" Hello ")
print(result)  # "[hello]"
```

This is useful for methods such as `Agent._prepare_response`, `Agent._prepare_inputs`, or other internal extension points that do not warrant being promoted to standalone `Module` instances.

!!! warning "Streaming Limitation"
    Guards with `on="post"` are **not compatible** with `stream=True`, since the full response is needed for validation. Using both raises a `ValueError` at initialization.
