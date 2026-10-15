# Local Service Discovery And Startup

`connect_local_service()` connects to a matching local [HTTP/SSE backend](service-http.md)
and starts one when no process owns the runtime. Several frontends can reuse the
same AgentService. Closing a client or exiting its launcher leaves the daemon
running, including while an Agent works or waits for approval.

Install `msgflux[service]` for server support. For the Coding harness, use
`msgflux[coding,service]` so workspace query tools (`grep`, `glob`, and
`ls`), including those selected by read-only agents, have their optional
dependencies available:

```bash
uv add 'msgflux[coding,service]'
```

The process manager uses POSIX locks; the native HTTP client remains usable on
other platforms.

## Automatic Endpoint Discovery

`connect_local_service()` ensures that the configured backend is ready before
returning an owned HTTP client. It checks authenticated health and instance
identity, reuses a matching server, or starts one when no process holds its
lifetime lock. Concurrent callers coordinate through the startup lock and receive
independent clients for the same daemon.

The default `~/.msgflux/runtime/daemon.json` stores the URL, including its actual
port, instance ID, PID, configuration identity and authentication token. The
server publishes this owner-only record atomically after its health check passes.
Graceful shutdown removes only the record belonging to that instance. If a
process crashes, discovery removes the stale record after verifying that no
process owns the lifetime lock and starts a replacement.

Use the client to discover the URL without reading the private record yourself:

```python
from msgflux.runtime.service.local import connect_local_service

async def discover_backend():
    client = await connect_local_service()
    try:
        return client.base_url
    finally:
        await client.aclose()
```

This example returns the validated backend URL. Closing the client releases its
HTTP pool and leaves the daemon running. Discovery is a step within connection;
it does not guarantee the server will remain alive for subsequent requests.

## Register Agents From TOML

Start the coding backend from your project directory:

```bash
uv run --extra service msgflux service
```

When the default `~/.msgflux/config.toml` does not exist, the launcher creates a
starter file. Set a model in that file before launching the service: msgFlux does
not guess a provider or model. Missing model configuration produces an actionable
error before a daemon starts or opens its store. You can provide the default at
initialization with `msgflux init --model provider/model-id`, or edit
`defaults.model` afterward. Once configured, construction remains lazy: starting
the service does not call a model or create a conversation directory. Coding
agents read their selected provider credential from a named account bundle only
when a model request needs it.

To create the file without starting the server:

```bash
uv run --extra service msgflux init
```

To initialize with a model already selected:

```bash
uv run --extra service msgflux init --model provider/model-id
```

For initialization from the SDK:

```python
from msgflux.runtime.service.local.manifest import (
    default_service_config_path,
    initialize_service_config,
)

config_path = initialize_service_config(
    default_service_config_path(),
    model="provider/model-id",
)
```

This creates `~/.msgflux/` and its starter config without starting the backend.
Call `await connect_local_service()` when you need a running service.

The `--model` option applies only when creating a new file. Initialization never
overwrites an existing config.

Without `--model`, the generated backend configuration contains:

```toml
[defaults]
agent = "coding"
account = "default"
# Choose an available provider/model, for example:
# model = "provider/model-id"
# reasoning_effort = "medium"
# accounts_dir = "./accounts"  # defaults to ~/.msgflux/accounts

[agents.coding]
harness = "coding"
subagents = ["lite-coding"]

[agents.lite-coding]
harness = "coding"

# Example read-only explorer:
# [agents.explorer]
# harness = "coding"
# permissions = ["read"]
```

`coding` and `lite-coding` are registered agent IDs, not profiles or modes.
`defaults.agent` selects the agent used by `open_thread()` when the caller omits
an agent ID. Reopening a thread retains its original agent binding. The service
registers all entries and resolves a separate AgentSession for each conversation
thread.

A coding entry can override `model` and `reasoning_effort` at agent level.
Otherwise it inherits the declared defaults; omitted `reasoning_effort` uses
`medium`. The runtime does not select another
model or account when a request fails. Credentials are not copied into the TOML
file.

`defaults.account` selects the account for new prompt executions and defaults to
`"default"`. `defaults.accounts_dir` optionally selects the credential bundle
root. A relative path is resolved from the config file's directory; when omitted,
the root is `~/.msgflux/accounts/`. Under that root, each account has a separate
directory and credential record for each provider. For example, configure an API key without
putting it in TOML:

```bash
msgflux accounts login openai --account-name work
msgflux accounts inspect work
```

The first command prompts for the key and stores it in the `work` bundle. The
second lists configured providers and status without printing credentials. The
`msgflux accounts` commands use `~/.msgflux/accounts/` unless `--accounts-dir`
is supplied. Set `defaults.accounts_dir` to use another root for the service.
Provider and account records are read from that root only; the coding harness
does not fall back to a process environment key or another account when a
credential is missing. API-key providers are discovered from msgFlux's model
provider registry. The managed native OAuth adapter currently supports
`openai-codex`; registering an API-compatible provider does not imply that its
OAuth login is supported.

This automatic resolver setup applies to coding harness entries. A custom
`factory` entry owns its model construction and must attach `Accounts.resolver()`
to models that should use managed credentials.

The CLI can select a different account for one new prompt. The account is
recorded with the admitted run, so retries with the same request ID reuse that
run and its account. Resume uses the recorded account even if
`defaults.account` has changed. Admissions created before account selection was
persisted keep an unknown account; managed authentication reports that identity
cannot be recovered and requires a new execution with an explicit selection.

### Builtin And Custom Subagents

Use a preset to supply builtin instructions, description and tool/permission
defaults. The Explorer investigates code and reports file references:

```toml
[agents.explorer]
harness = "coding"
preset = "explorer"
```

List `explorer` in a parent's `subagents` to expose it through AgentTool.
Explicit `description`, `permissions`, tools and `prompt_file` override preset
defaults, including empty tool/permission lists. A prompt file replaces preset
instructions while preserving host context and effective permissions. Models and
reasoning continue to inherit `[defaults]`; presets never choose credentials.

For a custom reviewer:

```toml
[agents.reviewer]
harness = "coding"
description = "Review changes for defects and missing tests."
prompt_file = "prompts/reviewer.md"
permissions = ["read"]
```

The same resolution is available to Python hosts:

```python
from msgflux.coding.config import coding_agent_options

explorer = coding_agent_options(
    "explorer", "provider/model-id", preset="explorer",
)
```

Use these resolved options with `create_coding_session_factory()` and a
`CodingAgentCatalog` as the existing factory does. Trusted application `factory`
entries own their construction; they do not accept coding presets or options.

### Tools And Subagents

`workspace` expands into the tools appropriate to the model. OpenAI models use
Bash, Read and ApplyPatch; other models use Bash, Read, Edit, Write and Delete.
Routed model IDs starting with `openai/` follow the same OpenAI selection.
Individual builtin names, such as `read`, `bash` or `web_fetch`, can also be
selected. An omitted tools table enables `workspace`; explicit `active = []`
permits no active tools. `active` sends the complete tool schema, while `deferred`
uses the existing tool-discovery flow.

Declaring `subagents` automatically adds AgentTool for those children; no `agent`
tool selector is needed. The coding example can dispatch to `lite-coding`; the
child uses its own configured model and inherits the parent's durable execution scope. Child calls return settled
output to AgentTool; the parent uses streaming for frontend observation. Child
references must name coding-harness entries and cannot form cycles. Individual
tools can be deferred. The harness explicitly enables default usage guidance for its configured tools.

### Workspace Permissions

Use `permissions` on each coding agent. Omitting it enables
`["read", "write", "execute"]`; `permissions = []` disables workspace access.

| Permission | Authority |
|------------|-----------|
| `read` | Read files, list directories, and search the workspace. |
| `write` | Create, edit, and delete workspace files and create directories. |
| `execute` | Run commands with the configured executor. |

`workspace` selects only tools permitted by this list. Without `execute`, it
adds Ls, Glob and Grep for navigation when `read` is enabled. Structured editing
tools require both `read` and `write` to prepare changes against existing files.
Explicitly selecting a tool in `active` or `deferred` without its required
permissions fails validation.

For a read-only explorer, set `permissions = ["read"]`:

```toml
[agents.coding]
harness = "coding"
subagents = ["lite-coding", "explorer"]
# permissions = ["read", "write", "execute"]  # default

[agents.explorer]
harness = "coding"
permissions = ["read"]
```

The explorer receives Read, Ls, Glob and Grep, with no Bash or editing tools.
Every descendant's permissions are intersected with its caller's permissions;
a child cannot grant itself additional authority. Restrictions are enforced by
the workspace, as well as by tool selection.

`execute` authorizes host commands with the local executor, which is not an OS
sandbox. Such commands can read or modify files independently of the `read` and
`write` grants for workspace file tools. Use only `["read"]` when you need a
read-only local agent.

### Prompt Defaults

If `prompt_file` is omitted, the coding harness uses its bundled system prompt.
The default stays in the library and follows library updates; initialization does
not write a `SYSTEM.md` copy. To replace its customizable instructions:

```toml
[agents.coding]
harness = "coding"
subagents = ["lite-coding"]
prompt_file = "prompts/coding.md"
```

The file path is resolved against the configuration file's directory. An
explicit missing, unreadable or invalid prompt fails clearly. Workspace context
and tool guidance are still assembled by the harness. The TOML example above
replaces the earlier `[agents.coding]` entry; do not declare the table twice.

### Custom Python Factories

Use `factory` instead of `harness` for agents constructed by application code:

```toml
[defaults]
agent = "assistant"

[agents.assistant]
factory = "my_agents:create_assistant"
```

Save `my_agents.py` in the launch directory or install it in the server environment:

```python
from pathlib import Path

import msgflux as mf
from msgflux.nn import Agent
from msgflux.runtime import AgentSession


async def create_assistant(thread):
    model = mf.Model.chat_completion(
        "openai-codex/gpt-6-luna", reasoning_effort="medium"
    )
    try:
        agent = Agent(
            name=thread.agent_id,
            model=model,
            agent_dir=Path.home() / ".msgflux",
            config={"stream": True},
        )
        return AgentSession(agent, on_close=model.aclose)
    except BaseException:
        await model.aclose()
        raise
```

Custom factories retain the `ServiceThread -> AgentSession` contract and choose
their models, prompts, tools and resource ownership in code. A factory can be
synchronous or asynchronous. Its module is imported only by the server, before
the admission journal opens; the frontend parses configuration without importing
application modules. `harness` and `factory` are mutually exclusive.

### Selecting And Initializing A File

```bash
uv run --extra service msgflux init --config ./agents.toml
uv run --extra service msgflux service --config ./agents.toml
```

Initialization publishes one complete file with owner-only permissions and never
overwrites an existing file, including one with invalid content. Concurrent
launchers preserve the file published first. Automatic creation applies only
when the implicit default is missing: an explicit nonexistent `--config` fails
until it is initialized. `--config` and `--factory` are mutually exclusive.

Relative config paths use the launch directory selected by `--cwd`. Relative
prompt paths use the config directory. The config's directory does not change
where application modules are imported or select a thread's workspace; pass that
project path when opening a thread. TUI appearance and key bindings belong in a
separate `tui.toml`; this backend file defines agents and service configuration.

Discovery identifies the config by its canonical path and a SHA-256 digest of
its contents. Editing a running daemon's file does not hot-reload it. A subsequent
connection reports a conflict until an explicit restart rereads the recorded
file. Restart preserves the active source kind and path; use a separate runtime
directory for another source.

Restart stops the current process before starting its replacement. If a new
factory cannot be imported, replacement startup fails and the service remains
stopped. Inspect `daemon.log`, correct the file or module, and reconnect. Admission
and conversation stores are retained. A config change between discovery and
startup fails the identity check. Restart does not provide uninterrupted handover
or roll back application code.

Built-in coding agents retain their local workspace identity in
`runtime_dir/workspaces.sqlite3`, keyed by the immutable thread ID. Recreating
a session or restarting the daemon preserves this identity so pending file
approvals can still be reviewed. The registry verifies the canonical root,
device and inode, backend and resource IDs, and the workspace configuration
revision. A changed root, replaced directory, or changed registered identity
raises `WorkspaceIdentityError` (also a `PermissionError`); the service factory
reports this as `ServiceRecoveryRequiredError` with the thread context and the
identity error as its cause. The service does not adopt the new directory or
retry with broader permissions. Keep this registry with the other durable
service files. Executions owned by a previous process still require trusted
host confirmation of quiescence before resuming.

For an embedded service, inspect the typed cause when an operation loads the
coding session. This assumes `service` and `thread_id` refer to an initialized
service and thread:

```python
from msgflux.runtime import ServiceRecoveryRequiredError, WorkspaceIdentityError

try:
    await service.workspace_policy(thread_id)
except ServiceRecoveryRequiredError as exc:
    if isinstance(exc.__cause__, WorkspaceIdentityError):
        print("Workspace identity needs recovery:", exc.__cause__)
    raise
```

`WorkspaceIdentityError` reports a known mismatch with a fixed public message;
in an embedded service, any underlying error remains available through its
cause. HTTP responses sanitize generic `PermissionError` messages.

## Define The Trusted Factory

For custom service construction, pass trusted application code as
`--factory module:callable`. This service factory receives the runtime directory as a `Path` and returns an AgentService,
directly or through an async function. Each registered Agent factory receives a
`ServiceThread` binding with its immutable `thread_id`, `agent_id`, and optional
project `cwd`. Models, credentials, workspace grants and tools are configured by
that code, rather than by discovery files or remote request bodies.

Save this as `my_backend.py` in your project:

```python
import re

import msgflux as mf
from msgflux.coding import CodingCheckpointExtension
from msgflux.data.stores import SQLiteCheckpointStore
from msgflux.nn import Agent
from msgflux.runtime import AgentService, AgentSession, AgentWorkspace, SQLiteServiceStore


def create_service(runtime_dir):
    service = AgentService(store=SQLiteServiceStore(runtime_dir / "service.sqlite3"))

    async def create_agent(thread):
        # Application policy keeps thread IDs safe as directory names.
        if not re.fullmatch(r"[A-Za-z0-9_-]+", thread.thread_id):
            raise ValueError("Unsupported persistent thread ID")
        if thread.cwd is None:
            raise ValueError("This coding service requires a project cwd")
        folder = runtime_dir.parent / "threads" / thread.thread_id
        folder.mkdir(parents=True, exist_ok=True)
        model = workspace = checkpoints = None

        async def close():
            if model is not None:
                await model.aclose()
            if workspace is not None:
                await workspace.aclose()
            if checkpoints is not None:
                checkpoints.close()

        try:
            model = mf.Model.chat_completion("openai/gpt-6-luna", reasoning_effort="medium")
            workspace = AgentWorkspace.local(thread.cwd)
            checkpoints = SQLiteCheckpointStore(str(folder / "checkpoints.sqlite3"))
            agent = Agent(
                name="main", model=model, workspace=workspace,
                checkpoint_store=checkpoints, config={"stream": True},
            )
            agent.register_extension("coding_checkpoints", CodingCheckpointExtension())
            return AgentSession(agent, on_close=close)
        except BaseException:
            await close()
            raise

    service.register("main", create_agent)
    return service
```

This example configures persistent admission and checkpoint stores, a workspace,
and the coding checkpoint policy. It leaves tools empty; pass the tools and
permissions appropriate to your application when constructing the Agent.
Provider credentials belong in the server environment or the provider's configured
authentication source. The daemon inherits its launcher's environment.

The runner owns the returned service and closes its admission journal on shutdown.
Factories must release resources on failure and provide cleanup callbacks for their
successful sessions. Callbacks must drain any delegated work they own before
closing its dependencies.

## Connect On Demand

```python
import asyncio
from pathlib import Path

from msgflux.runtime.service.local import connect_local_service


async def main():
    client = await connect_local_service(cwd=Path.cwd(), startup_timeout=30)
    try:
        health = await client.health()
        print("Runtime:", health.instance_id)
        thread = await client.open_thread(cwd=Path.cwd())
        receipt = await client.prompt(
            thread.thread_id, "Explain this project", request_id="explain-1"
        )
        print("Thread:", thread.thread_id, "Run:", receipt.run_id)
    finally:
        await client.aclose()


asyncio.run(main())
```

For custom factory entries or `--factory`, the module must be importable from a
stable installed location or the selected launch working directory. `connect_local_service(..., cwd=...)` supplies
the working directory used to import and launch the factory when a daemon needs to
start. If a matching daemon is already healthy, a caller from another directory
reuses it; its own launch `cwd` does not select a new Agent workspace. Pass the
desired project path to `client.open_thread("main", cwd=Path.cwd())` instead.
That path is canonicalized and stored in the immutable thread binding. Reopening
without a path returns the existing binding, while a different explicit path
conflicts. The trusted factory can reject missing or disallowed roots and decides
whether to create a workspace there; the path grants no permissions by itself.
The runtime does not change process working directory. Different project
frontends can reuse the same daemon and open separate threads with their own
`cwd`; the service factory remains registered once for the backend. Keep the
factory module in a stable installed location, and use `thread.thread_id` for its
checkpoint directory so project paths do not choose or collide in storage. Use a
separate runtime directory for separate application configurations or factories.

Concurrent launchers coordinate startup and return clients for the same ready
instance. Discovery requires a matching authenticated health identity before
returning a connection. Initialization and shutdown of an existing owner are waited
for within `startup_timeout`; an owner with published metadata but an invalid
identity/health response is reported as requiring recovery and is not replaced.

## Watch An Existing Thread

```bash
msgflux threads watch THREAD_ID
msgflux threads watch THREAD_ID --json
msgflux threads watch THREAD_ID --runtime-dir /path/to/runtime
```

`threads watch` connects to an existing thread without submitting a prompt. It
uses the same verified local daemon discovery as inspection commands and never
starts an absent service. The thread can have been created by an HTTP caller,
the SDK, or `msgflux -p`.

The connection first receives a snapshot and then live events. Plain output
prints a snapshot header with the thread ID and labels events by type and source.
`--json` writes the full snapshot and events as flushed NDJSON envelopes, using
the same `snapshot` and `event` formats as `msgflux -p --stream --json`.

Observation stays open after a run completes, so it can follow later runs on the
same thread. Ctrl+C exits with code 130 and closes only this observer; it does
not interrupt the agent. To request interruption, use
`msgflux threads interrupt THREAD_ID` separately.

Connection errors exit with a diagnostic. Run the command again to reconnect:
each connection receives a fresh snapshot followed by new events, without replay
of missed deltas. An idle or completed thread can still be observed; its saved
conversation is available in the JSON snapshot. A watcher does not emit a final
CLI `result` envelope and does not send another user message.

## Steer An Active Thread

```bash
msgflux threads steer THREAD_ID -m "Prioritize the integration tests"
msgflux threads steer THREAD_ID --message "Keep the public API unchanged" --json
```

`threads steer` selects the thread's active root run and sends a user message
through its existing AgentInbox. It does not submit another prompt or start a
new run. Like observation and inspection, it requires an existing verified local
daemon and accepts `--runtime-dir PATH`.

The response confirms that the message was queued, including `thread_id`,
`run_id`, and `notification_id`. The agent consumes it at its normal inbox drain
points, such as after a tool call; confirmation does not mean the model has read
it yet. Steering does not preempt an in-flight model request or interrupt tools.

An empty message, an idle or unknown thread, or an ambiguous active-run selection
is rejected. If the selected run ends before delivery, the service rejects that
target; the CLI does not redirect the message to a newer run. Use
`msgflux -p "Follow up" --thread THREAD_ID` for a new turn after completion.

## Validate Configuration Offline

```bash
msgflux config validate
msgflux config validate --config ./config.toml
msgflux config validate --json
```

With no `--config`, validation reads `~/.msgflux/config.toml`. It uses the same
manifest loader and coding catalog resolution as service startup. It checks TOML,
field types and unknown fields, defaults and agent references, model path syntax,
permissions, tool selectors, subagent references/cycles, and readable UTF-8 prompt
files. Reasoning configuration follows the same structural validation as the
manifest; this does not query model-specific provider capabilities.

Validation does not create a missing config, start a daemon, instantiate Agent
sessions or models, read credentials, refresh OAuth, or call providers. Custom
factory entries are validated for `module:attribute` syntax without importing or
executing the user's module. A successful result does not guarantee factory
importability, provider/model availability, or account access. Those remain
runtime checks.

Human output reports the config path, default agent, and declared agents.
Invalid configuration reports the loader's diagnostic on stderr. `--json` writes
one structured result to stdout, with `valid`, `config_path`, `default_agent`,
`agents`, and `errors`; invalid JSON results also stay on stdout and leave stderr
empty. Exit code `0` means valid and `1` means invalid. Validation stops at the
first error reported by the shared loader.

## List Agents And Threads

```bash
msgflux agents ls
msgflux agents ls --active --json
msgflux threads ls
msgflux threads ls --active --json
```

`agents ls` lists the connected service's registered agent IDs, with an
`ACTIVE_RUNS` count and `STATE` (`running` or `idle`). `--active` keeps only agents
with locally owned live root workers. A registered agent, an idle cached session,
or a persisted `running` receipt does not establish live execution. Subagent tasks
inside a root execution are not separate root service runs in this count.

`threads ls` lists persisted thread bindings: thread ID, agent ID, active run
count/state, and working directory. JSON also includes each binding's
`interactive` flag. `--active` keeps only threads with locally owned root workers.
An `idle` state means no live root worker in this service; approvals, unfinished
checkpoints or recovery requirements can still exist. Use `runs inspect` for
that evidence.

The commands use existing metadata and live worker APIs without constructing Agent
sessions, opening conversation checkpoints, or starting an absent daemon. Counts
are observations that can change immediately after collection; activity and
bindings are collected through separate HTTP requests, not an atomic combined
snapshot. Both support `--runtime-dir PATH` to select another local daemon.
If the daemon predates live-run inspection, these listings report that
`GET /v1/runs/active` is unsupported and ask you to restart or upgrade the server.
An authenticated health check confirms instance identity, not that every new
endpoint is present. After updating a checkout, explicitly restart its daemon
from that checkout with `msgflux service restart`; listing never upgrades or
restarts a running server automatically.

Human output uses aligned tables. JSON contains `{"agents": [...]}` or
`{"threads": [...]}`. An empty list exits successfully; an unavailable service
or invalid arguments produce an error on stderr and a nonzero exit code.

## Inspect An Agent's Loaded Defaults

```bash
msgflux agents inspect coding
msgflux agents inspect coding --json
```

Inspection queries the active server's registered configuration. It shows whether
this is the default agent, the service's default account name, and the available
configuration: model, reasoning effort, permissions, active/deferred tool defaults,
subagents, preset, prompt origin, and source config path/digest. Workspace tool
selectors have already been expanded for the configured model/provider and
permissions by the same coding catalog used to build sessions.

The result describes **registration defaults**, not a particular thread's live
Agent. Per-run account selection, restored checkpoint model state, workspace
policy overrides, deferred tools loaded during a conversation, and conditional
runtime tools remain session-specific. The coding harness can insert task
management, subagent dispatch, and interactive communication tools at runtime;
`tools_active` and `tools_deferred` describe the resolved configuration rather
than the exact tool schema sent to a model. Use run/thread inspection for their
runtime evidence.

For manifest-backed coding agents, `prompt_source` is `default`, `preset`, or
`file`. This identifies the source of custom instructions; the harness's base
system prompt is still present. Preset identity and prompt file path are included
when configured, but prompt contents are never returned. Custom factories only
publish their factory reference and source config metadata unless the host
explicitly supplies additional metadata. Unknown fields stay null rather than
being inferred by executing the factory.

The command does not start an absent server, construct sessions, create a
workspace/checkpoint, load a model, or read credentials or prompt contents. It
also does not reread the local TOML: editing it after startup leaves the loaded
registration unchanged until the service is restarted. The config digest lets
operators identify the loaded file version. An unknown agent returns a clear
error. An old daemon missing the inspection endpoint reports that a restart or
upgrade is required; it is never restarted automatically.

Human output is a labeled summary. JSON returns `AgentInspection`:

```json
{
  "agent_id": "coding",
  "is_default": true,
  "account_name": "work",
  "configuration": {
    "harness": "coding",
    "model": "provider/model-id",
    "reasoning_effort": "medium"
  }
}
```

This shortened example omits additional configuration fields. `configuration`
is null when a programmatic registration did not provide metadata. Inspection
exits with code `0` on success; unknown agents, unavailable services or request
failures exit with code `1` and diagnostics on stderr. `--runtime-dir PATH`
selects a different local daemon.

### SDK And Host Registration

The SDK exposes the same authenticated operation:

```python
inspection = await client.inspect_agent("coding")
print(inspection.account_name)
if inspection.configuration is not None:
    print(inspection.configuration.model)
```

This calls `GET /v1/agents/coding/inspection` and decodes `AgentInspection`. It
observes the server registration without opening a thread or invoking a model.
Embedded hosts can call the synchronous `service.inspect_agent("coding")`.

Programmatic factories can publish safe metadata explicitly:

```python
from my_application import make_session
from msgflux.runtime.service import (
    AgentConfiguration,
    AgentService,
    SQLiteServiceStore,
)

service = AgentService(
    store=SQLiteServiceStore("./service.sqlite3"),
    default_agent="reviewer",
    default_account="work",
)
service.register(
    "reviewer",
    make_session,
    configuration=AgentConfiguration(
        model="provider/model-id",
        permissions=("read",),
        tools_active=("read", "grep"),
        tools_deferred=(),
    ),
)
print(service.inspect_agent("reviewer"))
```

Here `make_session` is the application's existing `ServiceThread -> AgentSession`
factory. Registration validates and freezes the supplied metadata without
calling it. The host is responsible for keeping metadata consistent with its
factory; inspection does not validate the factory's implementation. Include
only public configuration, never secrets. The account name and default-agent
flag come from the service itself. Omitting `configuration` preserves lazy
registration and reports null metadata.

## Inspect The Daemon

Inspect the local service without starting it:

```bash
msgflux service status
msgflux service status --runtime-dir ~/.msgflux/runtime --json
```

An active service reports its verified URL, PID and instance ID. JSON output
contains `status`, `runtime_dir`, `url`, `pid`, and `instance_id`; it never
contains the authentication token. Exit code `0` means `running`. Exit code `1`
means `stopped` (no metadata) or `unreachable` (metadata exists but the
authenticated identity check failed). Unverified identities have no URL in the
result. Inspection creates no files and leaves stale metadata untouched.
Malformed or insecure metadata produces a concise error on stderr.

The SDK offers the same read-only operation:

```python
from msgflux.runtime.service.local import inspect_local_service

status = await inspect_local_service()
if status.status == "running":
    print(status.url)
```

This example discovers the active address without launching the daemon. Use
`connect_local_service()` when you want to start a service if none exists.

## Inspect Service And Runs

Use `service inspect` for the verified daemon identity and its recorded launch
configuration:

```bash
msgflux service inspect
msgflux service inspect --json
```

The human output includes runtime directory, address, PID, instance ID, launch
source and working directory. JSON contains `service` (the same status record as
`service status`), `factory`, `config_path`, and `cwd`. Authentication tokens are
never included. An absent or unverified daemon has null launch details.
Unlike `status`, inspection exits successfully when it reports `stopped` or
`unreachable`; invalid metadata or request errors still exit with code `1`.

Inspect a particular execution using its thread and run IDs:

```bash
msgflux runs inspect THREAD_ID RUN_ID
msgflux runs inspect THREAD_ID RUN_ID --json
```

This command uses the existing run inspection endpoint. The human output shows
receipt/checkpoint status, approval and background task counts, and recovery
reasons, including required worker quiescence. JSON returns the full
`RunInspection` record. It contains recovery evidence rather than the final
answer. These inspection commands never start a server. Run inspection requires
an active, authenticated daemon and reports errors on stderr with exit code `1`.
Use `--runtime-dir PATH` to select another local service.

## List Active Runs

```bash
msgflux runs ls --active
msgflux runs ls --active --json
msgflux runs ls THREAD_ID --active
msgflux runs ls --active --agent coding
msgflux runs ls THREAD_ID --active --agent coding --json
```

The table shows `THREAD_ID`, `AGENT`, `RUN_ID`, `REQUEST_ID`, and `STATUS`. JSON returns
`{"runs": [...]}` containing admission receipts. A thread ID filters the list;
an unknown thread is an error. `--agent NAME` filters by a registered agent ID;
an unknown agent is an error. Thread and agent filters can be combined. JSON
retains the existing receipt records. Empty results are successful.

The list is a snapshot of root workers owned by the connected service, including
scheduled workers and workers settling an interruption request. It does not
load Agent sessions or infer activity from checkpoint status. A persisted
`running` receipt after a server crash is recovery evidence, not a live worker,
and is excluded. Paused or completed executions and separate background tasks
are also excluded. Activity can change immediately after the response.

The authenticated `GET /v1/runs/active` endpoint exposes the same information.
Both the embedded service and HTTP client provide `await service.active_runs()`
and `await client.active_runs()`, respectively:

```python
for receipt in await client.active_runs():
    print(receipt.thread_id, receipt.run_id, receipt.status)
```

This example inspects live root execution without opening checkpoints or loading
idle sessions. Subagent activity belongs to its parent execution and task
observations; this list is not a subagent task inventory.

## Interrupt A Thread's Execution

```bash
msgflux threads interrupt THREAD_ID
msgflux threads stop THREAD_ID --json
```

`threads stop` is an alias for `threads interrupt`. Both select the thread's
locally active root run and request cooperative interruption through the existing
service endpoint. They preserve the server and other threads. Selection captures
the run ID; if it finishes before the request arrives, the command does not
interrupt a newer run. Already interrupted or idle threads are successful no-ops;
unknown threads are errors. Neither command starts an absent service.

JSON contains `thread_id`, `run_id`, and `status`: `interrupt_requested` means the
signal was sent, `no_active_run` means there was no locally owned worker, and
`not_active` means the selected worker finished before accepting interruption.
An interruption request is not confirmation that all work has stopped.
Cooperative code must observe the signal; use `runs inspect` or SDK `watch()` to
observe settlement. `interrupted` remains terminal, as described below.

Use `service stop` to shut down the whole server. Use `--runtime-dir PATH` with
these commands to select another local service. `--stop-timeout` applies only to
server shutdown, not to a thread interruption request.

## Resume A Paused Run

```bash
msgflux runs resume THREAD_ID RUN_ID
msgflux runs resume THREAD_ID RUN_ID --json
```

This requests recovery of the existing checkpoint and returns its admission
receipt immediately. It does not submit a new prompt, wait for completion, or
start an absent daemon. The plain output shows the run ID and receipt status;
JSON returns `AdmissionReceipt`. Observe progress using the SDK's `watch()`.

Pending approvals still require a decision before execution can proceed.
`completed` and `interrupted` runs are terminal under the current service
contract: the command returns their existing receipts without restarting them.
An accepted receipt confirms admission, not successful completion. Recovery that
requires old-worker quiescence must be handled by the trusted host; this command
cannot assert quiescence or bypass recovery checks.

## Gracefully Stop The Daemon

```bash
msgflux service stop
msgflux service down
msgflux service stop --stop-timeout 45 --json
```

`down` is an alias for `stop`. Both verify the current daemon's authenticated
identity, request cooperative shutdown, and wait for its lifetime lock to be
released before reporting `stopped`. Active runs receive interruption requests;
workers and factory-owned resources are joined and closed by the existing
service shutdown path. No replacement daemon starts. Stopping an absent service
is successful and creates no runtime files; stopping an already stopped service
is also successful.

The default deadline is 30 seconds. `--stop-timeout` must be finite and positive.
A timeout exits with code `1`; the server may still be closing its resources.
An unverified live instance is refused. The command never force-kills a process
using a recorded PID. JSON returns a `LocalServiceStatus` with `status="stopped"`
after successful shutdown. Errors go to stderr. Use `--runtime-dir PATH` to
select the service to stop.

The SDK exposes the same operation:

```python
from msgflux.runtime.service.local import stop_local_service

status = await stop_local_service(stop_timeout=45)
print(status.status)  # stopped
```

This example waits for graceful shutdown without constructing a new client or
starting a replacement service. Concurrent connectors wait on the shared startup
lock; after it is released, an explicit connector can start the service again.

## Gracefully Restart The Daemon

`restart_local_service()` asks the authenticated current instance to shut down,
waits for its lifetime lock to be released, and starts a replacement under the
same startup lock used by concurrent connectors. With no arguments it uses the
factory or manifest and launch directory recorded by the current daemon:

```python
import asyncio

from msgflux.runtime.service.local import restart_local_service


async def main():
    client = await restart_local_service(restart_timeout=45)
    try:
        print("New runtime:", (await client.health()).instance_id)
    finally:
        await client.aclose()


asyncio.run(main())
```

This example returns a client connected to the replacement instance. The
restart timeout bounds the complete stop and startup wait. If graceful shutdown
is accepted but the old daemon does not release its lock before the deadline,
the call raises `TimeoutError`; the old daemon continues closing its resources.
No replacement starts while it holds the lifetime lock. A connection attempt
may report `ServiceRecoveryRequiredError` while the old HTTP server is no longer
healthy; retry after shutdown completes. The timeout does not force-kill a worker
or signal a process.

Restart uses the existing service shutdown behavior: active runs receive a
cooperative interruption request, and shutdown joins their workers before closing
factory resources. Their receipts and checkpoints retain the interruption state.
Restart does not automatically replay prompts or resume paused approvals.

Pass `factory` and `cwd` to select new trusted launch configuration:

```python
client = await restart_local_service(
    "my_backend:create_service", runtime_dir="~/.msgflux/runtime",
    cwd="/absolute/project/path", restart_timeout=45,
)
```

An explicit factory must match the recorded factory while its daemon is alive.
If metadata is unavailable, supply a factory explicitly; a dead daemon's PID is
never used to infer or signal an owner. A healthy older daemon that does not
provide authenticated graceful shutdown raises `ServiceRecoveryRequiredError`;
stop or upgrade it through its existing operator-managed mechanism before
restarting.

The CLI keeps the existing foreground invocation unchanged. `serve` is the
default action; `restart` derives the factory and launch directory from private
runtime metadata and prints the new instance identity and loopback URL:

```bash
uv run --extra service msgflux service restart \
  --runtime-dir ~/.msgflux/runtime --restart-timeout 45
```

The returned object is an ordinary `AgentServiceClient`. Use `watch()` to get an
atomic snapshot and future events. Reconnecting after a server restart requires
calling `connect_local_service()` with the same config or factory again so the
client receives the new URL/token;
old connections are not silently redirected or retried.

## Run In The Foreground

For an explicitly managed persistent process:

```bash
uv run --extra service msgflux service \
  --factory my_backend:create_service \
  --cwd /absolute/project/path
```

The explicit spelling `msgflux service serve --factory ...` is equivalent.

The equivalent Python entry point is
`python -m msgflux.runtime.service.local.cli`. A foreground server and clients
using on-demand discovery share the same lifetime lock. A competing server fails
without overwriting the active instance's metadata.

Ctrl+C or an explicit operating-system stop requests shutdown. Active SSE responses
are given a bounded graceful shutdown interval; the service then joins its owned
foreground work and closes factory resources. Agent interruption remains cooperative,
so non-cooperative application code can delay resource cleanup. There is no idle
shutdown timer and no shutdown request when a client closes.

## Files And Identity

The default directory is `~/.msgflux/runtime/`. Override it with `runtime_dir` in
Python or `--runtime-dir` on the command line.

```text
~/.msgflux/runtime/
├── daemon.json
├── daemon.lock
├── startup.lock
└── daemon.log
```

| File | Purpose |
| --- | --- |
| `daemon.json` | Versioned instance ID, factory or manifest path/digest, cwd, loopback URL, informational PID, and private connection token |
| `daemon.lock` | Kernel lock held for the daemon's process lifetime |
| `startup.lock` | Coordinates frontend discovery, spawn and readiness |
| `daemon.log` | Startup/server diagnostics for on-demand launches |

The selected directory is owned by the current user and secured to mode 0700.
Metadata and lock/log files use mode 0600; metadata is published atomically and
symlink metadata/lock paths are rejected. The token is local runtime authentication,
not a provider credential. Do not share the metadata file. Metadata representations
redact the token.

The backend binds only to `127.0.0.1` on an available port. `GET /v1/health` requires
the bearer token and returns the runtime instance ID and protocol version without
constructing an Agent. A PID from a discovery file is never used to signal or kill
a process. Stale metadata may be replaced when no owner holds the lifetime lock;
failed startup cleanup operates only on the child handle created by that launcher.

Factory-owned conversation stores stay separate from discovery files. The example
adds `runtime/service.sqlite3` and `threads/<thread_id>/checkpoints.sqlite3`; the
manifest runner owns the admission journal, while the factories choose conversation
storage and model credentials.

## Failure And Recovery

A launcher may exit immediately after connecting; the daemon's lifecycle is
independent of that frontend's event loop. If startup fails or times out, inspect
`daemon.log`. A canceled/failed startup releases the launcher's own child and startup
lock; it does not terminate a process identified by stale metadata.

A process crash releases its kernel lock. The next connection can start a new
instance while preserving application-owned persistent stores. This does not
implicitly retry started Agent inputs or assert quiescence of surviving external
commands. Existing checkpoint, workspace, command-receipt and approval recovery
checks remain authoritative. A trusted factory/host can perform explicit recovery
before serving clients; network clients cannot manufacture `worker_stopped` evidence.
