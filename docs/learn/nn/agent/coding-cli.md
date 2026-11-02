# Coding Command Line

Install `msgflux[coding,service]` to use the local coding backend,
including workspace navigation tools used by read-only agents. The `msgflux` command
manages the service and submits noninteractive tasks. The TUI is a separate frontend.

## Initialize And Run

```bash
msgflux accounts login provider --account-name default
msgflux init --model provider/model-id
msgflux -p "Review the current changes"
```

The login command chooses the provider's registered authentication method: API
keys use a hidden prompt and OAuth adapters start their login flow. The coding
backend uses the managed `default` account unless selected otherwise; credentials
are not copied into the service config. See [Accounts](accounts.md).

Initialization creates `~/.msgflux/config.toml` once and preserves an existing
file. Choose a model available through your configured credentials. A prompt
connects to the configured daemon, starting it when absent, and creates a thread
whose workspace is the current directory. Use `--cwd` to select another project;
relative paths are resolved from the invocation directory. The command prints
the final answer on stdout. As soon as the service confirms
admission, plain mode prints `thread_id`, `agent_id`, `request_id`, and `run_id`
on stderr and flushes them, including when the thread ID was generated. Save
these IDs to continue or inspect the execution later. Commentary, reasoning and
tools remain in the service's observation stream and history unless `--stream`
is selected. JSON modes keep identity data in their stdout records.

```bash
msgflux service
msgflux service status
msgflux service restart
msgflux -p "Investigate the failing test" --agent explorer
```

`service` runs in the foreground; `service status` shows the verified address
without starting the daemon; `service restart` replaces the discovered local
daemon. Add `--json` to `service status` for machine-readable diagnostics.
`--agent` selects an entry from the backend config. Without it, the command
uses `defaults.agent`. Declare the Explorer preset before selecting it.
The former `msgflux-service` console entry has been replaced by `msgflux service`.

## Read The Task From A File

```bash
msgflux --prompt-file task.md
msgflux --prompt-file - < task.md
git diff | msgflux --prompt-file - --json
```

`--prompt-file` reads the task from a UTF-8 file, or from standard input when
the path is `-`. It replaces `-p/--prompt` and is required in its place: exactly
one of the two must be supplied, and passing both is a usage error. The file is
read before the service is discovered, so an unreadable path, invalid UTF-8, or
blank-only input exits with the normal usage message and code `2` without
starting or contacting a daemon and without writing any configuration file.
Standard input is decoded strictly, so invalid bytes are rejected instead of
being replaced. Trailing newlines are kept and sent as part of the task.

`-p` still takes its text literally. In particular, `msgflux -p -` sends the
single character `-` rather than reading stdin; only `--prompt-file -` reads it.

## JSON And Continuation

```bash
msgflux -p "Summarize the project" --json
msgflux -p "Investigate the first finding" --thread THREAD_ID --json
```

`--json` emits one final JSON object on stdout:

```json
{
  "status": "completed",
  "thread_id": "...",
  "agent_id": "coding",
  "model": "provider/model-id",
  "request_id": "...",
  "run_id": "...",
  "output": "The final answer",
  "error": null,
  "version": 1
}
```

The IDs identify the durable conversation, request and execution. Continuing a
thread preserves its agent and workspace; an explicit conflicting `--agent` or
`--cwd` is rejected. `--config` selects another backend config, and `--runtime-dir`
selects its discovery/state directory. Configuration identity still controls
whether an existing daemon can be reused.

Print threads store `interactive = false`, which makes the coding factory omit
SendUserMessageTool and give the model instructions to work without questions or
progress messages. Other service threads remain interactive by default. The mode
is part of the thread binding and survives restart; use a new thread when switching
between interactive and print modes. Application factories receive this flag on
ServiceThread and are responsible for honoring it.

## Send Without Waiting

```bash
msgflux -p "Review the current changes" --detach
msgflux -p "Review the current changes" --detach --json
```

`--detach` waits for the service to admit the request, returns its execution IDs
and current receipt status, and closes the client. The service continues running
the agent independently. It does not open an event watcher or wait for a final
answer. No additional login is needed when the selected account is already
configured.

In text mode the IDs are flushed to stderr and stdout contains `status=accepted`,
`status=running`, or another receipt status. JSON mode prints one `PromptResult`
object with the IDs and status; `output` remains null because this command only
retrieves the admission receipt. Exit code 0 means the request was admitted with
status `accepted`, `running`, or `completed`. A later model or tool failure can
still change the recorded outcome. Setup errors and receipts already reporting a
failed, paused, or interrupted state exit with code 1.

Use `msgflux threads result THREAD_ID --wait` to retrieve the final answer,
`msgflux runs inspect THREAD_ID RUN_ID` to inspect the execution, or
`msgflux threads interrupt THREAD_ID` to request interruption. A subsequent
`msgflux -p "Follow up" --thread THREAD_ID` sends a new turn when the current
execution has settled. The `threads watch THREAD_ID` command observes an existing thread without
sending a turn.

`--detach` and `--stream` are mutually exclusive. If the HTTP admission response
is lost, the command looks up the same `request_id`; it never sends the prompt
again. If admission cannot be confirmed, the error includes the known IDs so
you can inspect them before deciding what to do next.

## Select And Inspect The Model

The coding agent's model comes from `agents.<agent-id>.model` in the backend
TOML, falling back to `defaults.model`. Account selection supplies credentials;
it does not choose the model.

Inspect the defaults loaded by the running service:

```bash
msgflux agents inspect coding
msgflux agents inspect builder --json
```

The first command displays the registered agent's default model and account.
The JSON command exposes the same registration metadata for automation; it
does not claim to inspect a thread-specific override.

Choose a different model when creating a thread:

```bash
msgflux -p "Review this change" --agent builder   --model provider/model-id --account-name work --detach --json
```

`--model` uses the same `provider/model-id` syntax as the TOML model setting.
The service binds the selected model to the new thread and returns it in the
CLI result's `model` field. Human-readable admission identities on stderr
include `model=...`; `msgflux threads ls --json` also exposes the binding.

New coding threads preserve this selection across daemon restarts and changes
to registration defaults. Continue without repeating the override:

```bash
msgflux -p "Now check the edge cases" --thread THREAD_ID
```

An explicit model conflicting with the existing binding is rejected; create a
new thread to select another model. A legacy thread without a saved model
binding reports `model: null`; its historical model is not inferred from the
current configuration. Custom factories must explicitly declare support for
model overrides before accepting one.

The override selects the top-level agent's model. Configured tools, permissions
and independently configured subagents keep their existing settings. The
selected account must supply credentials for the effective model's provider.

## Select An Account

```bash
msgflux accounts login openai-codex --account-name work
msgflux -p "Review the current changes" --account-name work --json
```

`--account-name` selects the credential bundle for this execution, including its
subagents and background tasks. Omit it to use `[defaults].account` (`default`
when unspecified). A later prompt on the same thread can select a different
account. Recovery retains the original recorded account; changing the default
does not change an admitted run. Missing providers or invalid credentials fail
without selecting a different account or reading global provider credentials.

Accounts normally live under `~/.msgflux/accounts`. A trusted host can set
`[defaults].accounts_dir` to another directory; use the same path with
`msgflux accounts login --accounts-dir PATH`. Credentials are resolved for each
request, rather than stored on cached models or shared HTTP clients.

## Live Events

```bash
msgflux -p "Review the current changes" --stream
msgflux -p "Review the current changes" --stream --json
```

`--stream` observes the execution through the service's `watch()` API and prints
records as they arrive. Plain mode labels each event with its type and source;
text deltas show their text, while structured data uses indented field names.
Tool arguments, errors and multiline results remain readable without compact
JSON objects. For example, a Bash completion can show:

```text
[tool.end agent:main/tool:bash]
  tool_name: bash
  arguments:
    command: pwd
  result:
    stdout: /home/user/project
    returncode: 0
  error: null
```

The command still prints the recorded final answer at the end. Tools, commentary (when emitted),
reasoning, task updates and subagent events retain their source information.
Noninteractive coding agents are still instructed to work without progress
commentary or user questions; streaming does not change the agent's behavior.

Combining `--stream --json` emits **NDJSON**: one JSON object per line, flushed
immediately. Each envelope has `version = 1` and one of these types:

| `type` | Payload | Meaning |
| --- | --- | --- |
| `snapshot` | `snapshot` | Thread state at connection or reconnection. |
| `event` | `event` | An unmodified service EventRecord, including `type`, `timestamp`, `data`, `run_id` and `source_path`. |
| `result` | `result` | The final result described above, including status, IDs, output and error. |

For example, these simplified records illustrate the envelope format:

```json
{"type":"snapshot","snapshot":{"thread_id":"thd_example","namespace":"coding","messages":[],"active_runs":[],"running_tools":[],"background_tasks":[],"approvals":[],"version":1},"version":1}
{"type":"event","event":{"type":"tool.start","timestamp":"2026-10-12T12:00:00Z","data":{"name":"read"},"run_id":"run_example","source_path":["coding"],"version":1},"version":1}
{"type":"result","result":{"status":"completed","thread_id":"thd_example","agent_id":"coding","request_id":"req_example","run_id":"run_example","output":"The final answer","error":null,"version":1},"version":1}
```

A normally terminating command emits exactly one `result` envelope, including
when setup fails, the run fails or pauses, or Ctrl+C interrupts observation. A
`run.end` event is an execution event, not the final CLI result; use the `result`
envelope to determine the exit outcome. Embedded newlines in JSON strings are
escaped, so consumers can parse each stdout line independently. `--json` without
`--stream` continues to emit a single final object without an envelope.

Reconnection emits a fresh snapshot and continues with live events. The service
does not replay deltas missed during disconnection: this stream is not a complete
event log. A snapshot can contain earlier conversation history. Consumers should
reconcile snapshots rather than treating them as additional model messages. The
final result remains authoritative even if intermediate events were missed.

## Errors, Approvals And Disconnection

To observe an existing thread without sending another prompt, use:

```bash
msgflux threads watch THREAD_ID
msgflux threads watch THREAD_ID --json
```

This observer remains open between runs. Ctrl+C disconnects it without
interrupting execution. JSON output includes the initial snapshot and new events,
with no final `result` envelope. See [Watch An Existing Thread](service-local.md#watch-an-existing-thread)
for discovery, reconnection, and output details.

To send an instruction to a thread, whether it is currently running or idle:

```bash
msgflux threads steer THREAD_ID -m "Focus on the failing integration test"
```

On a live thread this queues the message into the current run's inbox without
starting another prompt. On an idle thread it admits the message as a new prompt
and starts a run. The JSON reply distinguishes both cases by `status`
(`queued` or `started`). See [Steer A Thread](service-local.md#steer-a-thread)
for delivery timing, guards and retry advice.

Successful tasks exit with code 0. A failed, paused or interrupted run exits with
code 1. Plain-text errors go to stderr and include available execution IDs;
JSON mode returns those IDs, status and error in its final stdout record. Parsing
errors use the normal CLI usage message and exit code 2.

Exception messages are preserved. If a transport or setup exception has an empty
or whitespace-only message, the CLI uses its class name (for example,
`ReadTimeout`) so the diagnostic remains useful in text, JSON and NDJSON modes.
This does not imply that the run stopped; retain the reported IDs to inspect its
state after a connection failure.

Approvals do not trigger a terminal question in print mode. An approval pause
returns its reason and identifiers so a reviewer can decide and resume through
service APIs. Completed background results remain repeatably accessible through
the TaskTool bucket.

The frontend uses `watch()` internally and reconnects after transient transport
loss. It does not resubmit an admitted prompt. The final answer is loaded from the
checkpoint for the specific run, so a later turn cannot replace that answer. If
the daemon has lost ownership of an execution, the command reports the service's
recovery error instead of attempting unsafe replay.

If the admission response is lost, the frontend looks up its already generated
`request_id` and recovers the `run_id` from the existing receipt; it does not
send the prompt again. If that receipt is absent or cannot be confirmed, the
command reports failure with the IDs it knows and does not invent a run or
reexecute the prompt.

Ctrl+C explicitly requests interruption of the selected execution through the
service and exits with code 130. Within a five-second timeout, the CLI recovers
an in-flight admission receipt using the same `request_id`, sends the interruption
request and confirms settlement without submitting the prompt again. The final
record reports `interrupted` when confirmed,
`interrupt_requested` when the server accepted the signal but settlement has not
yet been confirmed, or `detached` when the interruption could not be confirmed.
Known IDs and a concrete reason remain available for inspection. If the run
completed during the interruption request, its completed result is preserved,
although the CLI still exits with code 130.

```bash
msgflux -p "Run a long investigation" --stream --json --detach-on-interrupt
```

With `--detach-on-interrupt`, Ctrl+C only disconnects observation. The final
record reports `detached`, and an admitted execution continues in the backend.
This also exits with code 130. The option works with both final-only and streaming
output, with or without JSON. To interrupt a detached execution explicitly, use
`await client.interrupt(thread_id, run_id)`.

Interruption is cooperative and targets the selected run. It does not undo
filesystem changes or other effects that have already occurred. Independently
running background tasks should be inspected and interrupted separately.

## Retrieve A Recorded Answer

```bash
msgflux threads result THREAD_ID
msgflux threads result THREAD_ID --wait
msgflux threads result THREAD_ID --wait --wait-timeout 60
msgflux runs result THREAD_ID RUN_ID --json
```

The thread command selects the most recently admitted execution. The run command
selects one execution explicitly. Both accept `--wait`, `--wait-timeout` and
`--json`; see
[Retrieve A Thread Or Run Result](service-local.md#retrieve-a-thread-or-run-result)
for waiting, output and exit-code behavior.

Without `--wait`, the query returns immediately. A running execution has
`output: null`; it never substitutes an older answer. Text mode prints only the
completed answer to stdout. JSON returns `RunResult`, containing `thread_id`,
`run_id`, `status`, `output`, and `error`.

The SDK offers the same queries:

```python
latest = await client.thread_result(thread_id)
recorded = await client.run_result(thread_id, run_id)
if recorded.status == "completed":
    print(recorded.output)
```

These calls do not wait. Their HTTP equivalents are
`GET /v1/threads/{thread_id}/result` and
`GET /v1/threads/{thread_id}/runs/{run_id}/result`.
Only final assistant text is exposed. A completed run without final assistant
text has `output: null`; tool payloads are not returned. Other statuses also
have no output, and failed or paused receipts preserve their concrete cause.
The selected run never falls back to text from another run. This requires
checkpoint-backed agents; the coding factory configures them automatically.
A missing final checkpoint is still reported as a recovery error.

## Check Versions And Update

```bash
msgflux --version
msgflux version --check
msgflux version --check --json
msgflux update
```

`--version` (or `msgflux version`) prints the installed version without accessing
networks, configuration, or the service. `version --check` makes a fresh request
to the selected package index on every invocation, without a local cache, and
reports the installed and
latest eligible versions. A failed request exits with an error; it does not claim
the installation is up to date. JSON output includes `installed`, `latest`, and
`update_available`.

Stable installations check stable releases. Prerelease installations also check
prereleases; add `--pre` to a check or update to include prereleases from a stable
installation. Yanked releases, development releases, and distributions incompatible
with the running Python version are excluded. Updates never automatically
downgrade the package.

`msgflux update` supports packages installed through `uv tool` from a package
index. It selects the latest eligible version from that index and invokes
`uv tool install --force --upgrade msgflux[EXTRAS]==VERSION`, preserving the
tool's installed extras, additional index-based requirements, and Python
interpreter. This also updates installations pinned to a previous version.
Git, URL, editable, source-checkout, and other environment installations are
refused: update those using their original installation method. A single default
Simple index and recorded `find-links` settings are supported. Multiple or named
indexes, custom index authentication policies, other installer settings, or
additional exported tool executables require a manual update. No update occurs
when the installed version is already current.
Configuration files and managed accounts are preserved.

!!! warning "Updating restarts the service"

    When a local AgentService is running, update stops it **before replacing the
    package**, then starts it with the updated executable and its recorded
    configuration or factory and launch directory. Restart may interrupt active
    generations. **Automatic resumption of those generations is not guaranteed.**
    The command refuses an update when it detects active generations, unless
    `--force` is supplied. An unreachable service also blocks the update.

```bash
msgflux update --force
msgflux update --runtime-dir /path/to/runtime --timeout 60
```

`--force` permits interruption of active generations; it does not bypass service
identity checks. `--runtime-dir` selects the local runtime to inspect and restart;
`--timeout` bounds each service stop and restart operation in seconds. A stopped
service stays stopped after installation. Avoid submitting new work during the
update: inspection and shutdown are separate operations. If you run additional
service processes or runtime directories using the same installation, stop them
before updating; this command manages only the selected runtime.

If installation fails after stopping the service, the command attempts to start
it again and exits with an error. Package installation is not transactional and
this attempt does not guarantee rollback. If installation succeeds but restart
fails, the error explicitly reports that the package was updated. Correct the
reported startup problem and run `msgflux service restart` with the same
`--runtime-dir` (and configuration or factory, if needed).

### Use A Custom Package Index

```bash
msgflux version --check --index-url https://packages.example.com/team/simple/
msgflux update --index-url https://packages.example.com/team/simple/
```

Pass the **base URL** of an HTTP(S) Simple repository, without the package name.
The check requests `BASE_URL/msgflux/` and supports both JSON (PEP 691) and HTML
(PEP 503) responses, with the same Python-version and yanked-release filtering.
The update passes that same base URL to uv as `--default-index`, so release
selection and installation use the same default index. The selected release and
its dependencies must be available through the installer's package sources.
Requests are fresh even when using a mirror; the remote mirror controls how
quickly it synchronizes its own content.

An explicit `--index-url` overrides the default index recorded in the current
uv tool receipt. Without the flag, both commands reuse that recorded default
index when available and otherwise use `https://pypi.org/simple/`. uv records the
selected index on installation, so subsequent updates can reuse it. Local
`--version` and `version` do not read index settings or make network requests;
`version --index-url` requires `--check`.

This discovery path does not resolve uv configuration files, index environment
variables, keyring providers, or named-index authentication. Use `--index-url`
explicitly when the repository is configured only through those mechanisms.
The installer still uses uv's normal configuration for additional package
sources; the explicit default index passed by msgflux takes precedence over
uv's configured default.

## Track Child Threads

Create independent CLI threads linked to an existing parent on the same service:

```bash
msgflux -p "Inspect the tests" --agent explorer --parent PARENT_THREAD_ID --detach --json
msgflux -p "Implement the fix" --agent builder --parent PARENT_THREAD_ID --detach --json
msgflux threads children PARENT_THREAD_ID --json
```

A profile can be launched repeatedly: two commands with `--agent builder`
and the same `--parent` create two children with distinct thread and run IDs.
The children command lists both. Address an individual child using its returned
IDs with `runs result` or `threads steer`. Use `--thread` only when continuing
an existing child.

The service persists the parent relationship and rejects a missing parent,
self-parenting or a conflicting relationship on reopen. Omitting `--parent`
when continuing a child preserves its existing binding. Listing reads only
bindings and admission metadata; it does not load sessions, conversation
history or final answers. Receipt status describes persisted execution state,
not proof that a worker is alive.

Wait for any run in the selected child page to settle:

```bash
msgflux threads children PARENT_THREAD_ID --wait --wait-timeout 30 --json
msgflux threads children PARENT_THREAD_ID --wait --exclude-run HANDLED_RUN_ID --json
```

The first command returns a compact child identity and status, including failed,
paused or interrupted outcomes. Exit code 0 means observation succeeded; inspect
the child's status to determine whether its work succeeded. Already settled runs are eligible, so a fast
child cannot be missed between launch and observation. Repeat `--exclude-run`
to skip outcomes already handled. An empty eligible batch returns `empty`.
Retrieve an answer explicitly with `msgflux runs result THREAD_ID RUN_ID`.

Waiting pins the run identities visible in the initial page. Later children
and later turns do not silently join that batch. The deadline covers HTTP
requests and polling; on expiry the command returns `timeout`, exits with code
1 and leaves execution running. It does not resume paused work or recover
uncertain workers.

Pages contain at most 100 children by default. Use `--limit N` (maximum 1000)
and `--after-child CURSOR` with the returned `next_cursor` to observe another
page. The cursor orders child insertion and survives service restart.

This relationship records delegation metadata. Delivery through
`send_parent_message`, automatic parent wakeup and inherited delegation
credentials are separate capabilities; this step does not enable them.
