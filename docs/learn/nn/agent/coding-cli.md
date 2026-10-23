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

To send an additional instruction to the active run through its inbox:

```bash
msgflux threads steer THREAD_ID -m "Focus on the failing integration test"
```

This queues a message for the current run without creating another prompt or
waiting for completion. See [Steer An Active Thread](service-local.md#steer-an-active-thread)
for delivery timing and target selection.

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
msgflux runs result THREAD_ID RUN_ID --json
```

The thread command selects the most recently admitted execution. The run command
selects one execution explicitly. Both accept `--wait` and `--json`; see
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
