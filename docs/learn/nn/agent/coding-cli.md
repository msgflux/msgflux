# Coding Command Line

Install `msgflux[service]` to use the local coding backend. The `msgflux` command
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
whose workspace is the current directory. Use `--cwd` to select another project.
The command prints only the final answer; commentary, reasoning and tools remain
in the service's observation stream and history.

```bash
msgflux service
msgflux service restart
msgflux -p "Investigate the failing test" --agent explorer
```

`service` runs in the foreground; `service restart` replaces the discovered local
daemon. `--agent` selects an entry from the backend config. Without it, the command
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
text deltas show their text, while structured data uses JSON. The command still
prints the recorded final answer at the end. Tools, commentary (when emitted),
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

Successful tasks exit with code 0. A failed, paused or interrupted run exits with
code 1. Plain-text errors go to stderr and include available execution IDs;
JSON mode returns those IDs, status and error in its final stdout record. Parsing
errors use the normal CLI usage message and exit code 2.

Approvals do not trigger a terminal question in print mode. An approval pause
returns its reason and identifiers so a reviewer can decide and resume through
service APIs. Completed background results remain repeatably accessible through
the TaskTool bucket.

The frontend uses `watch()` internally and reconnects after transient transport
loss. It does not resubmit an admitted prompt. The final answer is loaded from the
checkpoint for the specific run, so a later turn cannot replace that answer. If
the daemon has lost ownership of an execution, the command reports the service's
recovery error instead of attempting unsafe replay.

Ctrl+C explicitly requests interruption of the selected execution through the
service and exits with code 130. The CLI allows up to five seconds to recover an
in-flight admission receipt, send the request and confirm settlement. It never
submits the prompt again. The final record reports `interrupted` when confirmed,
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

```python
result = await client.run_result(thread_id, run_id)
if result.status == "completed":
    print(result.output)
```

The HTTP equivalent is `GET /v1/threads/{thread_id}/runs/{run_id}/result`.
Only a completed run exposes final assistant text. Other statuses have no output;
failed and paused receipts preserve their concrete cause. Missing final
checkpoints or non-text answers are reported as errors, rather than returning
text from another run. This requires checkpoint-backed agents; the coding
factory configures them automatically.
