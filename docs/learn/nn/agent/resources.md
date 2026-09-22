# Durable Runtime Resources

`RuntimeResources` provides an explicit local application layout. Its minimum
directories keep immutable tool results separate from per-thread checkpoints:

```text
~/.msgflux/
  tool-results/
    res_<uuid>/
      content
      metadata.json
  threads/
    thd_<id>/
      checkpoint.sqlite
```

The root is configurable. Constructing an Agent or `RuntimeResources` does not
automatically create this layout or enable disk persistence. This is host-side
infrastructure, not a filesystem mount or permission grant to a model.

## Configure the Local Layout

```python
from msgflux.runtime import RuntimeResources

resources = RuntimeResources("~/.msgflux").initialize()
checkpoints = resources.checkpoint_store("thd_example")
results = resources.tool_result_store(max_result_bytes=64 * 1024 * 1024)

try:
    checkpoints.save_state(
        "main", "thd_example", "run_example", {"status": "running"},
    )
finally:
    checkpoints.close()
```

This example creates the directories explicitly and opens the existing
`SQLiteCheckpointStore` adapter at `threads/thd_example/checkpoint.sqlite`.
The caller owns the connection lifecycle. `resources.plans_path` exposes the
host directory for application-managed plans; no plan tool, automatic context
loading or concurrent-edit protocol is installed.

Only `threads` and `tool-results` are created by default. Add optional top-level
directories when initializing the layout:

```python
resources.initialize(extra_dirs=("plans", "skills"))
```

This explicitly creates the two additional directories. Initialization is
idempotent and does not remove existing directories. All extra names are checked
before creating anything: use single components such as `skills`, not absolute
paths, `..`, or nested paths. `plans_path` remains a convenience for locating an
optional plans directory; accessing the property does not create it. Creating
`skills` does not automatically configure or load Agent skills.

Pass the checkpoint adapter to your Agent's `checkpoint_store` configuration and
use the same thread ID in its execution scope. The factory does not inject scope
or restrict SQLite to a single thread. Nested `AgentTool` calls normally inherit
the parent's thread and checkpoint store, with separate namespaces/run IDs;
they do not require a separate SQLite file. An explicitly configured child store
overrides inheritance. Durable background scheduling additionally needs a durable
task store; this layout alone does not restore in-memory task records.

## Store Results Incrementally

```python
from msgflux.runtime import RuntimeResources

results = RuntimeResources("./runtime-data").tool_result_store()
reference = results.put(
    (chunk for chunk in (b"first part\n", b"second part\n")),
    media_type="text/plain; charset=utf-8",
)
descriptor = reference.to_dict()
```

`put()` consumes an iterable of byte chunks without joining the entire content.
It returns a frozen `msgspec.Struct` containing `result_id`, `size_bytes`,
`sha256` and `media_type`. `reference.uri` is a logical address such as
`tool-result://res_...`, not a host path or an authorization token. UUID4 follows
the existing runtime ID convention and works on Python 3.11 without a new
dependency. IDs avoid collisions; identical content is **not** deduplicated.

The local store stages the complete result, syncs its files and directory, then
publishes it with an atomic directory rename and syncs the store directory. It
rejects symlink/special result entries and never overwrites a published result.
The implementation requires POSIX filesystem APIs. Filesystem/device durability
guarantees still apply; this is not an OS sandbox against hostile local writers.

`max_result_bytes` defaults to 64 MiB **per stored result** and is host-configurable.
`max_store_bytes` additionally defaults to 1 GiB for the entire local store.
It counts content and metadata bytes, including interrupted staging writes,
not filesystem blocks, inode overhead or shell-capture temporary files.
The per-result ceiling is not a model-token limit or UI preview size. Exceeding
that ceiling raises `ToolResultTooLargeError`; producer failures propagate without returning
a partial reference. A process crash may leave a `.pending-*` directory, but it
is not a published result. Storage failures are not retried automatically.

## Read Ranges and Transmit Chunks

```python
preview = results.read(reference, offset=0, limit=4096)

for chunk in results.iter_bytes(reference, offset=4096, chunk_size=65536):
    send_bytes(chunk)  # application callback; consume before requesting the next

results.verify(reference)
```

`read()` materializes at most `limit` bytes (64 KiB by default). `iter_bytes()`
reads bounded chunks, with an optional total `limit`, suitable for incremental
transmission. Close an abandoned iterator to release its file descriptors.
Offsets and limits are **bytes**, so UTF-8 ranges can split a character; use an
incremental decoder when transmitting text. These synchronous APIs perform local
I/O; async applications should use worker execution with bounded handoff, not
block the event loop or materialize all chunks in a list.

Range reads validate descriptor identity and file size. `verify()` additionally
streams the complete content to check SHA256; range reads do not hash the entire
file. Missing content raises `FileNotFoundError`, while malformed metadata,
size mismatches and failed checksum verification raise `ToolResultIntegrityError`.
No missing result is silently replaced with empty text.

## Persist References and Relocate Storage

```python
import msgspec
from msgflux.runtime import ToolResultRef

state = {"tool_call_id": "call_example", "result": reference.to_dict()}
# Store state using the existing checkpoint adapter.

restored = msgspec.convert(state["result"], type=ToolResultRef)
results.verify(restored)
```

The plain descriptor is compatible with the current JSON checkpoint adapters.
The checkpoint owns the association with the tool call; the result store does not
maintain a duplicate per-thread call index. Save the checkpoint only after `put()`
returns successfully. File publication and SQLite commits are not one transaction:
a crash between them can leave an unreferenced complete result, not a checkpoint
pointing to an unfinished published write.

For a quiescent application, copy the closed checkpoint databases and their
referenced result directories to the new root. Reopen `RuntimeResources` there;
reference IDs and checksums remain unchanged. Copying a live SQLite file without
coordinating its WAL is not a supported backup procedure. Automated export/import,
reference-closure discovery and shared-plan permissions are
not implemented by this layout. Explicit offline garbage collection is described
below. Never delete results merely because one thread
was removed: other trajectories may still reference them.

## Opt-in Tool Output Offload

### Aggregate quota and offline maintenance

```python
results = resources.tool_result_store(
    max_result_bytes=64 * 1024 * 1024,
    max_store_bytes=1024 * 1024 * 1024,
)
usage = results.usage()  # size_bytes, results, pending
```

This bounds stored content/metadata across successful results and abandoned
staging directories. `put()` raises `ToolResultQuotaError` before crossing the
budget and removes its own incomplete write. It never evicts previous results.
Writers using this API serialize through a local POSIX advisory directory lock,
including independent processes. Configure the same quota for all writers;
unmediated filesystem writes and remote/distributed filesystems are outside this
guarantee. A slow producer holds the lock until its write completes or fails.

```python
# Stop ALL checkpoint/result writers and readers sharing this store first.
# Supply every live ToolResultRef from ALL threads, forks, historical records,
# exported checkpoints and other consumers, not only each latest state.
retained = application_reference_inventory
preview = results.collect_garbage(retained, quiescent=True)  # dry-run default

# After reviewing preview, while the application is still stopped:
removed = results.collect_garbage(retained, quiescent=True, dry_run=False)
```

The host owns the complete inventory and quiescence assertion; this API cannot
discover external checkpoints or stop application writers. Do not run it with
an incomplete inventory. It verifies retained references before deleting any
unreferenced results or interrupted staging directories. Missing/corrupt retained
references and unexpected store entries fail closed. Deletions are permanent
and not transactional as a batch; a maintenance interruption may remove only
some candidates. Normal Agent execution never invokes maintenance automatically.

### Configure offload

```python
from msgflux.nn import ToolLibrary
from msgflux.nn.extensions import ToolOutputOffloadExtension
from msgflux.runtime import RuntimeResources

def report() -> dict:
    """Return a structured report."""
    return {"rows": [{"text": "example" * 10000}]}

results = RuntimeResources("./runtime-data").tool_result_store()
library = ToolLibrary(
    "reports", [report],
    extensions=[ToolOutputOffloadExtension(
        results, max_inline_bytes=32768, preview_bytes=2048,
    )],
)
response = library([("call_report", "report", {})])
descriptor = response.tool_calls[0].result
```

This extension runs after ordinary `after_tool` handlers, before `tool.end` and
normal tool feedback serialization. Register it on an existing Agent with
`agent.tool_library.register_extension(extension.name, extension)`; remove it
with the returned handle. No offload runs without explicit registration.

Successful large results become a JSON-compatible dictionary:

```json
{
  "type": "tool_result_reference",
  "reference": {
    "result_id": "res_0123456789abcdef0123456789abcdef",
    "size_bytes": 70022,
    "sha256": "<SHA256 of stored bytes>",
    "media_type": "application/json"
  },
  "preview": "<bounded text excerpt>",
  "truncated": true
}
```

The descriptor is illustrative; size and checksum are computed from actual
stored bytes. Dict/list outputs are stored as complete JSON, not `str(dict)`.
Plain strings, including text returned by an Agent used as a tool, are stored as
`text/plain; charset=utf-8`. Strings are not guessed to be JSON and reparsed.
Small supported outputs retain their original object and type. Shell results
also support offload while preserving their native type, as described below.
Binary and other unsupported provider-native typed outputs remain unchanged.

The default inline threshold is 32 KiB of serialized UTF-8 and the preview is
2 KiB. Both are host policy settings, not extra tool arguments. Preview size can
be zero and cannot exceed the inline threshold. These are starting values for
an opt-in policy, not universal model token or UI memory limits. The descriptor
itself has overhead beyond the preview budget. A JSON preview is a **text
excerpt**, not necessarily valid standalone JSON; only the stored file is the
complete document.

Encoding is incremental, including large JSON string values and keys. The probe
retains at most the inline threshold plus a bounded encoded chunk. Supported
JSON containers require string keys and JSON primitives; cycles, nonfinite
numbers, unsupported nested values and nesting deeper than 64 levels fail
explicitly rather than silently stringify data. Storage/encoding failures clear
the output and report a bounded processing error. They do not restore the original
large payload. The tool may already have performed side effects: do not retry
automatically. Async execution uses the existing hook worker-thread path; an
in-flight write may complete after cancellation and leave an unreferenced result.

Except for the incremental builtin Bash path below, offload is not a bound on
memory allocated inside the tool. It does not bound ordinary hook inputs,
tool-emitted custom events or the main Agent's streamed answer. Later custom
policies can also replace outcomes; extension composition remains host-owned.
This extension does not grant model access to shared results. `ReadFileTool`
does not yet resolve result URIs: the application must authorize and implement
result retrieval before asking the model to depend on offloaded content. UI code
can resolve the descriptor with `ToolResultRef` and use bounded store reads after
authorization. No shared root mount or prompt change is installed automatically.

`ArtifactRegistry` remains the existing in-memory presentation mechanism; durable
results are not automatically expanded into it. Authorized model-facing resource
reads remain a separate integration.

## Shell Results

The same extension handles `ShellResult` returned by `BashTool`. When the complete
serialized batch exceeds `max_inline_bytes`, it stores one JSON document with
every command's complete `stdout`, `stderr`, status and return code. The returned
`ShellResult` keeps all command statuses/return codes and gains a typed
`output_reference: ToolResultRef`. Its stdout/stderr fields contain UTF-8-safe
previews. The total `preview_bytes` budget is divided equally across both fields
of every command, so a noisy stdout cannot consume the stderr preview allocation.
Very small budgets may produce empty previews. This text budget excludes status
metadata and reference overhead.

Normal small Shell results omit `output_reference` when serialized, preserving
their previous shape. Already-offloaded Shell results are not offloaded again.
In native Responses history the reference is retained in internal metadata;
projection to function-call history preserves it as `output_reference`.

The [OpenAI shell protocol](https://developers.openai.com/api/docs/guides/tools-shell)
uses stdout/stderr strings and per-command outcomes. The adapter retains that
wire shape, with a compact JSON offload notice in the first stdout for the model.
Internal metadata is stripped before sending the API request. UI consumers use
the structured reference, not that notice. Custom tools returning `ShellResult`
are transformed after execution. The builtin Bash tool additionally supports the
incremental path below. Offload cannot recover bytes discarded by a backend.

### Incremental Bash Capture

```python
from msgflux.nn import ToolLibrary
from msgflux.nn.extensions import ToolOutputOffloadExtension
from msgflux.tools.builtin import BashTool

library = ToolLibrary(
    "workspace", [BashTool(cwd="/")],
    extensions=[ToolOutputOffloadExtension(
        results,
        max_inline_bytes=32 * 1024,
        preview_bytes=2048,
        max_capture_bytes=8 * 1024 * 1024,
    )],
)
```

This sets an 8 MiB raw stdout/stderr budget for the entire Bash batch. The
extension supplies an optional hidden `shell_capture` dependency through the tool
context registry; it is never a model argument. Removing the extension restores
the ordinary buffered path. An authorized environment with a compatible executor
is still required: this installs no host shell executor or process permission.

Each command's streams go to two private temporary files shared by the batch,
with byte ranges preserving command boundaries. After execution, the
complete JSON batch is encoded incrementally into the result store. Large
results retain previews and a reference; small results return inline. Temporary
files close on success, failure and cancellation. Timeouts preserve bytes already
received. Commands after an exhausted batch budget are `not_executed`; exceeding
the current command's budget fails instead of reporting a complete result.

`max_capture_bytes` defaults to 1,000,000 raw bytes. This differs from the inline
threshold and the store's `max_result_bytes`: JSON escaping can increase size.
Budget for temporary captures and staged JSON coexisting, and concurrent calls;
these are per-call limits, not an aggregate disk quota.

For bounded executor memory, implement
`ProcessExecutor.execute_stream(..., on_output=...)`. Await
`on_output("stdout", chunk)` or `on_output("stderr", chunk)` with byte chunks no
larger than 64 KiB, then return `ProcessResult(returncode)` with empty buffers.
The environment retains its permission, binding, isolation, abort and deadline
checks. The callback applies backpressure: adapters must not enqueue unbounded
pending callbacks. `execute_stream` is the required backend method; the base
`execute()` method is only a bounded collecting adapter for callers that need a
buffered `ProcessResult`.

Disk operations are awaited in workers. Cancellation joins an in-flight operation
before closing its files; it can wait for storage and leave a complete orphaned
result if publication completed. No reference is returned before publication.
Process adapters must terminate and reap children on cancellation and failures;
capture does not implement sandboxing.

For an asyncio-based backend, the helper `msgflux.runtime.drain_subprocess`
drains an **already launched and authorized** process:

```python
from msgflux.runtime import ProcessResult, drain_subprocess

async def collect_authorized_process(process, request, on_output, abort_signal):
    code = await drain_subprocess(
        process, on_output,
        max_output_bytes=request.max_output_bytes,
        timeout_seconds=request.timeout_seconds,
        abort_signal=abort_signal,
    )
    return ProcessResult(code)  # output was delivered through the callback
```

The backend must launch with stdout/stderr pipes and enforce its declared
workspace/network policies **before** this helper runs. Draining is concurrent
and bounded; timeout, cancellation, quota and sink failures trigger child cleanup.
On POSIX, `owns_process_group=True` additionally signals the process's group:
use it only when the backend launched that process with `start_new_session=True`.
Without it, only the direct child is signalled. Descendants escaping their group
require backend-level containment; neither this helper nor a local directory
workspace provides an OS sandbox. Invalid helper arguments are rejected before
taking cleanup ownership, so the launcher still owns the child in that case.

## Retrieve Complete Output from an Event

```python
from contextlib import closing
from msgflux.runtime import get_tool_result_reference

# Run this synchronous delivery function outside an async event loop.
def transmit_complete_output(event, results, authorize_result, send_chunk):
    if event.type != "tool.end":
        return
    reference = get_tool_result_reference(event.data.get("result"))
    if reference is None:
        return  # ordinary inline output
    authorize_result(reference.result_id)  # application check; raises on denial
    offset = 0
    with closing(results.iter_bytes(reference, chunk_size=65536)) as chunks:
        for chunk in chunks:
            send_chunk(reference.result_id, offset, chunk)
            offset += len(chunk)
```

`get_tool_result_reference()` accepts the generic offload descriptor, a typed
result or dictionary with `output_reference`, or a stored native/function tool
output item. Function outputs also retain the reference in internal metadata,
independently of their serialized model-facing text. Metadata is not sent as an
extra provider wire field.
It returns the validated `ToolResultRef`, or `None` for ordinary outputs, and
rejects malformed structured references. It does not parse free text, perform
I/O or authorize access. No host path needs to be sent to a client: the configured
store resolves the ID independently of its physical location.

The example uses application callbacks for authorization and transmission.
`send_chunk` must consume or await bounded delivery before the next chunk is read;
do not enqueue all chunks or join the complete content first. A remote UI asks
its authorized backend for this result, rather than opening a server-local path.
The backend can serve a file download, ranged reads or a separate chunk stream.
Chunk offsets are byte offsets, and the media type describes the stored document:
a Shell result is complete JSON, not raw stdout. Closing the iterator releases
file descriptors even when transmission fails.

These extra deliveries are owned by the application, not automatically published
as Agent execution events. `tool.end`, normal model feedback and checkpoints
remain compact. Reinserting a whole result into the Agent event queue would
defeat offload's retention benefit. References provide location-independent
identity, not a public download URL or a bearer permission.
