# Structured retrieval and Shell offload

1. Move the location-independent `ToolResultRef` schema into a dependency-light
   private module, re-exporting it from its current runtime paths. Shell result
   types must not import the runtime package and introduce initialization cycles.
2. Add optional typed `output_reference` to `ShellResult`, omitted when absent.
   Add a public `get_tool_result_reference(result)` helper for ordinary offload
   envelopes, typed Shell results and their JSON-decoded representation. No free
   text parsing, host paths, URL credentials or implicit authorization.
3. Extend ToolOutputOffloadExtension: serialize the complete Shell batch as JSON
   incrementally, retaining status/returncode for every command and a shared
   bounded stdout/stderr preview budget. Keep the native result type. Store one
   immutable result per batch. Do not re-offload an already referenced ShellResult.
4. Preserve references through the OpenAI shell adapter in internal history
   metadata. The wire schema still uses stdout/stderr strings and outcomes; a
   compact JSON notice in stdout identifies truncation for the model. Never send
   internal metadata fields to the API. Preserve references when projecting to
   function-call history and across checkpoint hydration.
5. Document UI retrieval via the structured helper and existing ranged/chunked
   store reads. Consumers may forward these bytes on their own transport, after
   authorization, without republishing complete outputs into the Agent event hub.

Affected files: private reference module; runtime/tool_results.py and exports;
tools/shell.py; nn/extensions/tool_output.py; models/tool_adapters/openai_shell.py;
focused offload/native-shell/store tests; resources and streaming learning docs.

Tests: imports/public identity, absent-field compatibility, large stdout/stderr,
Unicode/batch shared budget, original status/exit codes, native and function-call
projection, streamed tool.end without complete content, reference extraction after
JSON serialization, malformed references, checkpoint round trip and missing data.
Run focused and full offline tests, Ruff and MkDocs. No real shell or paid calls.

Limits: output is still captured by the tool before this extension runs. This
reduces downstream retention, not the executor's initial allocation. Future
incremental process capture remains separate. Artifact store selection and access
control belong to the host; references are not bearer capabilities. No commits.

Protocol reference inspected: https://developers.openai.com/api/docs/guides/tools-shell
