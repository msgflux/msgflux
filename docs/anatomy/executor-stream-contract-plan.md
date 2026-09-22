# Executor stream contract

## Scope

Make incremental process delivery the required `ProcessExecutor` backend
contract. `execute_stream` is the only abstract backend operation; the public
`execute` method is a concrete bounded adapter that collects callback chunks
for callers that still need a `ProcessResult`.

## Order

1. Update `runtime/environment.py` while preserving live workspace,
   permission, capability, abort, timeout, and output-limit checks.
2. Migrate repository executor fixtures and examples to `execute_stream`.
3. Update streaming tests and the executor documentation paragraph.
4. Run focused streaming/workspace/offload tests and Ruff.

## Risks and validation

- Existing third-party executors implementing only `execute` will need the
  required method; this is an intentional contract change.
- The concrete adapter must bound callback chunk size and aggregate bytes,
  cancel/reap through the caller's existing abort/timeout path, and reject
  duplicate buffered output from a streaming backend.
- No host executor, sandbox claim, approval bypass, or filesystem policy is
  introduced here.
