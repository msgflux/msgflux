# AGENTS.md

Guidance for agents working in this repository.

Use `CONTRIBUTING.md` as the source of truth for the general development
workflow, including branch naming, commits, validation commands, and PR creation.

## Before Changing Code

- Understand the existing flow before adding new code.
- Check in detail which code paths call, instantiate, inherit from, serialize,
  document, or otherwise depend on the area being changed.
- Also inspect the code paths that will start consuming the new behavior.
- Keep changes DRY: reuse existing helpers, abstractions, provider patterns, and
  documentation structure before adding new code paths.
- Do not duplicate code or logic unless the plan or PR description explains why
  duplication is intentional.
- Preserve local patterns for APIs, naming, errors, tests, and documentation.

## Planning

- For complex tasks, write a plan before implementing.
- The plan must list affected files, implementation order, risks, required
  tests, and documentation updates.
- If questions come up during planning, ask the developer before proceeding.
- If the task requires too many changes for one reviewable PR, split the plan
  into incremental PRs. When needed, make the dependency order between PRs
  explicit.

## Editing

- When moving blocks of code or documentation, consider using a small Python
  script or another structured transformation instead of copying text manually.
- Review the resulting diff to confirm that only the intended block moved and no
  content was lost or duplicated.

## Providers

- New providers must follow the existing patterns in
  `src/msgflux/models/providers/`.
- Before adding a provider, compare it with similar providers, exports in
  `__init__.py`, tests, examples, and related documentation.
- Do not introduce divergent provider behavior without explaining the reason in
  the plan or PR description.

## Documentation

- Architectural changes may document internal behavior in `docs/anatomy/`.
- Any public API change must update or add documentation in `docs/learn/`.
- `docs/learn/` pages must explain how the feature works, include code
  examples, and describe what each example does.
- Follow the existing documentation style, including heading structure,
  examples, tables, admonitions, and `mkdocs.yml` navigation updates when
  applicable.

## Scope and Commits

- Do not make unrelated changes outside the developer request or agreed plan.
- Do not stage or commit files unless the developer explicitly asks for it.
- When asked to commit, include only files changed by the agent for the current
  request. Review `git status` and the relevant diffs before staging.
- Never include unrelated user changes, generated artifacts, caches, build
  outputs, or local artifacts in a commit unless they are explicitly required by
  the plan.
- Keep commits focused on related changes. Do not create one broad commit that
  mixes independent fixes, refactors, docs, and feature work.
- If the working tree already contains unrelated changes, leave them untouched
  and mention them when reporting status.
- If a required change falls outside the current plan, update the plan or ask the
  developer before continuing.

## Pull Requests

- PR descriptions must be precise and detailed.
- Include motivation, previous behavior, new behavior, relevant files, public
  API impact, executed tests, and updated documentation.
- Include code snippets when they help explain the change.
- For dependent PRs, state the merge order and what each PR delivers.

## Verification

- Run the relevant checks described in `CONTRIBUTING.md`.
- For documentation changes, validate with MkDocs when applicable.
- If a test or validation command cannot be run, document the reason in the PR.
