# AutoModule

## ✦₊⁺ Overview

`AutoModule` packages a msgFlux module as a small, reusable bundle that can be
loaded from a local directory, GitHub, or Hugging Face Hub.

The bundle has three files:

```text
repo/
├── module.json
├── module.py
└── state.json
```

- `state.json` is the exact output of `module.state_dict()`.
- `module.json` is the editable loading contract.
- `module.py` is optional Python code for factories, custom classes, tools,
  hooks, MCP clients, or other runtime pieces that do not belong in a state dict.

The core idea is that exported model positions become **model slots**. Users
replace those slots by friendly keys from `module.json`, not by internal paths.

### Key Features

- **State-first export**: write an exact `state.json` from `state_dict()`.
- **Editable model slots**: rename generated slots into human-facing aliases.
- **Safe model replacement**: replace models before the original provider is
  deserialized.
- **Local-first authoring**: test bundles with `local://...` before publishing.
- **Remote loading**: load the same bundle from GitHub or Hugging Face Hub.
- **Optional Python entrypoints**: add `create()` or `get_class()` only when the
  bundle needs runtime code.

## Minimal Local Example

This example is fully local and does not require network access or API keys. It
uses the `fake` provider so you can verify the export/load contract without
calling a model.

Create `author_export.py`:

```python
from pathlib import Path

import msgflux as mf
from msgflux import nn


agent = nn.Agent(
    name="ticket_router",
    model=mf.Model.chat_completion("fake/default"),
    instructions="Route support tickets to the correct team.",
    expected_output="Return the team name and a one-line rationale.",
    config={"return_messages": True},
)

bundle = mf.AutoModule.export(agent, path="./dist/ticket-router")

print(f"exported to: {bundle}")
print((Path(bundle) / "module.json").read_text())
```

Run it:

```bash
python author_export.py
```

The export creates:

```text
dist/ticket-router/
├── module.json
├── module.py
└── state.json
```

`state.json` is the exact state:

```json
{
  "expected_output": "Return the team name and a one-line rationale.",
  "instructions": "Route support tickets to the correct team.",
  "name": "ticket_router",
  "config": {
    "return_messages": true
  },
  "generator.model": {
    "msgflux_type": "model",
    "provider": "fake",
    "model_type": "chat_completion",
    "state": {
      "model_id": "default",
      "kwargs": {}
    }
  }
}
```

`module.json` contains the generated model slot:

```json
{
  "schema_version": 1,
  "msgflux_version": ">=0.5.0",
  "entrypoint": "module.py",
  "state": "state.json",
  "models": {
    "generator-model": {
      "path": "generator.model",
      "provider": "fake",
      "model_type": "chat_completion",
      "model_id": "default",
      "state": {
        "model_id": "default",
        "kwargs": {}
      }
    }
  },
  "files": [],
  "metadata": {}
}
```

The generated key `generator-model` is intentionally mechanical. Authors can
rename it to a clearer public alias without changing the internal `path`:

```json
"models": {
  "main": {
    "path": "generator.model",
    "provider": "fake",
    "model_type": "chat_completion",
    "model_id": "default"
  }
}
```

After renaming the slot to `main`, users override it with:

```python
models={"main": "openai/gpt-4.1-mini"}
```

## Load Into An Existing Instance

`AutoModule.export()` writes an empty `module.py` by default. You can still load
the exported state into an instance you create yourself.

Create `consumer_load_into.py`:

```python
import msgflux as mf
from msgflux import nn


target = nn.Agent(
    name="placeholder",
    model=mf.Model.chat_completion("fake/placeholder"),
    instructions="This will be replaced by state.json.",
)

ref = mf.AutoModule("local://./dist/ticket-router")
ref.load_into(
    target,
    models={"generator-model": "fake/user-model"},
)

print(target.name)
print(target.instructions)
print(target.model.model_id)
```

Run it:

```bash
python consumer_load_into.py
```

Expected output:

```text
ticket_router
Route support tickets to the correct team.
user-model
```

This path does not execute `module.py`. It only reads `module.json` and
`state.json`.

## Add A Factory For create()

To support `AutoModule.create(...)`, fill `module.py` with a factory that creates
a compatible skeleton. The state will be loaded after the factory returns.

Edit `dist/ticket-router/module.py`:

```python
import msgflux as mf
from msgflux import nn


def create():
    return nn.Agent(
        name="placeholder",
        model=mf.Model.chat_completion("fake/placeholder"),
        instructions="This will be replaced by state.json.",
    )
```

Create `consumer_create.py`:

```python
import msgflux as mf


agent = mf.AutoModule.create(
    "local://./dist/ticket-router",
    trust_remote_code=True,
    models={"generator-model": "fake/create-user-model"},
)

print(agent.name)
print(agent.instructions)
print(agent.model.model_id)
```

Run it:

```bash
python consumer_create.py
```

Expected output:

```text
ticket_router
Route support tickets to the correct team.
create-user-model
```

`trust_remote_code=True` is required because `create()` executes Python from
`module.py`.

## Add A Class Hook

If you want users to get the class without instantiating it, expose
`get_class()` in `module.py`.

```python
import msgflux as mf
from msgflux import nn


class TicketRouter(nn.Agent):
    def __init__(self):
        super().__init__(
            name="placeholder",
            model=mf.Model.chat_completion("fake/placeholder"),
            instructions="This will be replaced by state.json.",
        )


def get_class(name=None):
    return TicketRouter
```

Then consumers can choose when to instantiate:

```python
import msgflux as mf


ref = mf.AutoModule("local://./dist/ticket-router")

TicketRouter = ref.get_class(trust_remote_code=True)
agent = TicketRouter()

ref.load_into(
    agent,
    models={"generator-model": "fake/class-user-model"},
)

print(agent.name)
print(agent.model.model_id)
```

For multiple classes, use the optional `name` argument:

```python
def get_class(name=None):
    classes = {
        "router": TicketRouter,
        "reviewer": TicketReviewer,
    }
    return classes[name or "router"]
```

```python
Reviewer = ref.get_class("reviewer", trust_remote_code=True)
```

## Model Replacement Semantics

Model slots are resolved from `module.json`:

```json
"models": {
  "planner": {
    "path": "planner.generator.model",
    "provider": "fake",
    "model_type": "chat_completion",
    "model_id": "planner"
  },
  "writer": {
    "path": "writer.generator.model",
    "provider": "fake",
    "model_type": "chat_completion",
    "model_id": "writer"
  }
}
```

Users replace those slots by alias:

```python
workflow = mf.AutoModule.create(
    "gh://owner/research-workflow",
    revision="abc123",
    trust_remote_code=True,
    models={
        "planner": "openai/gpt-4.1",
        "writer": mf.Model.chat_completion("openai/gpt-4.1-mini"),
    },
)
```

Accepted replacement values:

- `"provider/model-id"` strings.
- model instances.
- serialized model dictionaries.

Replacements are applied inside `load_state_dict()` before the original
serialized model is deserialized. If the exported state contains
`google/gemini-*` and the user passes `openai/gpt-*`, the Google provider is not
initialized.

## State-Only Bundles

A bundle does not need `create()` or `get_class()` if consumers will create the
target instance themselves:

```python
ref = mf.AutoModule("local://./dist/ticket-router")
ref.load_into(target, models={"generator-model": "fake/user-model"})
```

In that case:

- `check_requirements()` works.
- `load_into()` works.
- `create()` raises a clear error until `module.py:create`, `module.py:get_class`,
  `factory`, or `class` is provided.
- `get_class()` raises a clear error until `module.py:get_class` or `class` is
  provided.

## Publishing

Once the local bundle works, publish the folder as a repository:

```text
research-workflow/
├── module.json
├── module.py
└── state.json
```

Load from GitHub:

```python
workflow = mf.AutoModule.create(
    "gh://owner/research-workflow",
    revision="abc123",
    trust_remote_code=True,
    models={"planner": "openai/gpt-4.1"},
)
```

Load from Hugging Face Hub:

```python
workflow = mf.AutoModule.create(
    "hf://owner/research-workflow",
    revision="abc123",
    trust_remote_code=True,
    models={"planner": "openai/gpt-4.1"},
)
```

Supported repository identifiers:

```python
mf.AutoModule("local://./dist/ticket-router")
mf.AutoModule("owner/repo")          # GitHub by default
mf.AutoModule("gh://owner/repo")     # GitHub
mf.AutoModule("github.com/owner/repo")
mf.AutoModule("hf://owner/repo")     # Hugging Face Hub
mf.AutoModule("huggingface.co/owner/repo")
```

Use pinned revisions for reproducible remote loads. Branches like `main` can
change over time.

## Cache And Offline Loading

GitHub and Hugging Face files are cached under the AutoModule cache directory.
The default is:

```text
~/.cache/msgflux/auto/
```

Useful options:

```python
ref = mf.AutoModule(
    "gh://owner/research-workflow",
    revision="abc123",
    cache_dir="./.msgflux-cache",
    local_files_only=True,
)
```

- `revision`: branch, tag, or commit SHA.
- `cache_dir`: override where files are stored.
- `local_files_only=True`: fail if a file is missing from cache.
- `force_download=True`: refresh cached files.

Local bundles (`local://...`) are read directly from disk and do not use the
cache.

## Authoring Guidelines

- Use `fake/*` models in exported skeletons unless the default provider is safe
  to initialize in every consumer environment.
- Treat generated model aliases as editable API. Rename them before publishing.
- Keep `state.json` faithful. Put human-facing decisions in `module.json`.
- Put tools, hooks, MCP clients, filesystem resources, and other runtime-only
  objects in `module.py`.
- Prefer pinned remote revisions in production examples.

## Troubleshooting

`AutoModule ... requires trust_remote_code=True`

: The operation needs to execute Python from `module.py`. Pass
  `trust_remote_code=True` only for repositories you trust.

`Unknown model slot`

: The key in `models={...}` does not exist in `module.json`. Check the `models`
  section and use one of the declared aliases.

`state key is not a serialized model`

: A model slot points to a path that exists but does not contain a serialized
  model. Fix the slot `path` in `module.json`.

`create() requires module.py:create, module.py:get_class, factory, or class`

: The bundle is state-only. Use `load_into()` or add a factory/class hook to
  `module.py`.
