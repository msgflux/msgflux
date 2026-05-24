# AutoModule

`AutoModule` loads reusable msgFlux modules from GitHub or Hugging Face Hub.
It uses a small `module.json` manifest and, when needed, a `module.py` factory.

The recommended remote layout is:

```text
repo/
├── module.json
├── module.py
└── state.json
```

`module.json` is the manifest:

```json
{
  "schema_version": 1,
  "msgflux_version": ">=0.5.0",
  "factory": "module.py:create",
  "state": "state.json",
  "models": {
    "default": "generator.model"
  },
  "files": ["tools.py"]
}
```

Fields:

- `factory`: callable used by `create()`. Remote Python code requires
  `trust_remote_code=True`.
- `class`: optional class entrypoint used by `get_class()`.
- `state`: optional JSON state exported from `module.state_dict()`.
- `models`: aliases for model override targets in the created module.
- `files`: extra repository files to download before importing `module.py`.

## Loading

Create an instance:

```python
import msgflux as mf

agent = mf.AutoModule.create(
    "hf://msgflux/cli-agent",
    trust_remote_code=True,
    models={"default": mf.Model.chat_completion("openai/gpt-4.1-mini")},
)
```

Or inspect first:

```python
ref = mf.AutoModule("gh://msgflux/cli-agent", revision="main")

info = ref.check_requirements()
agent = ref.create(
    trust_remote_code=True,
    models={"default": "openai/gpt-4.1-mini"},
)
```

Load only the exported class:

```python
AgentClass = ref.get_class(trust_remote_code=True)
```

## Factory

The factory should recreate runtime pieces that are not safe or portable in a
state dict, such as Python tools, MCP clients, hooks, and local resources.
Use the fake provider as a placeholder model when the real model will be loaded
from `state.json` or overridden by the user.

```python
import msgflux as mf
import msgflux.nn as nn

from tools import inspect_repo


def create(config=None, module_path=None):
    return nn.Agent(
        name="cli_agent",
        model=mf.Model.chat_completion("fake/placeholder"),
        tools=[inspect_repo],
    )
```

`create()` runs in this order:

1. call the factory or instantiate the class;
2. apply `state.json` through `load_state_dict()`;
3. apply user model overrides.

This means user-provided `models={...}` has precedence over the exported state.

## Sources

Supported repository identifiers:

```python
mf.AutoModule("owner/repo")          # GitHub by default
mf.AutoModule("gh://owner/repo")     # GitHub
mf.AutoModule("github.com/owner/repo")
mf.AutoModule("hf://owner/repo")     # Hugging Face Hub
mf.AutoModule("huggingface.co/owner/repo")
```

Use `revision` for reproducible loads:

```python
ref = mf.AutoModule("hf://msgflux/cli-agent", revision="abc123")
```
