# Installed Agent Extension Packages

Publish an ordinary Python package to make an `AgentExtension` available by
name in the service configuration. Installation makes it discoverable;
`agents.<id>.extensions` explicitly activates it for an agent.

The `AgentExtension` API is available in `1.0.0rc2`. Package discovery and the
configuration and CLI integration described here require the current development
version; they are not included in that release.

## Create A Small Package

This package contains two independent extensions: a configurable instruction
suffix and a counter of model requests within a durable run.

```text
demo-agent-extensions/
├── pyproject.toml
└── src/
    └── demo_agent_extensions/
        ├── __init__.py
        └── extensions.py
```

Save the following as `pyproject.toml`:

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "demo-agent-extensions"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = ["msgflux>=1.0.0rc2"]

[project.entry-points."msgflux.agent_extensions"]
prompt_suffix = "demo_agent_extensions.extensions:PromptSuffix"
request_counter = "demo_agent_extensions.extensions:RequestCounter"

[tool.hatch.build.targets.wheel]
packages = ["src/demo_agent_extensions"]
```

An entry point maps a discovery name to a synchronous extension factory. The
factory may be an `AgentExtension` class, as in this example, or a callable
returning a fresh extension instance.

Leave `src/demo_agent_extensions/__init__.py` empty. Save both implementations
in `src/demo_agent_extensions/extensions.py`:

```python
from dataclasses import replace

from msgflux.nn import AgentExtension
from msgflux.nn.hooks import Hook, ModelContext


class PromptSuffix(AgentExtension):
    def __init__(self, text: str = "Keep answers concise."):
        super().__init__("prompt_suffix")
        self.text = text

    def hooks(self):
        return (Hook(event="transform_system_prompt", handler=self.append),)

    def append(self, context: ModelContext) -> ModelContext:
        return replace(
            context,
            system_prompt=f"{context.system_prompt}\n\n{self.text}".strip(),
        )


class RequestCounter(AgentExtension):
    def __init__(self):
        super().__init__("request_counter")

    def hooks(self):
        return (Hook(event="before_request", handler=self.count),)

    def count(self, context):
        state = self.durable_state()
        state["requests"] = state.get("requests", 0) + 1
```

`PromptSuffix` changes the rendered system prompt for each model request; it
does not mutate the Agent's canonical prompt. Hook contexts are dataclasses,
so the handler returns a replaced `ModelContext`.

`RequestCounter` keeps serializable state under its extension name. The count
is restored from the last checkpoint when resuming that run. A new `run_id`
starts at zero: this is not a thread-wide counter or an exactly-once count of
provider requests. An increment made after the last committed checkpoint can be
lost on a crash. Neither example needs external resources or cleanup.

## Install In The Server Environment

For a project that runs the service through `uv run`, install the package in
that project:

```bash
uv add ./demo-agent-extensions
```

For a service launched from a `uv tool` installation of msgFlux, inject the
package into that tool's environment:

```bash
uv tool inject msgflux ./demo-agent-extensions
```

Run these commands from the directory containing `demo-agent-extensions/`.
Install the package where the **server** runs; installing it only in a remote
client's environment does not add an extension to the service. Restart an
already running daemon after changing packages or configuration; discovery
does not hot-reload a live Agent.

## Activate And Configure

### Edit Configuration From The CLI

With the package installed, activate an extension for an existing agent:

```bash
msgflux extensions add request_counter --agent coding
msgflux extensions add prompt_suffix --agent coding \
  --as coding_prompt \
  --option 'text=Keep answers concise and report verification.'
msgflux extensions remove prompt_suffix --agent coding
```

These commands edit the existing `~/.msgflux/config.toml`. Use `--config PATH`
for another file and install `msgflux[service]` for configuration editing. They
do not download packages, create a missing config, start a daemon, or restart a
running service. Comments and formatting in the TOML file are preserved.

`--agent` is required. The first command adds `request_counter` to that agent's
selection without storing constructor defaults. The second creates the
`coding_prompt` alias with an explicit resolved factory, copies any configured
options from `prompt_suffix`, applies the supplied override, and selects the
alias for `coding`. It does not change other agents selecting `prompt_suffix`.
An alias is an additional registration: adding it does not replace an existing
`prompt_suffix` selection. The third command removes that original selection,
if present, so only the alias runs on this agent. Keep both selections only when
you intend both extension instances to run.

Repeat `--option KEY=VALUE` to override several constructor keyword arguments.
Only the supplied overrides are changed; omitted options retain their configured
values or factory defaults. Keys are flat keyword names, not dotted paths, and
duplicate keys in one command are rejected.

Values use TOML syntax for booleans, numbers, arrays, inline tables, and quoted
strings. An unquoted text value is accepted as a string. Shell quoting protects
spaces but does not itself add TOML quotes: `--option 'text=123'` sets an integer,
whereas `--option 'text="123"'` sets the string `"123"`. Each option must be
accepted by the selected factory's signature.

Options under `extensions.<name>` are shared by every agent selecting that
name. An add command that would change options used by another agent is refused;
use `--as` to create a separate alias instead. An existing alias cannot be
redirected to a different factory by an add command.

Inspect available constructor options before configuring them:

```bash
msgflux extensions inspect prompt_suffix --options
msgflux extensions inspect prompt_suffix --options --json
```

`--options` imports only the selected factory and inspects its signature; it does
not call the constructor. Plain `inspect` remains metadata-only. Missing required
arguments or arguments that do not fit the signature prevent an add command
from writing anything. Constructor-specific value checks still happen when the
service builds the Agent. A rejected edit leaves the original file unchanged.

Remove an extension from one agent without deleting its reusable configuration:

```bash
msgflux extensions remove coding_prompt --agent coding
```

Add and remove are idempotent. Removing a selection preserves the global factory
and options, and does not alter other agents or subagents. Each command returns
the configuration path, agent and extension identities, whether the file changed,
and `restart_required`. Add `--json` for a structured receipt. A successful edit
does not change a running daemon: restart it explicitly when ready.

### Edit TOML Directly

Add the selected names to an agent in `config.toml`:

```toml
[defaults]
agent = "coding"
model = "provider/model-id"

[agents.coding]
harness = "coding"
extensions = ["prompt_suffix", "request_counter"]

[extensions.prompt_suffix.options]
text = "Keep answers concise and include the verification result."
```

Replace `provider/model-id` with your model. The extension options become
keyword arguments to its factory. Omitting the options table uses the factory's
defaults; `request_counter` needs no options. A top-level extension definition
alone does not activate it.

Each agent/session construction receives fresh extension instances. Selecting
the same package in two threads does not share mutable instance state.
Subagents do not inherit their parent's extension selections: add the names
to each child agent that needs them. Durable state remains scoped to the
extension and run, independently of instance construction. Keep configured
names and saved state formats compatible across deployments: changing an alias
or upgrading a package does not migrate existing checkpoint state automatically.

You can also use these classes directly through
`Agent(extensions=[PromptSuffix(), RequestCounter()])`; package discovery is
only needed for configuration by name.

### Use The Catalog From Python

The SDK exposes the same catalog for applications that construct their own
Agents. This example lists metadata, creates the configured extension, and
attaches it to an Agent using an existing `model` instance:

```python
from msgflux.nn import Agent, AgentExtensionCatalog, ExtensionConfig

catalog = AgentExtensionCatalog({
    "prompt_suffix": ExtensionConfig(
        options={"text": "Keep answers concise and report verification."},
    ),
})

for info in catalog.list():
    print(info.name, info.factory)

for option in catalog.options("prompt_suffix"):
    print(option.name, option.required, option.default_available, option.default)

extension = catalog.create("prompt_suffix")
agent = Agent(name="assistant", model=model, extensions=[extension])
```

`list()` reads metadata without importing plugins. `options()` imports the
selected factory and returns `ExtensionOption` records without calling its
constructor. Records describe annotations, required arguments, defaults, and
variadic keyword support. A complex default that cannot be represented safely
has `has_default=True` and `default_available=False`; its value is not exposed.

`create()` imports and calls the selected factory. Creating another Agent
requires another `create()` call,
not reusing an extension already registered on the first Agent. An explicit
`ExtensionConfig(factory="module:attribute", options={...})` overrides entry
points just as the TOML configuration does.

For manually constructed Agents, extensions that own resources use the existing
registration/removal lifecycle: remove by name with
`agent.remove_extension("prompt_suffix")`, or keep the handle returned by
`agent.register_extension(...)` and call `handle.remove()`. Use
`await handle.aremove()` for asynchronous extension cleanup. See
[Agent Extensions](extensions.md) for active-run cleanup boundaries, hooks,
tools, and state ownership.

### Explicit Factory Overrides

Use a configured factory when an entry point is unavailable or multiple
installed distributions advertise the same name:

```toml
[agents.coding]
harness = "coding"
extensions = ["prompt_suffix"]

[extensions.prompt_suffix]
factory = "demo_agent_extensions.extensions:PromptSuffix"

[extensions.prompt_suffix.options]
text = "Prefer a short answer with concrete next steps."
```

This replaces the earlier `agents.coding` table; do not declare it twice.
An explicit `module:attribute` factory wins over entry-point discovery for that
configured name. Without an override, duplicate entry-point names are ambiguous
and must be resolved before activation; msgFlux does not choose a package by
installation order.

## Inspect Before Starting A Service

```bash
msgflux extensions ls
msgflux extensions inspect prompt_suffix --json
msgflux extensions inspect prompt_suffix --config ./config.toml --json
msgflux config validate --config ./config.toml
```

The discovery commands read installed distribution metadata and configuration.
Plain `ls` and `inspect` do not start a service, import extension modules, or
construct extensions. Adding `--options` to `inspect` imports the selected
factory for signature inspection without constructing it.
With no `--config`, they include the default `~/.msgflux/config.toml` when it
exists. Use `--json` on either discovery command for structured output.

`config validate` goes further for selected extensions: it imports their
factories and checks that the configured keyword arguments fit their
signatures. It does not call constructors, build models, or make provider
requests. Factory modules must keep imports free of network activity and other
side effects. Validation of values implemented inside a constructor happens
only when the service builds the selected agent.

Installing an entry point grants no automatic activation. Only explicitly
selected extensions are resolved for construction. Their implementation is
trusted Python code running in the service process with its permissions.
