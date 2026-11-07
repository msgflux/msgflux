# Dynamic Chat Providers

Define a chat provider alias at runtime — in TOML or Python — without adding a
built-in provider to msgFlux. An alias derives its complete runtime from a
registered base provider, or delegates to a model class from an installed
package. See [Chat Completion](chat_completion.md) for the built-in providers
and [Accounts](../nn/agent/accounts.md) for credential storage.

## Define an alias in TOML

Each `[providers.<name>]` table declares one alias. Exactly one of `base` or
`factory` is required. `base_url` and `api_key_env` override the base
provider's endpoint and key environment variable. `options` passes supported,
non-secret constructor parameters (for example `temperature`); reserved names
such as `model_id`, `provider`, and `credential_resolver` cannot be set there,
and provider credentials must never be placed in `options`.

```toml
[providers.acme]
base = "anthropic"
base_url = "https://api.acme.example"
api_key_env = "ACME_API_KEY"

[providers.acme.options]
temperature = 0.7
```

This example creates an `acme` alias that speaks Anthropic's Messages protocol
at the Acme endpoint, reads its key from `ACME_API_KEY` at request time, and
defaults `temperature` to `0.7` (per-call constructor arguments still win).

## Define an alias in Python

`ProviderConfig` is imported from `msgflux.models.catalog`; `ProviderCatalog`
is exported from the `msgflux` root. Pass the catalog explicitly to
`Model.chat_completion`:

```python
import msgflux as mf
from msgflux import ProviderCatalog
from msgflux.models.catalog import ProviderConfig

catalog = ProviderCatalog({
    "acme": ProviderConfig(
        base="anthropic",
        base_url="https://api.acme.example",
        api_key_env="ACME_API_KEY",
    ),
})

# "acme/model" resolves through the catalog; the model ID after "/" is
# passed through as-is — catalogs do not enumerate or validate model IDs.
model = mf.Model.chat_completion("acme/model", provider_catalog=catalog)
print(model("Hello!").consume())
```

This builds a catalog holding the `acme` declaration and creates a chat model
whose provider identity is `acme`. Without a catalog, `acme/model` would fail
as an unknown provider.

## What `base` inherits

Deriving from `base` inherits the base class's full runtime: wire protocol,
reasoning codecs and history handling, capabilities, transport, and the auth
mechanism (for example, Anthropic's `x-api-key` header). There is no separate
protocol flag — choosing the base selects the runtime, and `options` only
overrides supported non-secret constructor parameters.

The alias keeps its own identity throughout: its provider name in model IDs,
telemetry, and checkpoints; its own account namespace; and its own
`api_key_env`. It never silently falls back to the base provider's key
environment variable. If omitted for an API-key base, the variable defaults
to `<ALIAS>_API_KEY` (upper case, with punctuation replaced by underscores).
Set `api_key_env` explicitly to choose another variable, or store a key in an
account, below.

## Scoped, not global

Catalog declarations are scoped to the catalog. Nothing is registered
globally, so two catalogs may define `acme` differently without interfering.
`catalog.use()` makes a catalog the ambient scope for nested resolution —
useful for code paths such as model gateways that create models internally:

```python
with catalog.use():
    model = mf.Model.chat_completion("acme/model")
```

This example makes `catalog` active for the block; models created inside
resolve `acme` from it, and the previous scope is restored on exit.

## Use an installed factory class

When a base provider is not enough, point `factory` at a class that is already
installed on the host, using the `package.module:Class` form. The class is a
normal model class — for example, a subclass of a built-in provider that sets
its own provider identity, key variable, and endpoint:

```python
# installed_package/module.py — shipped in a package on the host
from msgflux.models.providers.anthropic import AnthropicChatCompletion
from msgflux.models.registry import register_model


@register_model
class AcmeChatCompletion(AnthropicChatCompletion):
    provider = "acme"
    api_key_env = "ACME_API_KEY"
    base_url = "https://api.acme.example"
    base_url_env = None
```

```toml
[providers.acme]
factory = "installed_package.module:AcmeChatCompletion"
```

The class above reuses Anthropic's runtime under the `acme` identity. The
factory path is imported only when the alias is resolved, and config files are
trusted inputs: factories are code already installed on the host, so only load
provider config from sources you trust. Automatic discovery of provider
packages via entry points is deferred — declare each alias explicitly, in TOML
or Python. Programmatic `@register_model` registration remains supported for
classes you import yourself.

## Accounts and the CLI

Keys for a dynamic alias are managed like any other provider's. In Python,
pass the same catalog to `Accounts` so it knows the alias's account namespace
and key variable; keys are still resolved from the environment or account
store at request time, never stored in the catalog:

```python
from msgflux.accounts import Accounts

accounts = Accounts(provider_catalog=catalog)
accounts.set_api_key("acme", "your-acme-key", account_name="work")
```

This stores the key under the `acme` namespace of the `work` account, without
modifying `os.environ` or printing the key.

Config files can also drive the CLI. Loading declarations does not import or
construct factories, so login works before any model is configured:

```bash
msgflux accounts login acme --config PATH
```

This prompts for the Acme key (or follows the alias's configured auth) and
stores it under the `acme` provider name, reading `PATH` only for the
`[providers.acme]` declaration.

## Restoring serialized models

A serialized dynamic model records its provider definition metadata. Restore
it with the same trusted catalog:

```python
state = model.serialize()
restored = mf.Model.from_serialized(
    provider=state["provider"],
    model_type=state["model_type"],
    state=state["state"],
    provider_catalog=catalog,
)
```

This recreates the model only if `catalog` still declares `acme` compatibly.
Restoration never imports a factory path based solely on checkpoint contents:
a missing or changed declaration fails instead of executing checkpoint-supplied
code. Always supply a catalog you trust, not one derived from the checkpoint
itself.

## Native authentication for custom classes

An alias using `openai-codex` inherits its OAuth flow. The CLI stores tokens in
the alias's own provider namespace. In Python, configured OAuth aliases use
that managed namespace by default; an explicit `credential_resolver` or
`auth_file` selects the caller's authentication instead. The built-in
`openai-codex` provider retains its existing authentication behavior.

For API-key providers with custom headers, subclass
`APIKeyCredentialResolver` so environment-based and managed-account requests
format the key consistently:

```python
from msgflux.models import APIKeyCredentialResolver


class AcmeCredentials(APIKeyCredentialResolver):
    header_name = "x-acme-key"
    value_prefix = ""
```

Assign `credential_resolver = AcmeCredentials()` on the provider class.
Its normal `resolve(owner)` reads the key at request time; the account resolver
uses `resolve_api_key(value)` to format a managed key with the same header.

`Model.providers(provider_catalog=catalog)` lists built-in and configured
provider names without importing configured factories. Model IDs remain
provider-defined. Config-derived aliases reuse their base provider's model
profile lookup, while telemetry and reasoning history retain the alias identity.
