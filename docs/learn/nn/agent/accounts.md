# Managed Model Accounts

Use `Accounts` to keep provider credentials in named local bundles. Each
account selects one credential for each provider. The default account is named
`default`; other names can represent distinct users or environments.

## Store an API Key

`set_api_key()` accepts the provider name and key value. It writes the key to
the account store under the provider's conventional environment variable name.
The operation does not change `os.environ` or print the key.

```python
from msgflux.accounts import Accounts

accounts = Accounts()
accounts.set_api_key("openai", "sk-your-key", account_name="work")
print(accounts.info("work"))
```

This example creates the `work` account, stores its OpenAI API key, and prints
only the configured provider names and status. Account names are local labels;
they do not cause credentials to fall back to another account.

## Sign in to Codex

The managed `openai-codex` OAuth adapter delegates the sign-in flow to an
installed Codex CLI. It runs the CLI in a temporary account-specific
`CODEX_HOME`, forces file credential storage, validates the resulting Codex
credential document, and publishes it only after login succeeds. An interrupted
or failed login before publication leaves the account's previous grant in place.
Publication is atomic and serialized with refresh. If cancellation arrives after
publication has started, the write is allowed to finish before cancellation is
reported, so the newly validated grant may be installed.

```python
import asyncio

from msgflux.accounts import Accounts


async def main() -> None:
    accounts = Accounts()
    await accounts.login("openai-codex", account_name="work")
    print(accounts.info("work"))


asyncio.run(main())
```

The example opens Codex CLI's normal interactive login for the `work` account.
The returned credential record redacts token values in its representation, and
`info()` reports only whether each provider is configured.

Codex CLI also recognizes environment credentials and workload identity. Managed
login removes inherited `OPENAI_API_KEY`, `CODEX_API_KEY`,
`CODEX_ACCESS_TOKEN`, `OPENAI_FEDERATION_RULE_ID`, and
`OPENAI_IDENTITY_TOKEN_FILE` from the child process so those sources cannot
silently choose a different identity. It sets `CODEX_HOME` only in that child.
The optional `OPENAI_WORKLOAD_IDENTITY_CONTEXT` is retained because Codex
documents it as audit attribution metadata, not an authentication setting.
These names follow the [Codex environment variable
reference](https://learn.chatgpt.com/docs/config-file/environment-variables)
and [Codex workload identity guide](https://developers.openai.com/api/docs/guides/workload-identity-federation).

## Inspect Accounts

`list()` returns account names. `info(account_name)` returns a mapping from
configured provider names to non-secret status fields. It never returns keys,
access tokens, refresh tokens, or OAuth payloads.

```python
from msgflux.accounts import Accounts

accounts = Accounts()
for account_name in accounts.list():
    print(account_name, accounts.info(account_name))
```

This example inventories configured account names and providers without loading
credential values into application output.

Credentials are stored below `~/.msgflux/accounts/` by default, with private
directory permissions and atomic file replacement. Pass `Accounts(root=...)` to
select a different storage root. Keep that directory private and out of source
control and build artifacts.

## Use Credentials From a Model

`Accounts.resolver(provider)` creates a request-time credential resolver for a
model. A fixed `account_name` is useful for a standalone Agent; during a managed
service run, the resolver uses the account selected when the run is admitted.
The model keeps the resolver, not the key or OAuth tokens.

```python
import msgflux as mf
import msgflux.nn as nn
from msgflux.accounts import Accounts

accounts = Accounts()
model = mf.Model.chat_completion(
    "openai/gpt-4.1-mini",
    credential_resolver=accounts.resolver("openai", account_name="work"),
)
agent = nn.Agent("assistant", model)
answer = agent("Summarize the latest deployment report.")
```

This example creates a model that reads the `work` account's OpenAI key when a
request is sent. For an Agent managed by `AgentService`, attach a resolver
without a fixed `account_name`; the service's immutable execution scope then
selects the account for each run.

## Use the Accounts CLI

The `accounts login` command accepts a hidden prompt by default. `--stdin`
reads one line from standard input, and `--from-env` imports the provider's
conventional key environment variable exactly as supplied. The latter two
options apply only to API-key providers. Set `--accounts-dir` to use a storage
root other than `~/.msgflux/accounts/`.

```bash
msgflux accounts login openai \
  --account-name work \
  --accounts-dir ./private-accounts \
  --json
```

This example prompts without echoing the API key and prints a JSON status record
that contains the account and provider names, but no credential values. Use
`--from-env` only when importing the environment value is intentional.

To select that account for a headless coding run, pass `--account-name` to the
normal msgFlux CLI:

```bash
msgflux --prompt "Summarize the deployment report." \
  --account-name work \
  --json
```

For the built-in coding harness, msgFlux connects the model to the configured
account store and resolves the selected account when the request runs. The
service default comes from its `[defaults].account` setting; this option
overrides that default for the admitted run. The selected account remains fixed
when the run is resumed. If you use a custom `--accounts-dir` for login, set the
same path as `accounts_dir` in the local service configuration's `[defaults]`
table so the coding harness reads the credentials you just stored.
