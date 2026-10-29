# Web Fetch Retrievers

## ✦₊⁺ Overview

Web fetch retrievers retrieve page content for URLs you already know. Use them when the user supplies a link or asks about a specific page; use [Web Search](web.md) when you need to discover URLs first.

Create one with `mf.Retriever.web_fetch(provider)`. The resulting retriever type is `web_fetch`, and every provider returns the same response shape:

```python
response.response_type  # "web_fetch"
response.data            # one entry per input URL, in input order
response.data[0].data.url
response.data[0].data.content
```

All web fetch providers use native `httpx2` clients. The default chain needs no credentials. Optional keys or tokens are read from the environment at request time and are never stored on retriever instances or in serialized state.

### Default chain

`mf.Retriever.web_fetch("fallback")` tries providers in order for each URL:

1. `markdown.new` (keyless)
2. Firecrawl scrape (keyless by default)

A provider is skipped when it raises an error or returns empty content. If both
fail, the retriever raises `RuntimeError` with both provider errors. This is the
chain used by the built-in `WebFetchTool`.

```python
import msgflux as mf

retriever = mf.Retriever.web_fetch("fallback")
try:
    response = retriever("https://example.com/article")
    print(response.data[0].data.content)
finally:
    retriever.close()
```

### Response and errors

Web fetch reports provider failures instead of converting them into empty
results. HTTP, quota, invalid-response, and empty-content failures raise
`RuntimeError` (or `ValueError` for invalid URLs and missing Cloudflare
configuration). The fallback retriever reports each child failure in its final
error message.

URLs without a scheme use `https://`. Only `http` and `https` URLs are allowed;
other schemes are rejected.

### Client lifecycle

Each retriever owns reusable synchronous and asynchronous clients:

- call `close()` after synchronous use;
- call `await aclose()` when the async host is done; it also closes the
  synchronous client.

Serialization keeps constructor configuration but excludes runtime clients.
Restore with `mf.Retriever.from_serialized(...)` and the clients are rebuilt from
the saved settings.

---

## 1. **Fallback Fetch**

`fallback` composes ordered fetch providers. With no arguments it uses the
markdown.new → Firecrawl chain described above. It owns its child retrievers,
including injected instances, and closes them when the chain is closed.
Injected instances are reused; serialization restores separate runtime clients.

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `retrievers` | `None` | Ordered provider names, retriever instances, or serialized retriever specs. `None` builds the default chain |
| `timeout` | `None` | Timeout passed to the default child retrievers |
| `web_parser` | `None` | markdown.new-compatible endpoint for the default chain |
| `default_headers` | `None` | Headers passed to the default markdown retriever |

### Examples

=== "Default"

    ```python
    import msgflux as mf

    retriever = mf.Retriever.web_fetch("fallback")
    response = retriever("https://example.com/page")

    print(response.data[0].data.url)
    print(response.data[0].data.content)

    retriever.close()
    ```

=== "Custom Order"

    ```python
    import msgflux as mf

    retriever = mf.Retriever.web_fetch(
        "fallback",
        retrievers=["markdown", "http"],
    )
    response = retriever("example.com/page")
    print(response.data[0].data.content)

    retriever.close()
    ```

=== "Custom Parser"

    ```python
    import msgflux as mf

    retriever = mf.Retriever.web_fetch(
        "fallback",
        web_parser="https://markdown.example/",
        default_headers={"X-Request-Source": "msgflux"},
    )
    response = retriever("https://example.com/page")
    print(response.data[0].data.content)

    retriever.close()
    ```

---

## 2. **markdown.new Fetch**

The `markdown` provider posts a URL to a markdown.new-compatible endpoint and
returns Markdown. It is keyless and is the primary provider in the default
chain.

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `endpoint` | `"https://markdown.new/"` | markdown.new-compatible endpoint URL |
| `method` | `"auto"` | Conversion method: `"auto"`, `"ai"`, or `"browser"` |
| `retain_images` | `False` | Ask markdown.new to keep image references |
| `default_headers` | `None` | Additional headers sent to the endpoint |
| `timeout` | `30.0` | Request timeout in seconds |

### Example

```python
import msgflux as mf

retriever = mf.Retriever.web_fetch(
    "markdown",
    method="browser",
    retain_images=True,
)
response = retriever("https://example.com/article")
print(response.data[0].data.content)

retriever.close()
```

markdown.new can rate-limit conversions. Those failures raise `RuntimeError` with
the HTTP status; the default fallback chain then tries
Firecrawl.

---

## 3. **Firecrawl Fetch**

The `firecrawl` fetch provider scrapes a page as Markdown through Firecrawl v2.
It is keyless by default and is the fallback provider in the default chain.

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `base_url` | `"https://api.firecrawl.dev/v2"` | Firecrawl v2 API root |
| `timeout` | `30.0` | Request timeout in seconds |

### Authentication and limits

- Without `FIRECRAWL_API_KEY`, requests are sent keyless and include no
  `Authorization` header.
- When `FIRECRAWL_API_KEY` is set, it is read at request time and sent as a
  Bearer token.
- Firecrawl applies its own keyless/account limits; msgFlux does not add a quota
  counter or an invented allowance.
- Rate-limit and quota failures are explicit `RuntimeError`s containing the HTTP
  status and Firecrawl's detail. They are not silently converted to empty
  results, and msgFlux does not purchase credits or change to a paid plan
  automatically.

### Examples

=== "Keyless"

    ```python
    import msgflux as mf

    retriever = mf.Retriever.web_fetch("firecrawl")
    response = retriever("https://example.com/article")
    print(response.data[0].data.content)

    retriever.close()
    ```

=== "Optional Key"

    ```python
    import msgflux as mf

    mf.set_envs(FIRECRAWL_API_KEY="...")

    retriever = mf.Retriever.web_fetch("firecrawl", timeout=20)
    response = retriever("https://example.com/article")
    print(response.data[0].data.content)

    retriever.close()
    ```

=== "Quota Error"

    ```python
    import msgflux as mf

    retriever = mf.Retriever.web_fetch("firecrawl")

    try:
        response = retriever("https://example.com/article")
    except RuntimeError as exc:
        # Includes the HTTP status and Firecrawl rate-limit/quota detail.
        print(exc)
    ```

---

## 4. **Direct HTTP Fetch**

The `http` provider fetches a URL directly and converts HTML responses to
readable text. Markdown responses are kept as-is.

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `default_headers` | `None` | Headers sent with every page request |
| `timeout` | `30.0` | Request timeout in seconds |

### Example

```python
import msgflux as mf

retriever = mf.Retriever.web_fetch(
    "http",
    default_headers={"User-Agent": "my-app/1.0"},
    timeout=15,
)
response = retriever("example.com/page")
print(response.data[0].data.url)      # https://example.com/page
print(response.data[0].data.content)

retriever.close()
```

This provider performs a normal outbound HTTP request. It does not render
JavaScript and does not bypass access controls; use markdown.new, Firecrawl, or
Cloudflare when a rendered browser page is required.

---

## 5. **Cloudflare Fetch**

The `cloudflare` provider uses Cloudflare Browser Run to convert a page to
Markdown. It is opt-in and always requires a Cloudflare account ID and Browser
Run API token.

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `account_id` | `None` | Cloudflare account ID; falls back to `account_id_env` at request time |
| `account_id_env` | `"CLOUDFLARE_ACCOUNT_ID"` | Environment variable used when `account_id` is not passed |
| `api_token_env` | `"CLOUDFLARE_API_TOKEN"` | Environment variable holding the Browser Run API token |
| `base_url` | `"https://api.cloudflare.com/client/v4"` | Cloudflare API root |
| `timeout` | `30.0` | Request timeout in seconds |

The token is read on every request and is never stored on the retriever instance.
Selecting `cloudflare`, including with a custom `base_url`, requires credentials;
the default fallback chain does not.

### Example

```python
import msgflux as mf

mf.set_envs(
    CLOUDFLARE_ACCOUNT_ID="...",
    CLOUDFLARE_API_TOKEN="...",
)

retriever = mf.Retriever.web_fetch(
    "cloudflare",
    timeout=30,
)
response = retriever("https://example.com/article")
print(response.data[0].data.content)

retriever.close()
```

You can also pass the account ID directly and use custom environment names:

```python
import msgflux as mf

mf.set_envs(MY_CF_TOKEN="...")

retriever = mf.Retriever.web_fetch(
    "cloudflare",
    account_id="...",
    api_token_env="MY_CF_TOKEN",
    base_url="https://api.cloudflare.com/client/v4",
)
response = retriever("https://example.com/article")
print(response.data[0].data.content)

retriever.close()
```

---

## 6. **Linkup Fetch**

The `linkup` provider fetches a known URL through Linkup's `/v1/fetch` endpoint.
It uses the same `LINKUP_API_KEY` as Linkup Search, read at request time. Its
native `httpx2` implementation requires no Linkup SDK.

| Parameter | Default | Description |
| --- | --- | --- |
| `mode` | `"standard"` | Linkup fetch strategy (`standard` or `pro`). |
| `render_js` | `False` | Render JavaScript before extracting content. |
| `include_raw_content` | `False` | Request extra raw content from the provider. |
| `extract_images` | `False` | Request image extraction from the provider. |
| `base_url` | `"https://api.linkup.so"` | Linkup API root. |
| `timeout` | `30.0` | Request timeout in seconds. |

```python
import msgflux as mf

retriever = mf.Retriever.web_fetch("linkup", render_js=True)
try:
    response = retriever("https://docs.python.org/3/")
    print(response.data[0].data.content)
finally:
    retriever.close()
```

This example renders a page and returns its Markdown using the shared
`web_fetch` response shape. The standard result exposes the URL and Markdown;
additional raw-content/image response fields are not included in that shape.
Use `await retriever.acall(...)` and `await retriever.aclose()` in async hosts.

---

## 7. **WebFetchTool**

`WebFetchTool` wraps the default fallback chain and returns only the fetched
content string. Use it directly or inject a configured retriever.

The tool's default parser endpoint is `https://markdown.new/`, and its default
timeout is 30 seconds. An explicit `timeout=` overrides that default.

### Examples

=== "Default Tool"

    ```python
    from msgflux.tools.builtin import WebFetchTool

    tool = WebFetchTool()
    print(tool("https://example.com/article"))
    tool.close()
    ```

=== "Injected Retriever"

    ```python
    import msgflux as mf
    from msgflux.tools.builtin import WebFetchTool

    retriever = mf.Retriever.web_fetch(
        "http",
        timeout=20,
        default_headers={"User-Agent": "my-app/1.0"},
    )
    tool = WebFetchTool(retriever=retriever)

    print(tool("https://example.com/article"))
    tool.close()
    ```

=== "Async"

    ```python
    import msgflux as mf
    from msgflux.tools.builtin import WebFetchTool

    retriever = mf.Retriever.web_fetch("markdown", timeout=20)
    tool = WebFetchTool(retriever=retriever)

    content = await tool.acall("https://example.com/article")
    print(content)

    await tool.aclose()
    ```

---

## 8. **Batch, Async, and Serialization**

### Batch and async

Pass a list to fetch multiple URLs. Results keep input order.

```python
import msgflux as mf

retriever = mf.Retriever.web_fetch("fallback")

response = await retriever.acall([
    "https://example.com/one",
    "https://example.com/two",
])

for entry in response.data:
    print(entry.data.url)
    print(entry.data.content)

await retriever.aclose()
```

### Serialization

Serialize configuration, then restore a new retriever. Runtime clients and
credentials are not serialized.

```python
import msgflux as mf

retriever = mf.Retriever.web_fetch(
    "markdown",
    method="ai",
    retain_images=True,
    timeout=15,
)

serialized = retriever.serialize()
restored = mf.Retriever.from_serialized(
    provider=serialized["provider"],
    retriever_type=serialized["retriever_type"],
    params=serialized["state"],
)

print(restored.method)         # ai
print(restored.retain_images)  # True
print(restored.timeout)        # 15

response = restored("https://example.com/article")
print(response.data[0].data.content)

retriever.close()
restored.close()
```
