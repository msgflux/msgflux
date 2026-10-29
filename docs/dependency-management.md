---
hide:
  - navigation
  - toc
---

# Dependency Management

## Workspace tools

Install `msgflux[coding]` to use `GlobTool` and `GrepTool`.
`pathspec` implements nested `.gitignore` matching and `regex` supplies timed
regular-expression searches. These optional packages are loaded only by the
query tools. `LsTool`, `ReadFileTool`, `WriteTool`, `EditTool` and `DeleteTool`
need no extra dependency. All filesystem access still goes through the live
authorized workspace; installing the extra does not grant filesystem access.

## Models

### Chat Completion

| Provider     | Dependency | Auth Env               |
|--------------|------------|------------------------|
| Anthropic    | `msgflux`  | `ANTHROPIC_API_KEY`    |
| Baseten      | `msgflux`  | `BASETEN_API_KEY`      |
| Brave        | `msgflux`  | `BRAVE_SEARCH_API_KEY` |
| Cerebras     | `msgflux`  | `CEREBRAS_API_KEY`     |
| Fireworks    | `msgflux`  | `FIREWORKS_API_KEY`    |
| Gemini       | `msgflux`  | `GEMINI_API_KEY`       |
| Groq         | `msgflux`  | `GROQ_API_KEY`         |
| Ollama       | `msgflux`  |                        |
| NVIDIA       | `msgflux`  | `NVIDIA_API_KEY`       |
| OpenAI       | `msgflux`  | `OPENAI_API_KEY`       |
| OpenCode Go  | `msgflux`  | `OPENCODE_GO_API_KEY`  |
| OpenCode Zen | `msgflux`  | `OPENCODE_ZEN_API_KEY` |
| OpenRouter   | `msgflux`  | `OPENROUTER_API_KEY`   |
| SambaNova    | `msgflux`  | `SAMBANOVA_API_KEY`    |
| Together     | `msgflux`  | `TOGETHER_API_KEY`     |
| vLLM         | `msgflux`  |                        |

OpenCode Go uses `OPENCODE_GO_API_KEY` as a Bearer credential for Chat
Completions, Responses, and native Anthropic Messages; it requires no SDK
dependency.

OpenCode Zen is OpenCode's paid, pay-as-you-go gateway (called OpenCode Console
in current OpenCode documentation). It uses `OPENCODE_ZEN_API_KEY` and the same
three explicit protocols, with `OPENCODE_ZEN_BASE_URL` overriding the
OpenAI-compatible base. The separate environment variables isolate msgFlux
configuration; they do not change the account or billing plan enforced by
OpenCode. See [OpenCode Zen](learn/models/chat_completion.md#opencode-zen) for
examples and protocol details.

### Image Embedder

| Provider    | Dependency | Auth Env         |
|-------------|------------|------------------|
| JinaAI      | `msgflux`  | `JINAAI_API_KEY` |

### Text To Image

| Provider    | Dependency | Auth Env             |
|-------------|------------|----------------------|
| ImageRouter | `msgflux`  | `IMAGEROUTER_API_KEY` |
| OpenAI      | `msgflux`  | `OPENAI_API_KEY`      |

### Image Text To Image

| Provider | Dependency | Auth Env         |
|----------|------------|------------------|
| OpenAI   | `msgflux`  | `OPENAI_API_KEY` |

### Moderation

| Provider    | Dependency        | Auth Env           |
|-------------|-------------------|--------------------|
| OpenAI      | `msgflux`         | `OPENAI_API_KEY`   |

### Speech To Text

| Provider | Dependency | Auth Env         |
|----------|------------|------------------|
| OpenAI   | `msgflux`  | `OPENAI_API_KEY` |
| vLLM     | `msgflux`  |                  |

### Text Classifier

| Provider    | Dependency        | Auth Env           |
|-------------|-------------------|--------------------|
| vLLM        | `msgflux`         |                    |

### Text Embedder

| Provider    | Dependency        | Auth Env           |
|-------------|-------------------|--------------------|
| JinaAI      | `msgflux`          | `JINAAI_API_KEY`   |
| Ollama      | `msgflux`         |                    |
| OpenAI      | `msgflux`         | `OPENAI_API_KEY`   |
| Together    | `msgflux`         | `TOGETHER_API_KEY` |
| vLLM        | `msgflux`         |                    |

### Text Reranker

| Provider    | Dependency        | Auth Env           |
|-------------|-------------------|--------------------|
| JinaAI      | `msgflux`          | `JINAAI_API_KEY`   |
| vLLM        | `msgflux`         |                    |

### Text To Speech

| Provider    | Dependency        | Auth Env           |
|-------------|-------------------|--------------------|
| OpenAI      | `msgflux`         | `OPENAI_API_KEY`   |
| Together    | `msgflux`         | `TOGETHER_API_KEY` |

## Parsers

### PDF

| Provider  | Dependency  |
|-----------|-------------|
| `pypdf`   | `pypdf`     |

### Word (.docx)

| Provider      | Dependency    |
|---------------|---------------|
| `python_docx` | `python-docx` |

### PowerPoint (.pptx)

| Provider      | Dependency    |
|---------------|---------------|
| `python_pptx` | `python-pptx` |

### Excel (.xlsx)

| Provider  | Dependency |
|-----------|------------|
| `openpyxl` | `openpyxl` |

### HTML

| Provider        | Dependency       |
|-----------------|------------------|
| `beautifulsoup` | `beautifulsoup4` |

### CSV / TSV

| Provider | Dependency |
|----------|------------|
| `csv`    | built-in   |

### Email (.eml)

| Provider | Dependency |
|----------|------------|
| `email`  | built-in   |

## Retrievers

### Lexical

| Provider    | Dependency    |
|-------------|---------------|
| BM25        | built-in      |
| BM25s       | `bm25s`       |
| Rank BM25   | `rank-bm25s`  |

### Fuzzy

| Provider    | Dependency    |
|-------------|---------------|
| RapidFuzz   | `rapidfuzz`   |

### Web Fetch

All fetch providers use the base `httpx2` dependency and need no provider SDK.

| Provider | Auth Env |
| --- | --- |
| markdown.new (`markdown`) | None |
| Firecrawl | Optional `FIRECRAWL_API_KEY`; keyless mode is available |
| Linkup | `LINKUP_API_KEY` |
| Cloudflare | `CLOUDFLARE_ACCOUNT_ID`, `CLOUDFLARE_API_TOKEN` |
| Direct HTTP (`http`) | None |
| Ordered fallback (`fallback`) | Uses its configured child providers |

See [Web Fetch Retrievers](learn/retrievers/web_fetch.md) for defaults and examples.

### Web Search

HTTPX2 is a base msgFlux dependency; providers listing it below require no
additional HTTP client installation.

| Provider    | Installation                     | Auth Env                         |
|-------------|----------------------------------|-----------------------------------|
| Trafilatura | `httpx2`, `trafilatura`           |                                   |
| SearXNG     | `httpx2`                          |                                   |
| SerpApi     | `httpx2`                          | `SERPAPI_KEY`                     |
| Ceramic     | `httpx2`                          | `CERAMIC_API_KEY`                 |
| Firecrawl   | `httpx2`                          | `FIRECRAWL_API_KEY` (optional)    |
| Linkup      | `httpx2`                          | `LINKUP_API_KEY`                  |
| Tavily      | `httpx2`                          | `TAVILY_API_KEY`                  |
| Exa         | `httpx2`                          | `EXA_API_KEY`                     |
| Brave       | `httpx2`                          | `BRAVE_SEARCH_API_KEY`            |
| arXiv       | `httpx2`                          |                                   |
| Wikipedia   | `httpx2`                          |                                   |

Firecrawl search is keyless by default; `FIRECRAWL_API_KEY` is optional and is
read per request. arXiv parses Atom XML with Python's standard-library
`xml.etree.ElementTree`, so it needs no additional XML parser dependency.
