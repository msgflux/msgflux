---
hide:
  - navigation
  - toc
---

# Dependency Management

## Models

### Chat Completion

| Provider    | Dependency | Auth Env               |
|-------------|------------|------------------------|
| Brave       | `msgflux`  | `BRAVE_SEARCH_API_KEY` |
| Cerebras    | `msgflux`  | `CEREBRAS_API_KEY`     |
| Groq        | `msgflux`  | `GROQ_API_KEY`         |
| Ollama      | `msgflux`  |                        |
| OpenAI      | `msgflux`  | `OPENAI_API_KEY`       |
| OpenRouter  | `msgflux`  | `OPENROUTER_API_KEY`   |
| SambaNova   | `msgflux`  | `SAMBANOVA_API_KEY`    |
| Together    | `msgflux`  | `TOGETHER_API_KEY`     |
| vLLM        | `msgflux`  |                        |

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

### Web Search

HTTPX2 is a base msgFlux dependency; providers listing it below require no
additional HTTP client installation.

| Provider    | Installation                     | Auth Env                         |
|-------------|----------------------------------|-----------------------------------|
| Trafilatura | `httpx2`, `trafilatura`           |                                   |
| SearXNG     | `httpx2`                          |                                   |
| SerpApi     | `httpx2`                          | `SERPAPI_KEY`                     |
| Ceramic     | `httpx2`                          | `CERAMIC_API_KEY`                 |
| Linkup      | `linkup-sdk`                     | `LINKUP_API_KEY`                  |
| Tavily      | `tavily-python`                  | `TAVILY_API_KEY`                  |
| Exa         | `exa-py`                         | `EXA_API_KEY`                     |
| Brave       | `brave-search-python-client`     | `BRAVE_SEARCH_API_KEY`            |
| arXiv       | `arxiv`                          |                                   |
| Wikipedia   | `wikipedia`                      |                                   |
