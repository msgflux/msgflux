---
hide:
  - navigation
  - toc
---

# Dependency Management

## Models

### Chat Completion

| Provider    | Dependency        | Auth Env             |
|-------------|-------------------|----------------------|
| Cerebras    | `msgflux[openai]` | `CEREBRAS_API_KEY`   |
| Exa         | `msgflux[openai]` | `EXA_API_KEY`        |
| Groq        | `msgflux[openai]` | `GROQ_API_KEY`       |
| Ollama      | `msgflux[openai]` |                      |
| OpenAI      | `msgflux[openai]` | `OPENAI_API_KEY`     |
| OpenRouter  | `msgflux[openai]` | `OPENROUTER_API_KEY` |
| SambaNova   | `msgflux[openai]` | `SAMBANOVA_API_KEY`  |
| Together    | `msgflux[openai]` | `TOGETHER_API_KEY`   |
| vLLM        | `msgflux[openai]` |                      |

### Image Embedder

| Provider    | Dependency       | Auth Env           |
|-------------|------------------|--------------------|
| JinaAI      | `msgflux[httpx]` | `JINAAI_API_KEY`   |

### Text To Image

| Provider    | Dependency        | Auth Env           |
|-------------|-------------------|--------------------|
| OpenAI      | `msgflux[openai]` | `OPENAI_API_KEY`   |

### Image Text To Image

| Provider    | Dependency        | Auth Env           |
|-------------|-------------------|--------------------|
| OpenAI      | `msgflux[openai]` | `OPENAI_API_KEY`   |

### Moderation

| Provider    | Dependency        | Auth Env           |
|-------------|-------------------|--------------------|
| OpenAI      | `msgflux[openai]` | `OPENAI_API_KEY`   |

### Speech To Text

| Provider    | Dependency        | Auth Env           |
|-------------|-------------------|--------------------|
| OpenAI      | `msgflux[openai]` | `OPENAI_API_KEY`   |
| vLLM        | `msgflux[openai]` |                    |

### Text Classifier

| Provider    | Dependency        | Auth Env           |
|-------------|-------------------|--------------------|
| vLLM        | `msgflux[openai]` |                    |

### Text Embedder

| Provider    | Dependency        | Auth Env           |
|-------------|-------------------|--------------------|
| JinaAI      | `msgflux[httpx]`  | `JINAAI_API_KEY`   |
| Ollama      | `msgflux[openai]` |                    |
| OpenAI      | `msgflux[openai]` | `OPENAI_API_KEY`   |
| Together    | `msgflux[openai]` | `TOGETHER_API_KEY` |
| vLLM        | `msgflux[openai]` |                    |

### Text Reranker

| Provider    | Dependency        | Auth Env           |
|-------------|-------------------|--------------------|
| JinaAI      | `msgflux[httpx]`  | `JINAAI_API_KEY`   |
| vLLM        | `msgflux[openai]` |                    |

### Text To Speech

| Provider    | Dependency        | Auth Env           |
|-------------|-------------------|--------------------|
| OpenAI      | `msgflux[openai]` | `OPENAI_API_KEY`   |
| Together    | `msgflux[openai]` | `TOGETHER_API_KEY` |

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

| Provider    | Dependency    |
|-------------|---------------|
| Wikipedia   | `wikipedia`   |
