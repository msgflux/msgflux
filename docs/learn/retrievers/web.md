# Web Retrievers

## ✦₊⁺ Overview

Web retrievers query online sources at request time and return structured results through `mf.Retriever.web(...)`. The built-in `wikipedia` provider fetches article content with optional summaries and images.

---

## 1. **Wikipedia Search**

The `wikipedia` retriever fetches and returns Wikipedia article content at query time. Unlike lexical retrievers, it requires no pre-indexed corpus — it queries the Wikipedia API directly and returns structured results with title, content, and optionally images.

!!! info "Dependencies"
    Requires the `wikipedia` package: `pip install wikipedia`

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `language` | `"en"` | Wikipedia language code (`"pt"`, `"es"`, `"fr"`, …) |
| `summary` | `None` | Number of sentences to return — `None` returns the full article |
| `return_images` | `False` | Whether to include image URLs in results |
| `max_return_images` | `5` | Maximum number of image URLs per result |

### Examples

=== "Search"

    ```python
    import msgflux as mf

    retriever = mf.Retriever.web("wikipedia")
    response = retriever("machine learning", top_k=2)

    for result in response.data[0].results:
        print(result.data.title)
        print(result.data.content[:200])
    ```

=== "Summary"

    ```python
    import msgflux as mf

    retriever = mf.Retriever.web("wikipedia", summary=2)
    response = retriever("Eiffel Tower")

    print(response.data[0].results[0].data.content)
    # Eiffel Tower
    #
    # The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris.
    # It is named after the engineer Gustave Eiffel, whose company designed and built it.
    ```

=== "Images"

    ```python
    import msgflux as mf

    retriever = mf.Retriever.web(
        "wikipedia",
        return_images=True,
        max_return_images=3,
    )

    response = retriever("Colosseum")

    result = response.data[0].results[0]
    print(result.data.title)
    print(result.images)
    ```

=== "Languages"

    ```python
    import msgflux as mf

    queries = [
        ("pt", "inteligência artificial"),
        ("es", "aprendizaje automático"),
        ("fr", "réseau de neurones"),
    ]

    for language, query in queries:
        retriever = mf.Retriever.web("wikipedia", language=language, summary=3)
        response = retriever(query)
        print(response.data[0].results[0].data.content)
    ```

=== "Batch"

    ```python
    import msgflux as mf

    retriever = mf.Retriever.web("wikipedia", summary=2)

    queries = ["Python programming", "Rust programming language", "Go programming"]
    response = retriever(queries, top_k=1)

    for i, query in enumerate(queries):
        result = response.data[i].results[0]
        print(f"\n{query}: {result.data.title}")
        print(result.data.content)
    ```

=== "RAG"

    ```python
    import msgflux as mf

    retriever = mf.Retriever.web("wikipedia", summary=5)
    chat = mf.Model.chat_completion("openai/gpt-4.1-mini")

    def answer_with_wikipedia(question: str) -> str:
        response = retriever(question, top_k=2)

        context = "\n\n".join(
            result.data.content
            for result in response.data[0].results
        )

        return chat(messages=[{
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}",
        }]).consume()

    print(answer_with_wikipedia("How does the James Webb Space Telescope work?"))
    ```

=== "Async"

    ```python
    import msgflux as mf

    retriever = mf.Retriever.web("wikipedia", summary=3)

    queries = ["quantum computing", "photosynthesis", "black holes"]
    response = await retriever.acall(queries, top_k=1)

    for i, query in enumerate(queries):
        result = response.data[i].results[0]
        print(f"\n{query}: {result.data.title}")
    ```

---

## 2. **SerpApi Search**

The `serpapi` retriever queries SerpApi and returns structured search results from engines such as Google. Use it when you need general web, news, image, shopping, or localized search through SerpApi.

!!! info "Dependencies"
    Requires `httpx` and the `SERPAPI_KEY` env variable:
    `pip install httpx`

    For compatibility, `SERPAPI_API_KEY` and `SERP_API_KEY` are also accepted.
    Both synchronous and async calls use direct requests to
    `https://serpapi.com/search.json`.

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `engine` | `"google"` | Search engine to use, such as `"google"`, `"bing"`, or `"yahoo"` |
| `location` | `None` | Location for localized results, such as `"Austin,Texas"` |
| `gl` | `None` | Google country code, such as `"us"` or `"br"` |
| `hl` | `None` | Google UI language, such as `"en"` or `"pt"` |
| `safe` | `None` | Safe search mode, such as `"active"` or `"off"` |
| `tbm` | `None` | Search type, such as `"nws"` for news or `"isch"` for images |

### Examples

=== "Web"

    ```python
    import msgflux as mf

    mf.set_envs(SERPAPI_KEY="...")

    retriever = mf.Retriever.web("serpapi")
    response = retriever("latest Python release", top_k=3)

    for result in response.data[0].results:
        print(result.data.title)
        print(result.data.url)
        print(result.data.content)
    ```

=== "Localized"

    ```python
    import msgflux as mf

    retriever = mf.Retriever.web(
        "serpapi",
        location="Sao Paulo, Brazil",
        gl="br",
        hl="pt",
    )
    response = retriever("melhores frameworks Python", top_k=3)

    for result in response.data[0].results:
        print(result.data.title)
    ```

=== "News"

    ```python
    import msgflux as mf

    retriever = mf.Retriever.web("serpapi", tbm="nws", gl="us", hl="en")
    response = retriever("AI regulation", top_k=5)

    for result in response.data[0].results:
        print(result.data.title)
        print(result.data.date)
        print(result.data.url)
    ```

=== "Images"

    ```python
    import msgflux as mf

    retriever = mf.Retriever.web("serpapi", tbm="isch")
    response = retriever("James Webb Space Telescope", top_k=3)

    for result in response.data[0].results:
        print(result.data.title)
        print(result.images[0])
    ```

=== "Batch"

    ```python
    import msgflux as mf

    retriever = mf.Retriever.web("serpapi", engine="google")

    queries = ["Python packaging", "Rust async runtime"]
    response = retriever(queries, top_k=2)

    for i, query in enumerate(queries):
        print(f"\n{query}")
        for result in response.data[i].results:
            print(result.data.title)
    ```

=== "Async"

    ```python
    import msgflux as mf

    retriever = mf.Retriever.web("serpapi", gl="us", hl="en")

    response = await retriever.acall(["Python 3.14", "Django release"], top_k=2)

    for item in response.data:
        print(item.results[0].data.title)
    ```

---

## 3. **Brave Search**

The `brave` retriever queries Brave Search and can return web, news, or image results. Use it when you need search results from Brave with a single provider interface.

!!! info "Dependencies"
    Requires `brave-search-python-client` and the `BRAVE_SEARCH_API_KEY` env variable:
    `pip install brave-search-python-client`

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mode` | `"search"` | Search mode: `"search"`, `"news"`, or `"image"` |
| `return_images` | `False` | Whether to include thumbnail image URLs for web/news results |

### Examples

=== "Web"

    ```python
    import msgflux as mf

    mf.set_envs(BRAVE_SEARCH_API_KEY="...")

    retriever = mf.Retriever.web("brave")
    response = retriever("latest Python release", top_k=3)

    for result in response.data[0].results:
        print(result.data.title)
        print(result.data.url)
        print(result.data.content)
    ```

=== "Thumbnails"

    ```python
    import msgflux as mf

    retriever = mf.Retriever.web(
        "brave",
        mode="search",
        return_images=True,
    )
    response = retriever("Python tutorials", top_k=3)

    for result in response.data[0].results:
        print(result.data.title)
        print(result.images[0])
    ```

=== "News"

    ```python
    import msgflux as mf

    retriever = mf.Retriever.web("brave", mode="news")
    response = retriever("AI regulation", top_k=5)

    for result in response.data[0].results:
        print(result.data.title)
        print(result.data.date)
        print(result.data.url)
    ```

=== "Images"

    ```python
    import msgflux as mf

    retriever = mf.Retriever.web("brave", mode="image")
    response = retriever("James Webb Space Telescope", top_k=3)

    for result in response.data[0].results:
        print(result.data.title)
        print(result.images[0])
    ```

=== "Async"

    ```python
    import msgflux as mf

    retriever = mf.Retriever.web("brave", mode="search")

    response = await retriever.acall(["Python 3.14", "Django release"], top_k=2)

    for item in response.data:
        print(item.results[0].data.title)
    ```

---

## 4. **Tavily Search**

The `tavily` retriever queries Tavily and returns search results optimized for AI applications. It supports search depth, topic filters, time ranges, domain filters, generated answers, images, and raw page content.

!!! info "Dependencies"
    Requires `tavily-python` and the `TAVILY_API_KEY` env variable:
    `pip install tavily-python`

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `search_depth` | `"basic"` | Search depth: `"basic"` or `"advanced"` |
| `topic` | `"general"` | Topic category: `"general"`, `"news"`, or `"finance"` |
| `time_range` | `None` | Time range: `"day"`, `"week"`, `"month"`, `"year"` or `"d"`, `"w"`, `"m"`, `"y"` |
| `include_domains` | `None` | Domains to restrict search to |
| `exclude_domains` | `None` | Domains to exclude from search |
| `include_answer` | `False` | Whether Tavily should include an AI-generated answer |
| `include_images` | `False` | Whether to include image results |
| `include_raw_content` | `False` | Whether to include raw page content |

### Examples

=== "Web"

    ```python
    import msgflux as mf

    mf.set_envs(TAVILY_API_KEY="...")

    retriever = mf.Retriever.web("tavily")
    response = retriever("latest Python release", top_k=3)

    for result in response.data[0].results:
        print(result.data.title)
        print(result.data.url)
        print(result.data.content)
    ```

=== "Advanced"

    ```python
    import msgflux as mf

    retriever = mf.Retriever.web(
        "tavily",
        search_depth="advanced",
        topic="news",
        time_range="week",
    )
    response = retriever("latest AI news", top_k=5)

    for result in response.data[0].results:
        print(result.data.title)
        print(result.data.url)
    ```

=== "Raw Content"

    ```python
    import msgflux as mf

    retriever = mf.Retriever.web(
        "tavily",
        search_depth="advanced",
        include_raw_content=True,
    )

    response = retriever("Python packaging standards", top_k=2)

    for result in response.data[0].results:
        print(result.data.title)
        print(result.data.raw_content[:500])
    ```

=== "Filters"

    ```python
    import msgflux as mf

    retriever = mf.Retriever.web(
        "tavily",
        include_domains=["python.org", "pypi.org"],
        exclude_domains=["example.com"],
    )

    response = retriever("packaging metadata", top_k=3)

    for result in response.data[0].results:
        print(result.data.title)
        print(result.data.url)
    ```

=== "Async"

    ```python
    import msgflux as mf

    retriever = mf.Retriever.web("tavily", search_depth="advanced")

    response = await retriever.acall(["Python 3.14", "Django release"], top_k=2)

    for item in response.data:
        print(item.results[0].data.title)
    ```

---

## 5. **Linkup Search**

The `linkup` retriever queries Linkup and returns AI-oriented web results. It supports standard search, deeper agentic search, domain filters, image inclusion, and sourced answers.

!!! info "Dependencies"
    Requires `linkup-sdk` and the `LINKUP_API_KEY` env variable:
    `pip install linkup-sdk`

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `depth` | `"standard"` | Search depth: `"standard"` for faster search or `"deep"` for agentic search |
| `output_type` | `"searchResults"` | Output mode: `"searchResults"` or `"sourcedAnswer"` |
| `include_domains` | `None` | Domains to restrict search to |
| `exclude_domains` | `None` | Domains to exclude from search |
| `include_images` | `False` | Whether to ask Linkup to include images |

### Examples

=== "Web"

    ```python
    import msgflux as mf

    mf.set_envs(LINKUP_API_KEY="...")

    retriever = mf.Retriever.web("linkup")
    response = retriever("latest Python packaging changes", top_k=3)

    for result in response.data[0].results:
        print(result.data.title)
        print(result.data.url)
        print(result.data.content)
    ```

=== "Deep"

    ```python
    import msgflux as mf

    retriever = mf.Retriever.web(
        "linkup",
        depth="deep",
        include_domains=["python.org", "pypi.org"],
    )
    response = retriever("recent Python packaging changes", top_k=3)

    for result in response.data[0].results:
        print(result.data.title)
        print(result.data.url)
    ```

=== "Sourced Answer"

    ```python
    import msgflux as mf

    retriever = mf.Retriever.web(
        "linkup",
        depth="deep",
        output_type="sourcedAnswer",
    )

    response = retriever("What changed in Python packaging recently?", top_k=5)

    for source in response.data[0].results:
        print(source.data.title)
        print(source.data.url)
    ```

=== "Batch"

    ```python
    import msgflux as mf

    retriever = mf.Retriever.web("linkup", depth="standard")

    queries = ["Python packaging", "Rust async runtime"]
    response = retriever(queries, top_k=2)

    for i, query in enumerate(queries):
        print(f"\n{query}")
        for result in response.data[i].results:
            print(result.data.title)
    ```

=== "Async"

    ```python
    import msgflux as mf

    retriever = mf.Retriever.web("linkup", depth="deep")

    response = await retriever.acall(["Python 3.14", "Django release"], top_k=2)

    for item in response.data:
        print(item.results[0].data.title)
    ```

---

## 6. **Exa Search**

The `exa` retriever queries Exa for semantic web search results. It can return URLs only, or fetch page text together with each result for RAG and summarization workflows.

!!! info "Dependencies"
    Requires `exa-py` and the `EXA_API_KEY` env variable:
    `pip install exa-py`

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `search_type` | `"auto"` | Search type: `"auto"`, `"neural"`, `"fast"`, or `"deep"` |
| `include_domains` | `None` | Domains to restrict search to |
| `exclude_domains` | `None` | Domains to exclude from search |
| `start_published_date` | `None` | ISO date filter for results published after a date |
| `end_published_date` | `None` | ISO date filter for results published before a date |
| `include_text` | `True` | Whether to fetch page text with each result |
| `max_characters` | `None` | Maximum number of text characters returned per result |

### Examples

=== "Web"

    ```python
    import msgflux as mf

    mf.set_envs(EXA_API_KEY="...")

    retriever = mf.Retriever.web("exa", include_text=True)
    response = retriever("latest Python packaging changes", top_k=3)

    for result in response.data[0].results:
        print(result.data.title)
        print(result.data.url)
        print(result.data.content[:300])
    ```

=== "URL Only"

    ```python
    import msgflux as mf

    retriever = mf.Retriever.web("exa", include_text=False)
    response = retriever("Python web frameworks", top_k=5)

    for result in response.data[0].results:
        print(result.data.title)
        print(result.data.url)
    ```

=== "Filters"

    ```python
    import msgflux as mf

    retriever = mf.Retriever.web(
        "exa",
        include_domains=["python.org", "pypi.org"],
        start_published_date="2025-01-01",
        include_text=True,
        max_characters=2000,
    )

    response = retriever("packaging metadata standards", top_k=3)

    for result in response.data[0].results:
        print(result.data.title)
        print(result.data.url)
    ```

=== "Async"

    ```python
    import msgflux as mf

    retriever = mf.Retriever.web("exa", search_type="auto", include_text=True)

    response = await retriever.acall(["Python 3.14", "Django release"], top_k=2)

    for item in response.data:
        print(item.results[0].data.title)
    ```

---

## 7. **SearXNG Search**

The `searxng` retriever queries a local or self-hosted SearXNG instance and returns structured web results with title, content, URL, and optional image metadata. SearXNG is useful when you want free, private, local web search without adding a provider SDK or API key.

!!! info "Dependencies"
    Requires `httpx` and a running SearXNG instance with JSON output enabled:
    `pip install httpx`

    By default, msgFlux uses `http://localhost:8080`. Set `SEARXNG_BASE_URL`
    or pass `base_url` to point at another local/self-hosted instance.

    SearXNG's search API requires `q` and supports `format=json`. The SearXNG
    server must enable JSON in `search.formats`.

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `base_url` | `SEARXNG_BASE_URL` or `"http://localhost:8080"` | Base URL for the SearXNG instance |
| `categories` | `None` | Comma-separated active search categories, such as `"general,news"` |
| `engines` | `None` | Comma-separated active search engines, such as `"duckduckgo,wikipedia"` |
| `language` | `None` | Search language code |
| `time_range` | `None` | Time range filter: `"day"`, `"month"`, or `"year"` |
| `safesearch` | `None` | Safe search level: `0`, `1`, or `2` |
| `pageno` | `None` | Search page number |
| `timeout` | `30.0` | Request timeout in seconds |

### Local Docker

For local development, run SearXNG with JSON output enabled:

```yaml
# /tmp/msgflux-searxng/settings.yml
use_default_settings: true
server:
  secret_key: "change-me"
  bind_address: "0.0.0.0"
search:
  formats:
    - html
    - json
```

```bash
docker run --name msgflux-searxng -d \
  -p 8888:8080 \
  -v /tmp/msgflux-searxng:/etc/searxng:ro \
  docker.io/searxng/searxng:latest
```

Then use `base_url="http://localhost:8888"` or:

```bash
export SEARXNG_BASE_URL="http://localhost:8888"
```

### Examples

=== "Search"

    ```python
    import msgflux as mf

    retriever = mf.Retriever.web(
        "searxng",
        base_url="http://localhost:8888",
    )
    response = retriever("latest Python release", top_k=3)

    for result in response.data[0].results:
        print(result.data.title)
        print(result.data.url)
        print(result.data.content)
    ```

=== "Filters"

    ```python
    import msgflux as mf

    retriever = mf.Retriever.web(
        "searxng",
        base_url="http://localhost:8888",
        categories="general,news",
        engines="duckduckgo,wikipedia",
        language="en",
        time_range="month",
        safesearch=1,
    )

    response = retriever("Python packaging standards", top_k=5)

    for result in response.data[0].results:
        print(result.data.title)
        print(result.data.engine)
    ```

=== "Async"

    ```python
    import msgflux as mf

    retriever = mf.Retriever.web("searxng", base_url="http://localhost:8888")

    response = await retriever.acall(["Python 3.14", "Django release"], top_k=2)

    for item in response.data:
        print(item.results[0].data.title)
    ```

---

## 8. **arXiv Search**

The `arxiv` retriever searches arXiv papers and returns structured academic metadata such as title, summary, authors, publication dates, categories, and PDF URLs.

!!! info "Dependencies"
    Requires the `arxiv` package: `pip install arxiv`

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_results` | `10` | Maximum number of arXiv results fetched per query |
| `sort_by` | `"relevance"` | Sort criterion: `"relevance"`, `"lastUpdatedDate"`, or `"submittedDate"` |
| `sort_order` | `"descending"` | Sort order: `"ascending"` or `"descending"` |

### Examples

=== "Search"

    ```python
    import msgflux as mf

    retriever = mf.Retriever.web("arxiv")
    response = retriever("retrieval augmented generation", top_k=3)

    for result in response.data[0].results:
        print(result.data.title)
        print(result.data.authors)
        print(result.data.pdf_url)
        print(result.data.summary[:300])
    ```

=== "Recent"

    ```python
    import msgflux as mf

    retriever = mf.Retriever.web(
        "arxiv",
        max_results=5,
        sort_by="submittedDate",
        sort_order="descending",
    )

    response = retriever("large language model agents", top_k=5)

    for result in response.data[0].results:
        print(result.data.published)
        print(result.data.title)
        print(result.data.pdf_url)
    ```

=== "Batch"

    ```python
    import msgflux as mf

    retriever = mf.Retriever.web("arxiv", sort_by="relevance")

    queries = ["graph neural networks", "diffusion models"]
    response = retriever(queries, top_k=2)

    for i, query in enumerate(queries):
        print(f"\n{query}")
        for result in response.data[i].results:
            print(result.data.title)
    ```

=== "Async"

    ```python
    import msgflux as mf

    retriever = mf.Retriever.web("arxiv", sort_by="submittedDate")

    response = await retriever.acall(["RAG evaluation", "agent benchmarks"], top_k=2)

    for item in response.data:
        print(item.results[0].data.title)
    ```
