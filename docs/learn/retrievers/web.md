# Web Retrievers

## ✦₊⁺ Overview

Web retrievers query online sources at request time and return structured results through `mf.Retriever.web_search(...)`. Providers use native `httpx2` clients: install no provider SDK, read API keys from the environment at request time, and keep credentials out of serialized state. The default `WebSearchTool` engine is `retriever/firecrawl`, which works without a key.

To fetch a known URL instead of searching for one, see [Web Fetch](web_fetch.md).

Use `Retriever.web_search(...)` for queries and `Retriever.web_fetch(...)` for known URLs. `Retriever.web(...)` remains a compatibility alias for `web_search`; the marker class is `WebSearchRetriever` and its registry/serialized type is `web_search`. Existing serialized records with type `web` can still be restored through `Retriever.from_serialized(...)`. Custom search retrievers should inherit from `WebSearchRetriever` instead of the former `WebRetriever`.

### Client lifecycle

Each retriever owns reusable synchronous and asynchronous `httpx2` clients. Call `close()` after synchronous use, or `await aclose()` when the async host is done; `aclose()` also closes the synchronous client. Runtime clients are excluded from serialization, so a restored retriever rebuilds them from its saved `timeout` and `base_url`.

```python
import msgflux as mf

retriever = mf.Retriever.web_search("exa")
try:
    response = retriever("latest Python release", top_k=3)
    print(response.data[0].results[0].data.title)
finally:
    retriever.close()
```

For async code, await `retriever.aclose()` in the same cleanup path.

!!! note "Migrating from provider SDK clients"
    These retrievers now use native `httpx2` requests. Recreate previously
    serialized SDK-backed client instances from your constructor configuration
    before serializing them again; old SDK state does not contain the new HTTP
    client settings. The public search result fields are preserved, while SDK
    enum attributes such as arXiv sort fields are now strings. Provider-specific
    SDK retries and request pacing are not reproduced by the HTTP client;
    schedule bulk arXiv requests according to the provider's API usage policy.


---

## 1. **Wikipedia Search**

The `wikipedia` retriever fetches and returns Wikipedia article content at query time. Unlike lexical retrievers, it requires no pre-indexed corpus — it queries the Wikipedia API directly and returns structured results with title, content, and optionally images.

!!! info "Dependencies"
    HTTPX2 is installed with msgFlux. No Wikipedia SDK is required.

    Both synchronous and async calls use direct requests to
    `https://{language}.wikipedia.org/w/api.php`.

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `language` | `"en"` | Wikipedia language code (`"pt"`, `"es"`, `"fr"`, …) |
| `summary` | `None` | Number of sentences to return — `None` returns the full article |
| `return_images` | `False` | Whether to include image URLs in results |
| `max_return_images` | `5` | Maximum number of image URLs per result |
| `timeout` | `30.0` | Request timeout in seconds |

### Examples

=== "Search"

    ```python
    import msgflux as mf

    retriever = mf.Retriever.web_search("wikipedia")
    response = retriever("machine learning", top_k=2)

    for result in response.data[0].results:
        print(result.data.title)
        print(result.data.content[:200])
    ```

=== "Summary"

    ```python
    import msgflux as mf

    retriever = mf.Retriever.web_search("wikipedia", summary=2)
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

    retriever = mf.Retriever.web_search(
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
        retriever = mf.Retriever.web_search("wikipedia", language=language, summary=3)
        response = retriever(query)
        print(response.data[0].results[0].data.content)
    ```

=== "Batch"

    ```python
    import msgflux as mf

    retriever = mf.Retriever.web_search("wikipedia", summary=2)

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

    retriever = mf.Retriever.web_search("wikipedia", summary=5)
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

    retriever = mf.Retriever.web_search("wikipedia", summary=3)

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
    HTTPX2 is installed with msgFlux. Set the `SERPAPI_KEY` env variable.

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

    retriever = mf.Retriever.web_search("serpapi")
    response = retriever("latest Python release", top_k=3)

    for result in response.data[0].results:
        print(result.data.title)
        print(result.data.url)
        print(result.data.content)
    ```

=== "Localized"

    ```python
    import msgflux as mf

    retriever = mf.Retriever.web_search(
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

    retriever = mf.Retriever.web_search("serpapi", tbm="nws", gl="us", hl="en")
    response = retriever("AI regulation", top_k=5)

    for result in response.data[0].results:
        print(result.data.title)
        print(result.data.date)
        print(result.data.url)
    ```

=== "Images"

    ```python
    import msgflux as mf

    retriever = mf.Retriever.web_search("serpapi", tbm="isch")
    response = retriever("James Webb Space Telescope", top_k=3)

    for result in response.data[0].results:
        print(result.data.title)
        print(result.images[0])
    ```

=== "Batch"

    ```python
    import msgflux as mf

    retriever = mf.Retriever.web_search("serpapi", engine="google")

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

    retriever = mf.Retriever.web_search("serpapi", gl="us", hl="en")

    response = await retriever.acall(["Python 3.14", "Django release"], top_k=2)

    for item in response.data:
        print(item.results[0].data.title)
    ```

---

## 3. **Brave Search**

The `brave` retriever queries Brave Search and can return web, news, or image results. Use it when you need search results from Brave with a single provider interface.

!!! info "Dependencies"
    HTTPX2 is installed with msgFlux. No Brave SDK is required. Set the
    `BRAVE_SEARCH_API_KEY` env variable; the key is read per request.

    Synchronous and async calls use direct requests to
    `https://api.search.brave.com/res/v1/{web,news,images}/search`.

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mode` | `"search"` | Search mode: `"search"`, `"news"`, or `"image"` |
| `return_images` | `False` | Whether to include thumbnail image URLs for web/news results |
| `base_url` | `"https://api.search.brave.com"` | Brave Search API base URL |
| `timeout` | `30.0` | Request timeout in seconds |

### Examples

=== "Web"

    ```python
    import msgflux as mf

    mf.set_envs(BRAVE_SEARCH_API_KEY="...")

    retriever = mf.Retriever.web_search("brave")
    response = retriever("latest Python release", top_k=3)

    for result in response.data[0].results:
        print(result.data.title)
        print(result.data.url)
        print(result.data.content)
    ```

=== "Thumbnails"

    ```python
    import msgflux as mf

    retriever = mf.Retriever.web_search(
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

    retriever = mf.Retriever.web_search("brave", mode="news")
    response = retriever("AI regulation", top_k=5)

    for result in response.data[0].results:
        print(result.data.title)
        print(result.data.date)
        print(result.data.url)
    ```

=== "Images"

    ```python
    import msgflux as mf

    retriever = mf.Retriever.web_search("brave", mode="image")
    response = retriever("James Webb Space Telescope", top_k=3)

    for result in response.data[0].results:
        print(result.data.title)
        print(result.images[0])
    ```

=== "Async"

    ```python
    import msgflux as mf

    retriever = mf.Retriever.web_search("brave", mode="search")

    response = await retriever.acall(["Python 3.14", "Django release"], top_k=2)

    for item in response.data:
        print(item.results[0].data.title)
    ```

---

## 4. **Tavily Search**

The `tavily` retriever queries Tavily and returns search results optimized for AI applications. It supports search depth, topic filters, time ranges, domain filters, generated answers, images, and raw page content.

!!! info "Dependencies"
    HTTPX2 is installed with msgFlux. No Tavily SDK is required. Set the
    `TAVILY_API_KEY` env variable; the key is read per request.

    Synchronous and async calls use direct requests to
    `https://api.tavily.com/search`.

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
| `base_url` | `"https://api.tavily.com"` | Tavily API base URL |
| `timeout` | `30.0` | Request timeout in seconds |

### Examples

=== "Web"

    ```python
    import msgflux as mf

    mf.set_envs(TAVILY_API_KEY="...")

    retriever = mf.Retriever.web_search("tavily")
    response = retriever("latest Python release", top_k=3)

    for result in response.data[0].results:
        print(result.data.title)
        print(result.data.url)
        print(result.data.content)
    ```

=== "Advanced"

    ```python
    import msgflux as mf

    retriever = mf.Retriever.web_search(
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

    retriever = mf.Retriever.web_search(
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

    retriever = mf.Retriever.web_search(
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

    retriever = mf.Retriever.web_search("tavily", search_depth="advanced")

    response = await retriever.acall(["Python 3.14", "Django release"], top_k=2)

    for item in response.data:
        print(item.results[0].data.title)
    ```

---

## 5. **Linkup Search**

The `linkup` retriever queries Linkup and returns AI-oriented web results. It supports standard search, deeper agentic search, domain filters, image inclusion, and sourced answers.

!!! info "Dependencies"
    HTTPX2 is installed with msgFlux. No Linkup SDK is required. Set the
    `LINKUP_API_KEY` env variable; the key is read per request.

    Synchronous and async calls use direct requests to
    `https://api.linkup.so/v1/search`.

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `depth` | `"standard"` | Search depth: `"standard"` for faster search or `"deep"` for agentic search |
| `output_type` | `"searchResults"` | Output mode: `"searchResults"` or `"sourcedAnswer"` |
| `include_domains` | `None` | Domains to restrict search to |
| `exclude_domains` | `None` | Domains to exclude from search |
| `include_images` | `False` | Whether to ask Linkup to include images |
| `base_url` | `"https://api.linkup.so"` | Linkup API base URL |
| `timeout` | `30.0` | Request timeout in seconds |

### Examples

=== "Web"

    ```python
    import msgflux as mf

    mf.set_envs(LINKUP_API_KEY="...")

    retriever = mf.Retriever.web_search("linkup")
    response = retriever("latest Python packaging changes", top_k=3)

    for result in response.data[0].results:
        print(result.data.title)
        print(result.data.url)
        print(result.data.content)
    ```

=== "Deep"

    ```python
    import msgflux as mf

    retriever = mf.Retriever.web_search(
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

    retriever = mf.Retriever.web_search(
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

    retriever = mf.Retriever.web_search("linkup", depth="standard")

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

    retriever = mf.Retriever.web_search("linkup", depth="deep")

    response = await retriever.acall(["Python 3.14", "Django release"], top_k=2)

    for item in response.data:
        print(item.results[0].data.title)
    ```

---

## 6. **Exa Search**

The `exa` retriever queries Exa for semantic web search results. It can return URLs only, or fetch page text together with each result for RAG and summarization workflows.

!!! info "Dependencies"
    HTTPX2 is installed with msgFlux. No Exa SDK is required. Set the
    `EXA_API_KEY` env variable; the key is read per request.

    Synchronous and async calls use direct requests to
    `https://api.exa.ai/search`.

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
| `base_url` | `"https://api.exa.ai"` | Exa API base URL |
| `timeout` | `30.0` | Request timeout in seconds |

### Examples

=== "Web"

    ```python
    import msgflux as mf

    mf.set_envs(EXA_API_KEY="...")

    retriever = mf.Retriever.web_search("exa", include_text=True)
    response = retriever("latest Python packaging changes", top_k=3)

    for result in response.data[0].results:
        print(result.data.title)
        print(result.data.url)
        print(result.data.content[:300])
    ```

=== "URL Only"

    ```python
    import msgflux as mf

    retriever = mf.Retriever.web_search("exa", include_text=False)
    response = retriever("Python web frameworks", top_k=5)

    for result in response.data[0].results:
        print(result.data.title)
        print(result.data.url)
    ```

=== "Filters"

    ```python
    import msgflux as mf

    retriever = mf.Retriever.web_search(
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

    retriever = mf.Retriever.web_search("exa", search_type="auto", include_text=True)

    response = await retriever.acall(["Python 3.14", "Django release"], top_k=2)

    for item in response.data:
        print(item.results[0].data.title)
    ```

---

## 7. **SearXNG Search**

The `searxng` retriever queries a local or self-hosted SearXNG instance and returns structured web results with title, content, URL, and optional image metadata. SearXNG is useful when you want free, private, local web search without adding a provider SDK or API key.

!!! info "Dependencies"
    HTTPX2 is installed with msgFlux. A running SearXNG instance with JSON
    output enabled is also required.

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

    retriever = mf.Retriever.web_search(
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

    retriever = mf.Retriever.web_search(
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

    retriever = mf.Retriever.web_search("searxng", base_url="http://localhost:8888")

    response = await retriever.acall(["Python 3.14", "Django release"], top_k=2)

    for item in response.data:
        print(item.results[0].data.title)
    ```

---

## 8. **Ceramic Search**

The `ceramic` retriever queries Ceramic Search and returns structured web results with title, content, and URL. Ceramic is a web search provider based on lexical query matching.

!!! info "Dependencies"
    HTTPX2 is installed with msgFlux. Set the `CERAMIC_API_KEY` env variable.

    Both synchronous and async calls use direct requests to
    `https://api.ceramic.ai/search`.

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `timeout` | `30.0` | Request timeout in seconds |

### Examples

=== "Search"

    ```python
    import msgflux as mf

    mf.set_envs(CERAMIC_API_KEY="...")

    retriever = mf.Retriever.web_search("ceramic")
    response = retriever("California rental laws", top_k=3)

    for result in response.data[0].results:
        print(result.data.title)
        print(result.data.url)
        print(result.data.content)
    ```

=== "Async"

    ```python
    import msgflux as mf

    retriever = mf.Retriever.web_search("ceramic", timeout=10.0)
    response = await retriever.acall("California rental laws", top_k=3)

    for result in response.data[0].results:
        print(result.data.title)
    ```

---

## 9. **arXiv Search**

The `arxiv` retriever searches arXiv papers and returns structured academic metadata such as title, summary, authors, publication dates, categories, and PDF URLs.

!!! info "Dependencies"
    HTTPX2 is installed with msgFlux. No arXiv SDK is required; Atom XML is
    parsed with Python's standard-library `xml.etree.ElementTree`.

    Synchronous and async calls use direct requests to
    `https://export.arxiv.org/api/query`.

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_results` | `10` | Maximum number of arXiv results fetched per query |
| `sort_by` | `"relevance"` | Sort criterion: `"relevance"`, `"lastUpdatedDate"`, or `"submittedDate"` |
| `sort_order` | `"descending"` | Sort order: `"ascending"` or `"descending"` |
| `timeout` | `30.0` | Request timeout in seconds |

### Examples

=== "Search"

    ```python
    import msgflux as mf

    retriever = mf.Retriever.web_search("arxiv")
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

    retriever = mf.Retriever.web_search(
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

    retriever = mf.Retriever.web_search("arxiv", sort_by="relevance")

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

    retriever = mf.Retriever.web_search("arxiv", sort_by="submittedDate")

    response = await retriever.acall(["RAG evaluation", "agent benchmarks"], top_k=2)

    for item in response.data:
        print(item.results[0].data.title)
    ```

---

## 10. **Firecrawl Search**

The `firecrawl` retriever queries Firecrawl v2 search and is the default engine for `WebSearchTool` (`retriever/firecrawl`). It works without an API key.

!!! info "Dependencies"
    HTTPX2 is installed with msgFlux. No Firecrawl SDK is required.

    Firecrawl search is keyless by default. Set `FIRECRAWL_API_KEY` only when you
    want requests to include a Bearer token; the value is read at request time
    and is never stored on the retriever. msgFlux does not buy credits or switch
    to a paid provider automatically.

    Synchronous and async calls use direct requests to
    `https://api.firecrawl.dev/v2/search`.

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `base_url` | `"https://api.firecrawl.dev/v2"` | Firecrawl v2 API root |
| `include_domains` | `None` | Restrict results to these hostnames |
| `exclude_domains` | `None` | Exclude results from these hostnames |
| `tbs` | `None` | Time-based search value, such as `"qdr:w"` |
| `location` | `None` | Search location string |
| `country` | `None` | ISO country code for results |
| `categories` | `None` | Search categories, such as `["research", "pdf"]` |
| `timeout` | `30.0` | Request timeout in seconds |

### Request behavior

- Without a key, requests send no `Authorization` header. With
  `FIRECRAWL_API_KEY`, the same request sends `Authorization: Bearer ...`.
- `top_k` is sent as Firecrawl's `limit` and parsed results are also capped at
  `top_k`.
- Search results request Markdown content and map each result to
  `data.title`, `data.content`, and `data.url`. `content` uses the scraped
  Markdown when available and falls back to Firecrawl's `description`.
- The keyless path has no client-side quota counter or invented allowance.
  Firecrawl applies its own keyless/account limits.
- Rate-limit and quota failures are surfaced as `RuntimeError` with the HTTP
  status and provider detail, rather than becoming an empty result list. For
  example, an HTTP 429 includes `rate limit`, and a `success: false` payload
  includes Firecrawl's message such as `keyless quota exhausted`.

### Examples

=== "Default Search"

    ```python
    import msgflux as mf

    retriever = mf.Retriever.web_search("firecrawl")
    response = retriever("latest Python release", top_k=3)

    for result in response.data[0].results:
        print(result.data.title)
        print(result.data.url)
        print(result.data.content)
    ```

=== "Default WebSearchTool"

    ```python
    from msgflux.tools.builtin import WebSearchTool

    tool = WebSearchTool()
    print(tool.engine)  # retriever/firecrawl

    result = tool("latest Python release")
    print(result["data"])
    ```

=== "Filters"

    ```python
    import msgflux as mf

    retriever = mf.Retriever.web_search(
        "firecrawl",
        include_domains=["python.org", "pypi.org"],
        exclude_domains=["example.com"],
        tbs="qdr:w",
        country="US",
    )
    response = retriever("packaging metadata", top_k=5)

    for result in response.data[0].results:
        print(result.data.title)
    ```

=== "Optional API Key"

    ```python
    import msgflux as mf

    mf.set_envs(FIRECRAWL_API_KEY="...")

    retriever = mf.Retriever.web_search("firecrawl", timeout=20)
    response = retriever("Python packaging standards", top_k=2)

    for result in response.data[0].results:
        print(result.data.title)
    ```

=== "Quota Errors"

    ```python
    import msgflux as mf

    retriever = mf.Retriever.web_search("firecrawl")

    try:
        response = retriever("Python packaging standards", top_k=2)
    except RuntimeError as exc:
        # Includes the HTTP status and Firecrawl's rate-limit/quota detail.
        print(exc)
    ```

=== "Async"

    ```python
    import msgflux as mf

    retriever = mf.Retriever.web_search("firecrawl", country="US")

    response = await retriever.acall(["Python 3.14", "Django release"], top_k=2)

    for item in response.data:
        print(item.results[0].data.title)
    ```
