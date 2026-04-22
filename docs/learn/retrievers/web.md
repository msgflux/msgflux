# Web Retrievers

## ✦₊⁺ Overview

The `wikipedia` retriever fetches and returns Wikipedia article content at query time. Unlike lexical retrievers, it requires no pre-indexed corpus — it queries the Wikipedia API directly and returns structured results with title, content, and optionally images.

!!! info "Dependencies"
    Requires the `wikipedia` package: `pip install wikipedia`

---

## 1. **Quick Start**

???+ example

    ```python
    import msgflux as mf

    retriever = mf.Retriever.web("wikipedia")

    response = retriever("machine learning", top_k=2)

    for result in response.data[0].results:
        print(result.data.title)
        print(result.data.content[:200])
    ```

---

## 2. **Parameters**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `language` | `"en"` | Wikipedia language code (`"pt"`, `"es"`, `"fr"`, …) |
| `summary` | `None` | Number of sentences to return — `None` returns the full article |
| `return_images` | `False` | Whether to include image URLs in results |
| `max_return_images` | `5` | Maximum number of image URLs per result |

```python
import msgflux as mf

retriever = mf.Retriever.web("wikipedia",
    language="en",
    summary=3,           # Return only the first 3 sentences
    return_images=True,
    max_return_images=3,
)
```

---

## 3. **Summary Mode**

By default, the full article content is returned. Set `summary` to an integer to limit the response to the first N sentences — useful when feeding context to an LLM:

???+ example

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

---

## 4. **Images**

Enable `return_images=True` to get a list of image URLs from each article. Icons, logos, and SVGs are filtered automatically:

???+ example

    ```python
    import msgflux as mf

    retriever = mf.Retriever.web("wikipedia",
        return_images=True,
        max_return_images=3
    )

    response = retriever("Colosseum")

    result = response.data[0].results[0]
    print(result.data.title)    # "Colosseum"
    print(result.images)        # ["https://upload.wikimedia.org/...jpg", ...]
    ```

---

## 5. **Multilingual**

Set `language` to any Wikipedia language code:

???+ example

    === "Portuguese"

        ```python
        import msgflux as mf

        retriever = mf.Retriever.web("wikipedia", language="pt", summary=3)
        response = retriever("inteligência artificial")
        print(response.data[0].results[0].data.content)
        ```

    === "Spanish"

        ```python
        import msgflux as mf

        retriever = mf.Retriever.web("wikipedia", language="es", summary=3)
        response = retriever("aprendizaje automático")
        print(response.data[0].results[0].data.content)
        ```

    === "French"

        ```python
        import msgflux as mf

        retriever = mf.Retriever.web("wikipedia", language="fr", summary=3)
        response = retriever("réseau de neurones")
        print(response.data[0].results[0].data.content)
        ```

---

## 6. **Batch Queries**

```python
import msgflux as mf

retriever = mf.Retriever.web("wikipedia", summary=2)

queries = ["Python programming", "Rust programming language", "Go programming"]
response = retriever(queries, top_k=1)

for i, query in enumerate(queries):
    result = response.data[i].results[0]
    print(f"\n{result.data.title}")
    print(result.data.content)
```

---

## 7. **RAG Integration**

A typical pattern: retrieve Wikipedia context, then pass it to an LLM:

???+ example

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
            "content": f"Context:\n{context}\n\nQuestion: {question}"
        }]).consume()

    print(answer_with_wikipedia("How does the James Webb Space Telescope work?"))
    ```

---

## 8. **Async Support**

```python
import msgflux as mf

retriever = mf.Retriever.web("wikipedia", summary=3)

queries = ["quantum computing", "photosynthesis", "black holes"]
response = await retriever.acall(queries, top_k=1)

for i, query in enumerate(queries):
    result = response.data[i].results[0]
    print(f"\n{query} → {result.data.title}")
```

---

## 9. **SerpApi Search**

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

## 10. **Brave Search**

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

## 11. **Tavily Search**

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

## 12. **Linkup Search**

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

## 13. **Exa Search**

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
