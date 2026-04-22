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

## 9. **Tavily Search**

The `tavily` retriever queries Tavily and returns search results optimized for AI applications. It supports search depth, topic filters, time ranges, domain filters, generated answers, images, and raw page content.

!!! info "Dependencies"
    Requires `tavily-python` and the `TAVILY_API_KEY` env variable:
    `pip install tavily-python`

### Quick Start

???+ example

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

```python
import msgflux as mf

retriever = mf.Retriever.web(
    "tavily",
    search_depth="advanced",
    topic="news",
    time_range="week",
)
```

### Raw Content

Enable `include_raw_content=True` when the downstream model needs more complete page text:

???+ example

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

### Domain Filters

```python
import msgflux as mf

retriever = mf.Retriever.web(
    "tavily",
    include_domains=["python.org", "pypi.org"],
    exclude_domains=["example.com"],
)

response = retriever("packaging metadata", top_k=3)
```

### Async Search

```python
import msgflux as mf

retriever = mf.Retriever.web("tavily", search_depth="advanced")

response = await retriever.acall(["Python 3.14", "Django release"], top_k=2)

for item in response.data:
    print(item.results[0].data.title)
```
