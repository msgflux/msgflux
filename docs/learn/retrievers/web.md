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

## 9. **Brave Search**

The `brave` retriever queries Brave Search and can return web, news, or image results. Use it when you need search results from Brave with a single provider interface.

!!! info "Dependencies"
    Requires `brave-search-python-client` and the `BRAVE_SEARCH_API_KEY` env variable:
    `pip install brave-search-python-client`

### Quick Start

???+ example

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

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mode` | `"search"` | Search mode: `"search"`, `"news"`, or `"image"` |
| `return_images` | `False` | Whether to include thumbnail image URLs for web/news results |

```python
import msgflux as mf

retriever = mf.Retriever.web(
    "brave",
    mode="search",
    return_images=True,
)
```

### News Search

Use `mode="news"` to retrieve news results:

???+ example

    ```python
    import msgflux as mf

    retriever = mf.Retriever.web("brave", mode="news")
    response = retriever("AI regulation", top_k=5)

    for result in response.data[0].results:
        print(result.data.title)
        print(result.data.date)
        print(result.data.url)
    ```

### Image Search

Use `mode="image"` when the image URL is the primary result:

???+ example

    ```python
    import msgflux as mf

    retriever = mf.Retriever.web("brave", mode="image")
    response = retriever("James Webb Space Telescope", top_k=3)

    for result in response.data[0].results:
        print(result.data.title)
        print(result.images[0])
    ```

### Async Search

```python
import msgflux as mf

retriever = mf.Retriever.web("brave", mode="search")

response = await retriever.acall(["Python 3.14", "Django release"], top_k=2)

for item in response.data:
    print(item.results[0].data.title)
```
