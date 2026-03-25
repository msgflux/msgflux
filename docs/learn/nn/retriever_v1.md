# Retriever: Information Retrieval Module

The `Retriever` module provides a unified interface for information retrieval using web search, lexical search, semantic search, or vector databases.

**All code examples use the recommended import pattern:**

```python
import msgflux as mf
```

## Quick Start

### Traditional Initialization

```python
import msgflux as mf

# Create a vector database retriever
vector_db = mf.data.VectorDB.qdrant(
    collection_name="documents",
    url="http://localhost:6333"
)

# Create embedding model for semantic search
embedder = mf.Model.text_embedder("openai/text-embedding-3-small")

# Traditional retriever initialization
retriever = mf.nn.Searcher(
    retriever=vector_db,
    model=embedder,
    config={"top_k": 5, "threshold": 0.7}
)

# Use the retriever
results = retriever("What is machine learning?")
print(results)
```

### AutoParams Initialization (Recommended)

**This is the preferred and recommended way to define retrievers in msgflux.**

```python
import msgflux as mf

class DocumentRetriever(mf.nn.Searcher):
    """Semantic retriever for technical documentation."""

    # AutoParams automatically uses class name as 'name'
    # Define configuration as class attributes
    response_mode = "plain_response"

# Setup
vector_db = mf.data.VectorDB.qdrant(
    collection_name="documents",
    url="http://localhost:6333"
)

embedder = mf.Model.text_embedder("openai/text-embedding-3-small")

# Create instance with defaults
doc_retriever = DocumentRetriever(
    retriever=vector_db,
    model=embedder,
    config={"top_k": 5, "threshold": 0.7}
)

# Use the retriever
results = doc_retriever("What is machine learning?")
print(results)

# Override configuration at runtime
detailed_results = DocumentRetriever(
    retriever=vector_db,
    model=embedder,
    config={"top_k": 10, "threshold": 0.5, "return_score": True}
)
```

## Why Use AutoParams?

1. **Domain-Specific Retrievers**: Create specialized retrievers for different knowledge domains
2. **Reusable Configuration**: Share configuration across retriever instances
3. **Clear Intent**: Class name and docstring document the retriever's purpose
4. **Easy Customization**: Override defaults per instance or create retriever hierarchies
5. **Better Organization**: Group related retrievers by domain or use case

## Retriever Types

### 1. Vector Database Retrieval (Semantic)

Most common type - uses embeddings for semantic similarity search.

#### Traditional Approach

```python
import msgflux as mf

# Setup vector database
vector_db = mf.data.VectorDB.qdrant(
    collection_name="knowledge_base",
    url="http://localhost:6333"
)

# Create embedding model
embedder = mf.Model.text_embedder("openai/text-embedding-3-small")

# Create retriever
retriever = mf.nn.Searcher(
    retriever=vector_db,
    model=embedder,
    config={
        "top_k": 5,
        "threshold": 0.75,
        "return_score": True
    }
)

# Search
results = retriever("How do I use async functions in Python?")
for result in results["results"]:
    print(f"Score: {result['score']}")
    print(f"Content: {result['data']}")
```

#### AutoParams Approach (Recommended)

```python
import msgflux as mf

class KnowledgeBaseRetriever(mf.nn.Searcher):
    """Semantic retriever for company knowledge base with high precision."""

    response_mode = "plain_response"

# Setup
vector_db = mf.data.VectorDB.qdrant(
    collection_name="knowledge_base",
    url="http://localhost:6333"
)

embedder = mf.Model.text_embedder("openai/text-embedding-3-small")

# Create with strict thresholds for high precision
kb_retriever = KnowledgeBaseRetriever(
    retriever=vector_db,
    model=embedder,
    config={"top_k": 5, "threshold": 0.75, "return_score": True}
)

results = kb_retriever("How do I use async functions in Python?")
```

### 2. Web Retrieval

Search the web for information.

```python
import msgflux as mf

class WebResearcher(mf.nn.Searcher):
    """Web search retriever for real-time information."""

    response_mode = "plain_response"

# Create web retriever
web_search = mf.data.WebRetriever(api_key="your-api-key")

researcher = WebResearcher(
    retriever=web_search,
    config={"top_k": 10}
)

# Search current events
results = researcher("Latest news on artificial intelligence")
print(results)
```

### 3. Lexical Retrieval

Traditional keyword-based search (BM25, TF-IDF).

```python
import msgflux as mf

class DocumentSearcher(mf.nn.Searcher):
    """Fast lexical search for exact keyword matching."""

    response_mode = "plain_response"

# Create lexical retriever
lexical_search = mf.data.LexicalRetriever(
    documents=["doc1.txt", "doc2.txt", "doc3.txt"]
)

searcher = DocumentSearcher(
    retriever=lexical_search,
    config={"top_k": 3}
)

results = searcher("Python async await")
```

## Advanced Configuration

### Message Field Mapping

Use Message objects for structured input/output.

```python
import msgflux as mf

class ContextualRetriever(mf.nn.Searcher):
    """Retriever that processes structured message inputs."""

    response_mode = "message"  # Return Message object

vector_db = mf.data.VectorDB.qdrant(
    collection_name="docs",
    url="http://localhost:6333"
)

embedder = mf.Model.text_embedder("openai/text-embedding-3-small")

retriever = ContextualRetriever(
    retriever=vector_db,
    model=embedder,
    message_fields={
        "task_inputs": "query.user"  # Extract query from message.query.user
    },
    config={"top_k": 5}
)

# Use with Message
msg = mf.Message()
msg.set("query.user", "What is dependency injection?")

result_msg = retriever(msg)
print(result_msg.get("retriever.results"))
```

### Response Templates

Format retrieval results using Jinja templates.

```python
import msgflux as mf

class FormattedRetriever(mf.nn.Searcher):
    """Retriever with custom result formatting."""

    response_mode = "plain_response"

vector_db = mf.data.VectorDB.qdrant(
    collection_name="articles",
    url="http://localhost:6333"
)

embedder = mf.Model.text_embedder("openai/text-embedding-3-small")

retriever = FormattedRetriever(
    retriever=vector_db,
    model=embedder,
    templates={
        "response": """
        Found {{ results|length }} relevant articles:
        {% for result in results %}
        {{ loop.index }}. {{ result.data }} (Score: {{ result.score }})
        {% endfor %}
        """
    },
    config={"top_k": 5, "return_score": True}
)

formatted_results = retriever("machine learning best practices")
print(formatted_results)
```

### Runtime Configuration Override

Override configuration at call time.

```python
import msgflux as mf

class FlexibleRetriever(mf.nn.Searcher):
    """Retriever with runtime-configurable parameters."""

    response_mode = "plain_response"

vector_db = mf.data.VectorDB.qdrant(
    collection_name="docs",
    url="http://localhost:6333"
)

embedder = mf.Model.text_embedder("openai/text-embedding-3-small")

retriever = FlexibleRetriever(
    retriever=vector_db,
    model=embedder,
    config={"top_k": 3}  # Default
)

# Normal retrieval with defaults
results = retriever("python tutorial")

# Override task_inputs at runtime
results = retriever(
    "machine learning",
    task_inputs="deep learning neural networks"  # Override query
)
```

## Creating Retriever Hierarchies

Build specialized retrievers through inheritance.

```python
import msgflux as mf

# Base retriever for all documentation
class BaseDocRetriever(mf.nn.Searcher):
    """Base retriever for documentation with common config."""

    response_mode = "plain_response"

# High precision retriever for critical docs
class CriticalDocRetriever(BaseDocRetriever):
    """High precision retriever for critical documentation (compliance, security)."""

    # Inherits response_mode from BaseDocRetriever
    # Override defaults for stricter matching

# Broad retriever for general exploration
class ExploratoryRetriever(BaseDocRetriever):
    """Broad retriever for exploratory searches."""

    # Lower threshold for diverse results

# Setup
vector_db = mf.data.VectorDB.qdrant(
    collection_name="documentation",
    url="http://localhost:6333"
)

embedder = mf.Model.text_embedder("openai/text-embedding-3-small")

# Create instances with different thresholds
critical = CriticalDocRetriever(
    retriever=vector_db,
    model=embedder,
    config={"top_k": 3, "threshold": 0.85}  # Strict
)

exploratory = ExploratoryRetriever(
    retriever=vector_db,
    model=embedder,
    config={"top_k": 10, "threshold": 0.5}  # Permissive
)

# Use appropriately
compliance_docs = critical("GDPR data retention requirements")
related_topics = exploratory("data privacy regulations")
```

## Integration with Agents

Retrievers are commonly used as tools for agents.

```python
import msgflux as mf

# Define retriever
class CompanyKnowledgeRetriever(mf.nn.Searcher):
    """Retrieves information from company knowledge base."""

    response_mode = "plain_response"

vector_db = mf.data.VectorDB.qdrant(
    collection_name="company_kb",
    url="http://localhost:6333"
)

embedder = mf.Model.text_embedder("openai/text-embedding-3-small")

kb_retriever = CompanyKnowledgeRetriever(
    retriever=vector_db,
    model=embedder,
    config={"top_k": 5, "threshold": 0.7}
)

# Define retriever as a tool function
def search_knowledge_base(query: str) -> str:
    """Search the company knowledge base for information."""
    results = kb_retriever(query)
    return results

# Create agent with retriever tool
class SupportAgent(mf.nn.Agent):
    """Customer support agent with access to knowledge base."""

    temperature = 0.7
    max_tokens = 2000
    max_tool_iterations = 3

model = mf.Model.chat_completion("openai/gpt-4")

support_agent = SupportAgent(
    model=model,
    tools=[search_knowledge_base]
)

# Agent can now use retriever to answer questions
response = support_agent("How do I reset my password?")
print(response)
```

## Configuration Options

### Complete Parameter Reference

```python
import msgflux as mf

class ConfiguredRetriever(mf.nn.Searcher):
    """Fully configured retriever example."""

    # Response behavior
    response_mode = "plain_response"  # or "message"

# Initialize with all options
retriever = ConfiguredRetriever(
    retriever=vector_db,              # Required: retriever backend
    model=embedder,                   # Optional: for semantic retrieval
    message_fields={                  # Optional: Message field mapping
        "task_inputs": "query.text"
    },
    templates={                       # Optional: Jinja response templates
        "response": "Results: {{ content }}"
    },
    config={                          # Optional: retriever-specific config
        "top_k": 5,                   # Max results to return
        "threshold": 0.7,             # Minimum similarity score
        "return_score": True,         # Include scores in results
        "dict_key": "content"         # Extract specific dict key from results
    },
    name="custom_retriever"           # Optional: custom name
)
```

## Async Support

Retrievers support asynchronous operation.

```python
import msgflux as mf
import asyncio

class AsyncRetriever(mf.nn.Searcher):
    """Async retriever for concurrent searches."""

    response_mode = "plain_response"

vector_db = mf.data.VectorDB.qdrant(
    collection_name="docs",
    url="http://localhost:6333"
)

embedder = mf.Model.text_embedder("openai/text-embedding-3-small")

retriever = AsyncRetriever(
    retriever=vector_db,
    model=embedder,
    config={"top_k": 5}
)

async def search_multiple():
    """Perform multiple searches concurrently."""
    results = await asyncio.gather(
        retriever.aforward("machine learning"),
        retriever.aforward("deep learning"),
        retriever.aforward("neural networks")
    )
    return results

# Run async searches
results = asyncio.run(search_multiple())
```

## Best Practices

### 1. Use Domain-Specific Retrievers

```python
# Good - Clear, specialized retrievers
class TechnicalDocRetriever(mf.nn.Searcher):
    """Retriever for technical documentation (APIs, guides)."""
    response_mode = "plain_response"

class CustomerQueryRetriever(mf.nn.Searcher):
    """Retriever for customer support queries."""
    response_mode = "plain_response"

class ProductCatalogRetriever(mf.nn.Searcher):
    """Retriever for product information."""
    response_mode = "plain_response"
```

### 2. Tune Thresholds by Use Case

```python
# High precision for critical applications
class ComplianceRetriever(mf.nn.Searcher):
    """High precision retriever for compliance documentation."""
    response_mode = "plain_response"

compliance = ComplianceRetriever(
    retriever=vector_db,
    model=embedder,
    config={"top_k": 3, "threshold": 0.9}  # Very strict
)

# High recall for exploratory search
class ResearchRetriever(mf.nn.Searcher):
    """Broad retriever for research and exploration."""
    response_mode = "plain_response"

research = ResearchRetriever(
    retriever=vector_db,
    model=embedder,
    config={"top_k": 20, "threshold": 0.5}  # Permissive
)
```

### 3. Return Scores for Transparency

```python
class TransparentRetriever(mf.nn.Searcher):
    """Retriever that returns similarity scores for verification."""
    response_mode = "plain_response"

retriever = TransparentRetriever(
    retriever=vector_db,
    model=embedder,
    config={
        "top_k": 5,
        "return_score": True  # Always include scores
    }
)

results = retriever("query")
for r in results["results"]:
    if r["score"] < 0.7:
        print(f"Warning: Low confidence result (score: {r['score']})")
```

## Migration Guide

### From Traditional to AutoParams

**Before (Traditional):**
```python
retriever = mf.nn.Searcher(
    retriever=vector_db,
    model=embedder,
    config={"top_k": 5, "threshold": 0.7},
    response_mode="plain_response"
)
```

**After (AutoParams - Recommended):**
```python
class MyRetriever(mf.nn.Searcher):
    """Semantic retriever for my use case."""
    response_mode = "plain_response"

retriever = MyRetriever(
    retriever=vector_db,
    model=embedder,
    config={"top_k": 5, "threshold": 0.7}
)
```

## Summary

- **Use AutoParams** for defining retrievers - cleaner and more maintainable
- **Traditional initialization** works for quick, one-off retrievers
- Supports **vector DB**, **web search**, and **lexical search**
- Configure via **message_fields**, **templates**, and **config** options
- Commonly integrated with **Agents** as tools
- **Async support** for concurrent retrieval operations

The Retriever module provides flexible, powerful information retrieval - use AutoParams to organize retrievers by domain and use case.