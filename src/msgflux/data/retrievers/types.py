class LexicalRetriever:
    """It are based on the exact matching of terms between the query
    and the documents.
    """

    retriever_type = "lexical"


class SemanticRetriever:
    """Use vector representations (embeddings) to capture deeper semantic meanings."""

    retriever_type = "semantic"


class WebRetriever:
    """Retriever web content based-on queries."""

    retriever_type = "web"
