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


class WeatherRetriever:
    """Retrieve current, forecast, or historical weather data."""

    retriever_type = "weather"


class FuzzyRetriever:
    """Use approximate string matching to find similar documents."""

    retriever_type = "fuzzy"
