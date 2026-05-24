from backend.rag.providers.base import VectorStoreProvider


class SemanticRetriever:
    """Retrieves relevant document chunks using dense semantic similarity search."""

    def __init__(self, provider: VectorStoreProvider) -> None:
        self._provider = provider

    def retrieve(self, query: str, n_results: int = 5) -> list[dict]:
        """Return the top-n most similar chunks for a query."""
        return self._provider.similarity_search(query, top_k=n_results)

    def retrieve_formatted(self, query: str, n_results: int = 5) -> str:
        """Return retrieved chunks as a formatted string for prompt injection."""
        results = self.retrieve(query, n_results)
        if not results:
            return "No relevant documents found."
        return "\n\n".join(
            f"[Score: {r['score']:.2f}] {r['content']}" for r in results
        )
