from backend.rag.vector_store import VectorStore


class HybridRetriever:
    """Retrieves relevant document chunks using semantic similarity search."""

    def __init__(self, vector_store: VectorStore) -> None:
        self._store = vector_store

    def retrieve(self, query: str, n_results: int = 5) -> list[dict]:
        """Return the top-n relevant chunks for a query."""
        return self._store.query(query_text=query, n_results=n_results)

    def retrieve_formatted(self, query: str, n_results: int = 5) -> str:
        """Return retrieved chunks as a single formatted string for prompt injection."""
        results = self.retrieve(query, n_results)
        if not results:
            return "No relevant documents found."
        return "\n\n".join(
            f"[Score: {r['score']:.2f}] {r['content']}" for r in results
        )
