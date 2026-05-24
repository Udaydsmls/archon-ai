import hashlib

from backend.rag.chunker import Chunk
from backend.rag.providers.base import VectorStoreProvider
from backend.rag.vector_store import VectorStore


class ChromaProvider(VectorStoreProvider):
    """Wraps the existing ChromaDB VectorStore to satisfy the VectorStoreProvider interface."""

    def __init__(self) -> None:
        self._store = VectorStore()

    def upsert(self, chunks: list[Chunk]) -> None:
        """Index a list of chunks into ChromaDB."""
        self._store.add(
            documents=[c.content for c in chunks],
            metadatas=[c.metadata for c in chunks],
            ids=[hashlib.sha256(c.content.encode()).hexdigest()[:16] for c in chunks],
        )

    def similarity_search(self, query: str, top_k: int = 5) -> list[dict]:
        """Run a dense similarity search against the ChromaDB collection."""
        return self._store.query(query_text=query, n_results=top_k)

    def hybrid_search(self, query: str, top_k: int = 5) -> list[dict]:
        """Delegate to similarity search — ChromaDB does not natively support BM25."""
        return self.similarity_search(query, top_k)
