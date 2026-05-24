from abc import ABC, abstractmethod

from backend.rag.chunker import Chunk


class VectorStoreProvider(ABC):
    """Abstract interface for all vector store backends."""

    @abstractmethod
    def upsert(self, chunks: list[Chunk]) -> None:
        """Store or update a list of chunks in the backend."""
        ...

    @abstractmethod
    def similarity_search(self, query: str, top_k: int = 5) -> list[dict]:
        """Return top-k results ranked by semantic similarity."""
        ...

    @abstractmethod
    def hybrid_search(self, query: str, top_k: int = 5) -> list[dict]:
        """Return top-k results combining semantic and keyword search."""
        ...
