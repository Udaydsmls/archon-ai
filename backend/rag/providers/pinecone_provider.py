import hashlib

from sentence_transformers import SentenceTransformer

from backend.config import settings
from backend.rag.chunker import Chunk
from backend.rag.providers.base import VectorStoreProvider


class PineconeProvider(VectorStoreProvider):
    """VectorStoreProvider backed by a Pinecone serverless index."""

    _EMBEDDING_MODEL = "all-MiniLM-L6-v2"

    def __init__(self) -> None:
        from pinecone import Pinecone

        self._index = Pinecone(api_key=settings.pinecone_api_key).Index(
            settings.pinecone_index_name
        )
        self._encoder = SentenceTransformer(self._EMBEDDING_MODEL)

    def upsert(self, chunks: list[Chunk]) -> None:
        """Embed and upsert chunks into the Pinecone index."""
        embeddings = self._encoder.encode([c.content for c in chunks])
        vectors = [
            {
                "id": hashlib.sha256(c.content.encode()).hexdigest()[:16],
                "values": emb.tolist(),
                "metadata": {**c.metadata, "content": c.content},
            }
            for c, emb in zip(chunks, embeddings)
        ]
        self._index.upsert(vectors=vectors)

    def similarity_search(self, query: str, top_k: int = 5) -> list[dict]:
        """Query Pinecone by dense embedding and return scored results."""
        embedding = self._encoder.encode([query])[0].tolist()
        results = self._index.query(vector=embedding, top_k=top_k, include_metadata=True)
        return [
            {
                "content": match.metadata.get("content", ""),
                "metadata": match.metadata,
                "score": match.score,
            }
            for match in results.matches
        ]

    def hybrid_search(self, query: str, top_k: int = 5) -> list[dict]:
        """Delegate to similarity search — sparse hybrid requires a separate sparse index."""
        return self.similarity_search(query, top_k)
