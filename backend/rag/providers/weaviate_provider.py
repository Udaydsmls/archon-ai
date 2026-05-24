from sentence_transformers import SentenceTransformer

from backend.config import settings
from backend.rag.chunker import Chunk
from backend.rag.providers.base import VectorStoreProvider

_COLLECTION_NAME = "ArchonDocument"


class WeaviateProvider(VectorStoreProvider):
    """VectorStoreProvider backed by Weaviate v4 with auto-schema creation on first run."""

    _EMBEDDING_MODEL = "all-MiniLM-L6-v2"

    def __init__(self) -> None:
        import weaviate

        self._client = weaviate.connect_to_local(host=settings.weaviate_url)
        self._encoder = SentenceTransformer(self._EMBEDDING_MODEL)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Create the Weaviate collection if it does not already exist."""
        import weaviate.classes.config as wvc

        if not self._client.collections.exists(_COLLECTION_NAME):
            self._client.collections.create(
                name=_COLLECTION_NAME,
                properties=[
                    wvc.Property(name="content", data_type=wvc.DataType.TEXT),
                    wvc.Property(name="source", data_type=wvc.DataType.TEXT),
                ],
            )

    def upsert(self, chunks: list[Chunk]) -> None:
        """Embed and batch-insert chunks into the Weaviate collection."""
        collection = self._client.collections.get(_COLLECTION_NAME)
        embeddings = self._encoder.encode([c.content for c in chunks])
        with collection.batch.dynamic() as batch:
            for chunk, embedding in zip(chunks, embeddings):
                batch.add_object(
                    properties={"content": chunk.content, **chunk.metadata},
                    vector=embedding.tolist(),
                )

    def similarity_search(self, query: str, top_k: int = 5) -> list[dict]:
        """Run a near-vector search and return scored results."""
        collection = self._client.collections.get(_COLLECTION_NAME)
        embedding = self._encoder.encode([query])[0].tolist()
        results = collection.query.near_vector(
            near_vector=embedding,
            limit=top_k,
            return_metadata=["distance"],
        )
        return [
            {
                "content": obj.properties.get("content", ""),
                "metadata": obj.properties,
                "score": 1 - obj.metadata.distance,
            }
            for obj in results.objects
        ]

    def hybrid_search(self, query: str, top_k: int = 5) -> list[dict]:
        """Run Weaviate's native BM25 + vector hybrid search."""
        collection = self._client.collections.get(_COLLECTION_NAME)
        results = collection.query.hybrid(query=query, limit=top_k)
        return [
            {
                "content": obj.properties.get("content", ""),
                "metadata": obj.properties,
                "score": 1.0,
            }
            for obj in results.objects
        ]
