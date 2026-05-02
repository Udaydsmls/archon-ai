import chromadb
from chromadb.utils import embedding_functions

from backend.config import settings


class VectorStore:
    """ChromaDB-backed persistent vector store for document retrieval."""

    def __init__(self, collection_name: str = "documents") -> None:
        self._client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        self._ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=self._ef,
        )

    def add(self, documents: list[str], metadatas: list[dict], ids: list[str]) -> None:
        """Add documents with metadata to the collection."""
        self._collection.add(documents=documents, metadatas=metadatas, ids=ids)

    def query(self, query_text: str, n_results: int = 5) -> list[dict]:
        """Return the top-n most similar documents with similarity scores."""
        results = self._collection.query(
            query_texts=[query_text],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
        return [
            {
                "content": doc,
                "metadata": meta,
                "score": 1 - dist,
            }
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]

    def count(self) -> int:
        """Return the number of documents currently in the collection."""
        return self._collection.count()
