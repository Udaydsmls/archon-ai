from backend.config import settings
from backend.rag.providers.base import VectorStoreProvider


def get_vector_store_provider() -> VectorStoreProvider:
    """Return the configured vector store provider based on VECTOR_STORE_PROVIDER env var."""
    provider = settings.vector_store_provider.lower()

    if provider == "pinecone":
        from backend.rag.providers.pinecone_provider import PineconeProvider

        return PineconeProvider()

    if provider == "weaviate":
        from backend.rag.providers.weaviate_provider import WeaviateProvider

        return WeaviateProvider()

    from backend.rag.providers.chroma_provider import ChromaProvider

    return ChromaProvider()
