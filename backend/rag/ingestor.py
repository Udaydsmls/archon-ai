import hashlib
import uuid
from pathlib import Path

import httpx
from bs4 import BeautifulSoup

from backend.rag.chunker import TextChunker
from backend.rag.vector_store import VectorStore


class DocumentIngestor:
    """Ingests text files, PDFs, and URLs into the vector store."""

    def __init__(self, vector_store: VectorStore, chunker: TextChunker) -> None:
        self._store = vector_store
        self._chunker = chunker

    def ingest_text(self, text: str, metadata: dict | None = None) -> int:
        """Chunk and index a raw text string. Returns the number of chunks added."""
        metadata = metadata or {}
        chunks = self._chunker.chunk(text, metadata)
        if not chunks:
            return 0

        self._store.add(
            documents=[c.content for c in chunks],
            metadatas=[c.metadata for c in chunks],
            ids=[self._chunk_id(c.content) for c in chunks],
        )
        return len(chunks)

    def ingest_file(self, path: str | Path) -> int:
        """Read a text file and ingest its contents. Returns chunks added."""
        file_path = Path(path)
        text = file_path.read_text(encoding="utf-8")
        return self.ingest_text(text, metadata={"source": str(file_path)})

    def ingest_url(self, url: str) -> int:
        """Fetch a URL, extract its text, and ingest it. Returns chunks added."""
        response = httpx.get(url, timeout=15.0, follow_redirects=True)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        return self.ingest_text(text, metadata={"source": url})

    @staticmethod
    def _chunk_id(content: str) -> str:
        """Generate a stable ID for a chunk based on its content hash."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]
