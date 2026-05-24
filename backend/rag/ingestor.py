from pathlib import Path

import httpx
from bs4 import BeautifulSoup

from backend.rag.chunker import TextChunker
from backend.rag.providers.base import VectorStoreProvider


class DocumentIngestor:
    """Ingests text files, PDFs, and URLs into the configured vector store provider."""

    def __init__(self, provider: VectorStoreProvider, chunker: TextChunker) -> None:
        self._provider = provider
        self._chunker = chunker

    def ingest_text(self, text: str, metadata: dict | None = None) -> int:
        """Chunk and index a raw text string. Returns the number of chunks added."""
        chunks = self._chunker.chunk(text, metadata or {})
        if not chunks:
            return 0
        self._provider.upsert(chunks)
        return len(chunks)

    def ingest_file(self, path: str | Path) -> int:
        """Read a text file and ingest its contents. Returns chunks added."""
        file_path = Path(path)
        return self.ingest_text(
            file_path.read_text(encoding="utf-8"),
            metadata={"source": str(file_path)},
        )

    def ingest_url(self, url: str) -> int:
        """Fetch a URL, extract its text, and ingest it. Returns chunks added."""
        response = httpx.get(url, timeout=15.0, follow_redirects=True)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        return self.ingest_text(text, metadata={"source": url})
