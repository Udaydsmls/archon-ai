from dataclasses import dataclass


@dataclass
class Chunk:
    """A single text chunk with its source metadata and position index."""

    content: str
    metadata: dict
    chunk_index: int


class TextChunker:
    """Splits documents into overlapping fixed-size word chunks."""

    def __init__(self, chunk_size: int = 512, overlap: int = 64) -> None:
        self._chunk_size = chunk_size
        self._overlap = overlap

    def chunk(self, text: str, metadata: dict | None = None) -> list[Chunk]:
        """Split text into overlapping chunks and return them with metadata."""
        metadata = metadata or {}
        words = text.split()
        step = self._chunk_size - self._overlap
        chunks = []

        for i, start in enumerate(range(0, len(words), step)):
            content = " ".join(words[start : start + self._chunk_size])
            if content.strip():
                chunks.append(Chunk(content=content, metadata=metadata, chunk_index=i))

        return chunks
