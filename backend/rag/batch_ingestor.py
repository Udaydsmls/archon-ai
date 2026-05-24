import time
from pathlib import Path

import anthropic
import httpx
from bs4 import BeautifulSoup

from backend.config import settings
from backend.rag.chunker import Chunk, TextChunker
from backend.rag.providers.base import VectorStoreProvider

_SUMMARISE_PROMPT = (
    "Summarise the following document chunk in 2–3 sentences, preserving key facts:\n\n{content}"
)
_POLL_INTERVAL_SECONDS = 5


class BatchIngestor:
    """Ingests multiple documents using the Anthropic Batch API for cost-efficient summarisation."""

    def __init__(self, provider: VectorStoreProvider, chunker: TextChunker) -> None:
        self._provider = provider
        self._chunker = chunker
        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    def ingest_batch(self, urls: list[str] = (), file_paths: list[str] = ()) -> str:
        """Chunk all sources, submit a single Anthropic batch job, then upsert summaries.

        Returns the batch ID for status tracking.
        """
        chunks = self._collect_chunks(urls, file_paths)
        if not chunks:
            return ""

        batch_requests = [
            {
                "custom_id": f"chunk-{i}",
                "params": {
                    "model": settings.primary_model,
                    "max_tokens": 256,
                    "messages": [
                        {
                            "role": "user",
                            "content": _SUMMARISE_PROMPT.format(content=c.content),
                        }
                    ],
                },
            }
            for i, c in enumerate(chunks)
        ]

        batch = self._client.beta.messages.batches.create(requests=batch_requests)
        summaries = self._poll_until_complete(batch.id)
        self._upsert_summaries(chunks, summaries)
        return batch.id

    def _collect_chunks(
        self, urls: list[str], file_paths: list[str]
    ) -> list[Chunk]:
        """Fetch all sources, extract text, and return a flat list of chunks."""
        all_chunks: list[Chunk] = []
        for url in urls:
            try:
                response = httpx.get(url, timeout=15.0, follow_redirects=True)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, "html.parser")
                for tag in soup(["script", "style", "nav", "footer"]):
                    tag.decompose()
                text = soup.get_text(separator="\n", strip=True)
                all_chunks.extend(self._chunker.chunk(text, {"source": url}))
            except Exception:
                continue

        for path_str in file_paths:
            try:
                text = Path(path_str).read_text(encoding="utf-8")
                all_chunks.extend(self._chunker.chunk(text, {"source": path_str}))
            except Exception:
                continue

        return all_chunks

    def _poll_until_complete(self, batch_id: str) -> dict[str, str]:
        """Poll the Anthropic Batch API until the job finishes. Returns custom_id → summary map."""
        while True:
            batch = self._client.beta.messages.batches.retrieve(batch_id)
            if batch.processing_status == "ended":
                break
            time.sleep(_POLL_INTERVAL_SECONDS)

        summaries: dict[str, str] = {}
        for result in self._client.beta.messages.batches.results(batch_id):
            if result.result.type == "succeeded":
                text = next(
                    (b.text for b in result.result.message.content if hasattr(b, "text")), ""
                )
                summaries[result.custom_id] = text
        return summaries

    def _upsert_summaries(self, chunks: list[Chunk], summaries: dict[str, str]) -> None:
        """Build summary chunks and upsert them alongside the originals."""
        summary_chunks = []
        for i, chunk in enumerate(chunks):
            summary = summaries.get(f"chunk-{i}", "")
            if summary:
                summary_chunks.append(
                    Chunk(
                        content=summary,
                        metadata={**chunk.metadata, "type": "summary"},
                        chunk_index=chunk.chunk_index,
                    )
                )
        if summary_chunks:
            self._provider.upsert(summary_chunks)
        self._provider.upsert(chunks)
