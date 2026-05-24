from rank_bm25 import BM25Okapi

from backend.rag.providers.base import VectorStoreProvider

_RRF_K = 60


def _reciprocal_rank_fusion(
    dense_results: list[dict], bm25_scores: list[float]
) -> list[dict]:
    """Merge dense and BM25 ranked lists using Reciprocal Rank Fusion."""
    dense_ranks = {r["content"]: i for i, r in enumerate(dense_results)}
    bm25_order = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)
    bm25_ranks = {dense_results[i]["content"]: rank for rank, i in enumerate(bm25_order)}

    fused: dict[str, float] = {}
    for content in dense_ranks:
        fused[content] = 1 / (_RRF_K + dense_ranks[content]) + 1 / (
            _RRF_K + bm25_ranks.get(content, len(dense_results))
        )

    content_map = {r["content"]: r for r in dense_results}
    return [
        {**content_map[c], "score": fused[c]}
        for c in sorted(fused, key=fused.__getitem__, reverse=True)
        if c in content_map
    ]


class HybridRetriever:
    """Combines BM25 keyword search and dense semantic search via Reciprocal Rank Fusion."""

    def __init__(self, provider: VectorStoreProvider) -> None:
        self._provider = provider

    def retrieve(self, query: str, n_results: int = 5) -> list[dict]:
        """Return top-n results fused from BM25 and dense similarity search."""
        candidates = self._provider.similarity_search(query, top_k=n_results * 2)
        if not candidates:
            return []

        corpus = [r["content"].lower().split() for r in candidates]
        bm25_scores = BM25Okapi(corpus).get_scores(query.lower().split())
        return _reciprocal_rank_fusion(candidates, bm25_scores.tolist())[:n_results]

    def retrieve_formatted(self, query: str, n_results: int = 5) -> str:
        """Return fused results as a formatted string for prompt injection."""
        results = self.retrieve(query, n_results)
        if not results:
            return "No relevant documents found."
        return "\n\n".join(
            f"[RRF Score: {r['score']:.4f}] {r['content']}" for r in results
        )
