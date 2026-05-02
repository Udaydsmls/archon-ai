import time

from backend.agents.base import BaseAgent
from backend.agents.prompts import get_prompt
from backend.metrics.tracker import MetricsTracker
from backend.rag.retriever import HybridRetriever
from backend.state.schema import AgentState


class RAGAgent(BaseAgent):
    """Retrieves relevant document context and extracts key information for the pipeline."""

    def __init__(self, tracker: MetricsTracker, retriever: HybridRetriever) -> None:
        super().__init__(tracker)
        self._retriever = retriever
        self._system_prompt = get_prompt("rag_system_v1")

    @property
    def name(self) -> str:
        return "rag_agent"

    def run(self, state: AgentState) -> dict:
        """Retrieve document context and synthesize it into structured findings."""
        raw_results = self._retriever.retrieve(state["query"], n_results=5)
        context_text = "\n\n".join(
            f"[Score: {r['score']:.2f}]\n{r['content']}" for r in raw_results
        )

        user_message = (
            f"Query: {state['query']}\n\n"
            f"Retrieved Context:\n{context_text}\n\n"
            "Extract and organize the most relevant information from these documents."
        )

        start = time.perf_counter()
        response = self._client.messages.create(
            model=self._model,
            max_tokens=2048,
            system=[
                {
                    "type": "text",
                    "text": self._system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user_message}],
        )
        latency = self._elapsed_ms(start)

        metric_entry = self._record_metric(state["run_id"], response.usage, latency)
        extracted_text = next(
            (b.text for b in response.content if hasattr(b, "text")), ""
        )

        return {
            "rag_context": [
                {"content": extracted_text, "source_chunks": len(raw_results)}
            ],
            "metrics": [metric_entry],
        }
