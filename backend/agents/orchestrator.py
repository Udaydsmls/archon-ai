import uuid
from typing import Literal

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

from backend.agents.critic_agent import CriticAgent
from backend.agents.rag_agent import RAGAgent
from backend.agents.research_agent import ResearchAgent
from backend.agents.synthesis_agent import SynthesisAgent
from backend.config import settings
from backend.metrics.tracker import MetricsTracker
from backend.rag.hybrid_retriever import HybridRetriever
from backend.rag.providers.factory import get_vector_store_provider
from backend.rag.retriever import SemanticRetriever
from backend.state.schema import AgentState


def _route_after_critique(state: AgentState) -> Literal["synthesis_agent", "__end__"]:
    """Decide whether to revise the draft or finalize based on critique score."""
    at_limit = state.get("reflection_cycles", 0) >= settings.max_reflection_cycles
    passed = state.get("critique_score", 0.0) >= settings.critique_pass_threshold
    if passed or at_limit:
        return END
    return "synthesis_agent"


def _finalize(state: AgentState) -> dict:
    """Promote the current draft to the final report."""
    return {"final_report": state.get("draft", "")}


def _initialize_state(query: str, uploaded_files: list[dict] | None = None) -> AgentState:
    """Build the initial AgentState for a new run."""
    return AgentState(
        query=query,
        run_id=str(uuid.uuid4()),
        research_results=[],
        rag_context=[],
        draft="",
        critique_score=0.0,
        critique_feedback="",
        final_report="",
        reflection_cycles=0,
        metrics=[],
        uploaded_files=uploaded_files or [],
        error=None,
    )


class Orchestrator:
    """LangGraph-based orchestrator that coordinates all agents in the pipeline."""

    def __init__(self) -> None:
        self._tracker = MetricsTracker()
        self._graph = self._build_graph()

    def _build_graph(self):
        """Construct and compile the LangGraph StateGraph."""
        provider = get_vector_store_provider()
        retriever = (
            HybridRetriever(provider)
            if settings.use_hybrid_retrieval
            else SemanticRetriever(provider)
        )

        research_agent = ResearchAgent(self._tracker)
        rag_agent = RAGAgent(self._tracker, retriever)
        synthesis_agent = SynthesisAgent(self._tracker)
        critic_agent = CriticAgent(self._tracker)

        graph = StateGraph(AgentState)
        graph.add_node("research_agent", research_agent.run)
        graph.add_node("rag_agent", rag_agent.run)
        graph.add_node("synthesis_agent", synthesis_agent.run)
        graph.add_node("critic_agent", critic_agent.run)
        graph.add_node("finalize", _finalize)

        graph.set_entry_point("research_agent")
        graph.add_edge("research_agent", "rag_agent")
        graph.add_edge("rag_agent", "synthesis_agent")
        graph.add_edge("synthesis_agent", "critic_agent")
        graph.add_conditional_edges(
            "critic_agent",
            _route_after_critique,
            {"synthesis_agent": "synthesis_agent", END: "finalize"},
        )
        graph.add_edge("finalize", END)

        checkpointer = SqliteSaver.from_conn_string(settings.database_url)
        return graph.compile(checkpointer=checkpointer)

    def run(self, query: str, uploaded_files: list[dict] | None = None) -> AgentState:
        """Execute the full pipeline for a query and return the final state."""
        initial_state = _initialize_state(query, uploaded_files)
        config = {"configurable": {"thread_id": initial_state["run_id"]}}
        return self._graph.invoke(initial_state, config=config)

    def stream(self, query: str, uploaded_files: list[dict] | None = None):
        """Stream state updates node-by-node for real-time consumption."""
        initial_state = _initialize_state(query, uploaded_files)
        config = {"configurable": {"thread_id": initial_state["run_id"]}}
        yield from self._graph.stream(initial_state, config=config)

    def get_metrics(self, run_id: str) -> dict:
        """Return the aggregated metrics summary for a completed run."""
        return self._tracker.get_summary(run_id)
