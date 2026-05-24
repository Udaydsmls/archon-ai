import operator
from typing import Annotated, TypedDict


class AgentState(TypedDict):
    """Shared state flowing through the LangGraph agent pipeline."""

    query: str
    run_id: str
    research_results: list[dict]
    rag_context: list[dict]
    draft: str
    critique_score: float
    critique_feedback: str
    final_report: str
    reflection_cycles: int
    metrics: Annotated[list[dict], operator.add]
    uploaded_files: list[dict]
    error: str | None
