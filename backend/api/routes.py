from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from backend.agents.orchestrator import Orchestrator
from backend.api.auth import create_access_token, get_current_user, validate_api_key
from backend.api.models import (
    MetricsSummary,
    QueryRequest,
    RunSummary,
    TokenRequest,
    TokenResponse,
)
from backend.api.streaming import stream_run

router = APIRouter()
_orchestrator = Orchestrator()


@router.post("/auth/token", response_model=TokenResponse)
def get_token(request: TokenRequest) -> TokenResponse:
    """Exchange a valid API key for a JWT access token."""
    if not validate_api_key(request.api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key"
        )
    return TokenResponse(access_token=create_access_token(request.api_key))


@router.post("/run", response_model=RunSummary)
def run_query(
    request: QueryRequest, _: str = Depends(get_current_user)
) -> RunSummary:
    """Execute the full multi-agent pipeline synchronously and return the result."""
    final_state = _orchestrator.run(request.query)
    metrics = _orchestrator.get_metrics(final_state["run_id"])
    return RunSummary(
        run_id=final_state["run_id"],
        query=request.query,
        final_report=final_state.get("final_report", ""),
        reflection_cycles=final_state.get("reflection_cycles", 0),
        critique_score=final_state.get("critique_score", 0.0),
        metrics=metrics,
    )


@router.post("/run/stream")
def stream_query(
    request: QueryRequest, _: str = Depends(get_current_user)
) -> StreamingResponse:
    """Stream agent step events as Server-Sent Events for real-time UI updates."""
    return StreamingResponse(
        stream_run(_orchestrator, request.query),
        media_type="text/event-stream",
    )


@router.get("/run/{run_id}/metrics", response_model=MetricsSummary)
def get_run_metrics(
    run_id: str, _: str = Depends(get_current_user)
) -> MetricsSummary:
    """Return aggregated token, cost, and latency metrics for a completed run."""
    summary = _orchestrator.get_metrics(run_id)
    if not summary:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")
    return MetricsSummary(**summary)
