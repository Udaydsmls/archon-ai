import base64

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse

from backend.agents.orchestrator import Orchestrator
from backend.api.auth import create_access_token, get_current_user, validate_api_key
from backend.api.models import (
    BatchIngestRequest,
    BatchIngestResponse,
    MetricsSummary,
    QueryRequest,
    RunSummary,
    TokenRequest,
    TokenResponse,
)
from backend.api.streaming import stream_run
from backend.rag.batch_ingestor import BatchIngestor
from backend.rag.chunker import TextChunker
from backend.rag.providers.factory import get_vector_store_provider
from backend.safety.guards import GuardRailsValidator

router = APIRouter()
_orchestrator = Orchestrator()
_validator = GuardRailsValidator()


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
    validated_report, warnings = _validator.validate(final_state.get("final_report", ""))
    metrics = _orchestrator.get_metrics(final_state["run_id"])
    return RunSummary(
        run_id=final_state["run_id"],
        query=request.query,
        final_report=validated_report,
        reflection_cycles=final_state.get("reflection_cycles", 0),
        critique_score=final_state.get("critique_score", 0.0),
        metrics=metrics,
        safety_warnings=warnings,
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


@router.post("/run/multimodal", response_model=RunSummary)
async def run_multimodal(
    query: str = Form(...),
    files: list[UploadFile] = File(default=[]),
    _: str = Depends(get_current_user),
) -> RunSummary:
    """Accept a text query alongside image or PDF uploads and run the full pipeline."""
    uploaded_files = []
    for upload in files:
        content = await upload.read()
        uploaded_files.append(
            {
                "filename": upload.filename or "",
                "media_type": upload.content_type or "application/octet-stream",
                "base64_data": base64.b64encode(content).decode("utf-8"),
            }
        )

    final_state = _orchestrator.run(query, uploaded_files=uploaded_files)
    validated_report, warnings = _validator.validate(final_state.get("final_report", ""))
    metrics = _orchestrator.get_metrics(final_state["run_id"])
    return RunSummary(
        run_id=final_state["run_id"],
        query=query,
        final_report=validated_report,
        reflection_cycles=final_state.get("reflection_cycles", 0),
        critique_score=final_state.get("critique_score", 0.0),
        metrics=metrics,
        safety_warnings=warnings,
    )


@router.post("/ingest/batch", response_model=BatchIngestResponse)
def ingest_batch(
    request: BatchIngestRequest, _: str = Depends(get_current_user)
) -> BatchIngestResponse:
    """Submit a bulk document ingestion job using the Anthropic Batch API."""
    provider = get_vector_store_provider()
    ingestor = BatchIngestor(provider=provider, chunker=TextChunker())
    batch_id = ingestor.ingest_batch(urls=request.urls, file_paths=request.file_paths)
    return BatchIngestResponse(batch_id=batch_id)


@router.get("/run/{run_id}/metrics", response_model=MetricsSummary)
def get_run_metrics(
    run_id: str, _: str = Depends(get_current_user)
) -> MetricsSummary:
    """Return aggregated token, cost, and latency metrics for a completed run."""
    summary = _orchestrator.get_metrics(run_id)
    if not summary:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")
    return MetricsSummary(**summary)
