from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request body for initiating a research run."""

    query: str = Field(..., min_length=5, max_length=2000)


class RunSummary(BaseModel):
    """Top-level summary returned after a completed run."""

    run_id: str
    query: str
    final_report: str
    reflection_cycles: int
    critique_score: float
    metrics: dict
    safety_warnings: list[str] = []


class MetricsSummary(BaseModel):
    """Aggregated cost and performance metrics for a run."""

    run_id: str
    total_cost_usd: float
    total_tokens: int
    agents: dict


class TokenRequest(BaseModel):
    """Credentials for obtaining a JWT access token."""

    api_key: str


class TokenResponse(BaseModel):
    """JWT access token returned after successful authentication."""

    access_token: str
    token_type: str = "bearer"


class BatchIngestRequest(BaseModel):
    """Request body for bulk document ingestion via the Anthropic Batch API."""

    urls: list[str] = []
    file_paths: list[str] = []


class BatchIngestResponse(BaseModel):
    """Response containing the Anthropic batch job ID for status tracking."""

    batch_id: str
