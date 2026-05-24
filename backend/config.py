import os

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    anthropic_api_key: str
    tavily_api_key: str = ""
    redis_url: str = "redis://localhost:6379"
    database_url: str = "sqlite:///./runs.db"
    chroma_persist_dir: str = "./chroma_db"
    primary_model: str = "claude-sonnet-4-6"
    critic_model: str = "claude-opus-4-7"
    max_reflection_cycles: int = 3
    critique_pass_threshold: float = 7.0
    jwt_secret: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expiry_minutes: int = 60

    vector_store_provider: str = "local"
    pinecone_api_key: str = ""
    pinecone_index_name: str = "archon"
    weaviate_url: str = "localhost"
    use_hybrid_retrieval: bool = False

    langsmith_api_key: str = ""
    langsmith_project: str = "archon-ai"
    langchain_tracing_v2: bool = False

    guardrails_enabled: bool = False
    guardrails_redact_pii: bool = True
    guardrails_topic_allowlist: str = ""

    storage_backend: str = "sqlite"
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_region: str = "us-east-1"
    dynamo_runs_table: str = "archon_runs"
    dynamo_tenants_table: str = "archon_tenants"


settings = Settings()
