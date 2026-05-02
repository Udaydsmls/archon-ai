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
    critic_model: str = "claude-haiku-4-5-20251001"
    max_reflection_cycles: int = 3
    critique_pass_threshold: float = 7.0
    jwt_secret: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expiry_minutes: int = 60


settings = Settings()
