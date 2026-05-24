from backend.config import settings


class SafetyConfig:
    """Reads and exposes guardrails configuration from application settings."""

    @property
    def enabled(self) -> bool:
        """Whether the safety validation layer is active."""
        return settings.guardrails_enabled

    @property
    def redact_pii(self) -> bool:
        """Whether detected PII should be redacted from output."""
        return settings.guardrails_redact_pii

    @property
    def topic_allowlist(self) -> list[str]:
        """Comma-separated topic allowlist parsed into a list, or empty for no restriction."""
        raw = settings.guardrails_topic_allowlist.strip()
        return [t.strip() for t in raw.split(",") if t.strip()] if raw else []


safety_config = SafetyConfig()
