from dataclasses import dataclass, field
from datetime import datetime


_COST_RATES: dict[str, dict[str, float]] = {
    "claude-opus-4-7": {
        "input": 15.0,
        "output": 75.0,
        "cache_read": 1.50,
        "cache_write": 18.75,
    },
    "claude-sonnet-4-6": {
        "input": 3.0,
        "output": 15.0,
        "cache_read": 0.30,
        "cache_write": 3.75,
    },
    "claude-haiku-4-5-20251001": {
        "input": 0.80,
        "output": 4.0,
        "cache_read": 0.08,
        "cache_write": 1.0,
    },
}


@dataclass
class TokenUsage:
    """Token counts from a single Anthropic API response."""

    input_tokens: int
    output_tokens: int
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        """Sum of input and output tokens."""
        return self.input_tokens + self.output_tokens


@dataclass
class AgentMetric:
    """Performance and cost metrics for a single agent invocation."""

    agent_name: str
    run_id: str
    model: str
    token_usage: TokenUsage
    latency_ms: float
    tool_calls: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def cost_usd(self) -> float:
        """Calculated cost in USD based on model pricing."""
        rates = _COST_RATES.get(self.model, _COST_RATES["claude-sonnet-4-6"])
        return (
            self.token_usage.input_tokens * rates["input"]
            + self.token_usage.output_tokens * rates["output"]
            + self.token_usage.cache_read_tokens * rates["cache_read"]
            + self.token_usage.cache_write_tokens * rates["cache_write"]
        ) / 1_000_000

    @property
    def tokens_per_second(self) -> float:
        """Output tokens per second based on observed latency."""
        if self.latency_ms == 0:
            return 0.0
        return (self.token_usage.output_tokens / self.latency_ms) * 1000
