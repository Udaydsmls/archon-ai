from collections import defaultdict

from backend.metrics.models import AgentMetric


class MetricsTracker:
    """Aggregates and stores metrics across all agent invocations in a run."""

    def __init__(self) -> None:
        self._metrics: list[AgentMetric] = []

    def record(self, metric: AgentMetric) -> None:
        """Append a metric from a completed agent invocation."""
        self._metrics.append(metric)

    def get_run_metrics(self, run_id: str) -> list[AgentMetric]:
        """Return all metrics recorded for a specific run."""
        return [m for m in self._metrics if m.run_id == run_id]

    def get_total_cost(self, run_id: str) -> float:
        """Return total USD cost across all agents for a run."""
        return sum(m.cost_usd for m in self.get_run_metrics(run_id))

    def get_total_tokens(self, run_id: str) -> int:
        """Return total token count across all agents for a run."""
        return sum(m.token_usage.total_tokens for m in self.get_run_metrics(run_id))

    def get_summary(self, run_id: str) -> dict:
        """Return a per-agent summary of cost, tokens, latency, and throughput."""
        metrics = self.get_run_metrics(run_id)
        by_agent: dict[str, list[AgentMetric]] = defaultdict(list)
        for m in metrics:
            by_agent[m.agent_name].append(m)

        return {
            "run_id": run_id,
            "total_cost_usd": self.get_total_cost(run_id),
            "total_tokens": self.get_total_tokens(run_id),
            "agents": {
                name: {
                    "calls": len(ms),
                    "total_tokens": sum(m.token_usage.total_tokens for m in ms),
                    "total_cost_usd": sum(m.cost_usd for m in ms),
                    "avg_latency_ms": sum(m.latency_ms for m in ms) / len(ms),
                    "avg_tokens_per_second": sum(m.tokens_per_second for m in ms) / len(ms),
                    "total_tool_calls": sum(m.tool_calls for m in ms),
                }
                for name, ms in by_agent.items()
            },
        }

    def to_state_entry(self, metric: AgentMetric) -> dict:
        """Serialize a metric to a JSON-safe dict for inclusion in agent state."""
        return {
            "agent_name": metric.agent_name,
            "run_id": metric.run_id,
            "model": metric.model,
            "input_tokens": metric.token_usage.input_tokens,
            "output_tokens": metric.token_usage.output_tokens,
            "cache_read_tokens": metric.token_usage.cache_read_tokens,
            "cache_write_tokens": metric.token_usage.cache_write_tokens,
            "latency_ms": metric.latency_ms,
            "cost_usd": metric.cost_usd,
            "tokens_per_second": metric.tokens_per_second,
            "tool_calls": metric.tool_calls,
            "timestamp": metric.timestamp.isoformat(),
        }
