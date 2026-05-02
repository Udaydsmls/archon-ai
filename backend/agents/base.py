import time
from abc import ABC, abstractmethod

import anthropic

from backend.config import settings
from backend.metrics.models import AgentMetric, TokenUsage
from backend.metrics.tracker import MetricsTracker
from backend.state.schema import AgentState


class BaseAgent(ABC):
    """Abstract base for all agents. Handles Anthropic client setup and metric recording."""

    def __init__(self, tracker: MetricsTracker, model: str | None = None) -> None:
        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self._model = model or settings.primary_model
        self._tracker = tracker

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this agent used in metrics and tracing."""
        ...

    @abstractmethod
    def run(self, state: AgentState) -> dict:
        """Execute agent logic and return a partial AgentState update."""
        ...

    def _record_metric(
        self,
        run_id: str,
        usage: anthropic.types.Usage,
        latency_ms: float,
        tool_calls: int = 0,
    ) -> dict:
        """Build, record, and return a serialized metric for state storage."""
        metric = AgentMetric(
            agent_name=self.name,
            run_id=run_id,
            model=self._model,
            token_usage=TokenUsage(
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0),
                cache_write_tokens=getattr(usage, "cache_creation_input_tokens", 0),
            ),
            latency_ms=latency_ms,
            tool_calls=tool_calls,
        )
        self._tracker.record(metric)
        return self._tracker.to_state_entry(metric)

    def _execute_tool(self, tool_map: dict, tool_name: str, tool_input: dict) -> str:
        """Dispatch a tool call by name and return its string result."""
        tool = tool_map.get(tool_name)
        if tool is None:
            return f"Unknown tool: {tool_name}"
        try:
            return tool.run(**tool_input)
        except Exception as e:
            return f"Tool error: {e}"

    @staticmethod
    def _elapsed_ms(start: float) -> float:
        """Return milliseconds elapsed since a time.perf_counter() start point."""
        return (time.perf_counter() - start) * 1000
