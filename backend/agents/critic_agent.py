import json
import time

from backend.agents.base import BaseAgent
from backend.agents.prompts import get_prompt
from backend.config import settings
from backend.metrics.tracker import MetricsTracker
from backend.state.schema import AgentState


class CriticAgent(BaseAgent):
    """Scores a draft report and provides structured improvement feedback."""

    def __init__(self, tracker: MetricsTracker) -> None:
        super().__init__(tracker, model=settings.critic_model)
        self._system_prompt = get_prompt("critic_system_v1")

    @property
    def name(self) -> str:
        return "critic_agent"

    def run(self, state: AgentState) -> dict:
        """Evaluate the current draft and return a score and feedback as a state update."""
        user_message = (
            f"Original Query: {state['query']}\n\n"
            f"Draft Report:\n{state['draft']}\n\n"
            "Evaluate this report and respond with the required JSON."
        )

        start = time.perf_counter()
        response = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=[
                {
                    "type": "text",
                    "text": self._system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user_message}],
        )
        latency = self._elapsed_ms(start)

        metric_entry = self._record_metric(state["run_id"], response.usage, latency)
        raw_text = next((b.text for b in response.content if hasattr(b, "text")), "{}")

        try:
            critique = json.loads(raw_text)
            score = float(critique.get("score", 0.0))
            feedback = critique.get("feedback", "")
        except (json.JSONDecodeError, ValueError):
            score = 0.0
            feedback = raw_text

        return {
            "critique_score": score,
            "critique_feedback": feedback,
            "reflection_cycles": state.get("reflection_cycles", 0) + 1,
            "metrics": [metric_entry],
        }
