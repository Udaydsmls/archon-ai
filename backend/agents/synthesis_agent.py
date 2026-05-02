import time

from backend.agents.base import BaseAgent
from backend.agents.prompts import get_prompt
from backend.metrics.tracker import MetricsTracker
from backend.state.schema import AgentState
from backend.tools.base import Tool
from backend.tools.code_executor import CodeExecutorTool


class SynthesisAgent(BaseAgent):
    """Combines research and RAG context into a structured report, incorporating critique."""

    def __init__(self, tracker: MetricsTracker) -> None:
        super().__init__(tracker)
        tools = [CodeExecutorTool()]
        self._tool_map: dict[str, Tool] = {t.name: t for t in tools}
        self._tool_schemas = [t.to_anthropic_schema() for t in tools]
        self._system_prompt = get_prompt("synthesis_system_v1")

    @property
    def name(self) -> str:
        return "synthesis_agent"

    def run(self, state: AgentState) -> dict:
        """Produce a draft report from all accumulated context and prior critique."""
        research_text = "\n\n".join(
            r.get("content", "") for r in state.get("research_results", [])
        )
        rag_text = "\n\n".join(
            r.get("content", "") for r in state.get("rag_context", [])
        )

        user_message = (
            f"Query: {state['query']}\n\n"
            f"Research Findings:\n{research_text}\n\n"
            f"Document Context:\n{rag_text}"
        )

        if state.get("critique_feedback"):
            user_message += f"\n\nCritique Feedback to Address:\n{state['critique_feedback']}"

        messages = [{"role": "user", "content": user_message}]
        tool_calls_count = 0
        start = time.perf_counter()
        final_response = None

        while True:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=8096,
                system=[
                    {
                        "type": "text",
                        "text": self._system_prompt,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                tools=self._tool_schemas,
                messages=messages,
            )

            if response.stop_reason == "end_turn":
                final_response = response
                break

            if response.stop_reason == "tool_use":
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        result = self._execute_tool(self._tool_map, block.name, block.input)
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": result,
                            }
                        )
                        tool_calls_count += 1

                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": tool_results})

        latency = self._elapsed_ms(start)
        metric_entry = self._record_metric(
            state["run_id"], final_response.usage, latency, tool_calls_count
        )
        draft = next(
            (b.text for b in final_response.content if hasattr(b, "text")), ""
        )

        return {
            "draft": draft,
            "metrics": [metric_entry],
        }
