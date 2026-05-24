import time

from backend.agents.base import BaseAgent
from backend.agents.prompts import get_prompt
from backend.metrics.tracker import MetricsTracker
from backend.state.schema import AgentState
from backend.tools.base import Tool
from backend.tools.url_scraper import URLScraperTool
from backend.tools.vision_tool import VisionTool
from backend.tools.web_search import WebSearchTool


class ResearchAgent(BaseAgent):
    """Gathers information on a query using a ReAct (Reason + Act) loop."""

    def __init__(self, tracker: MetricsTracker) -> None:
        super().__init__(tracker)
        tools = [WebSearchTool(), URLScraperTool(), VisionTool()]
        self._tool_map: dict[str, Tool] = {t.name: t for t in tools}
        self._tool_schemas = [t.to_anthropic_schema() for t in tools]
        self._system_prompt = get_prompt("research_system_v1")
        self._vision_tool = VisionTool()

    @property
    def name(self) -> str:
        return "research_agent"

    def _extract_uploaded_files(self, state: AgentState) -> str:
        """Run vision extraction on any uploaded files and return combined text."""
        uploaded = state.get("uploaded_files", [])
        if not uploaded:
            return ""
        extractions = []
        for file in uploaded:
            result = self._vision_tool.run(
                base64_data=file.get("base64_data", ""),
                media_type=file.get("media_type", "image/jpeg"),
                filename=file.get("filename", ""),
            )
            extractions.append(result)
        return "\n\n".join(extractions)

    def run(self, state: AgentState) -> dict:
        """Execute the ReAct loop and return research results as a state update."""
        file_context = self._extract_uploaded_files(state)
        initial_content = (
            f"{file_context}\n\nQuery: {state['query']}" if file_context else state["query"]
        )

        messages = [{"role": "user", "content": initial_content}]
        tool_calls_count = 0
        start = time.perf_counter()
        final_response = None

        while True:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=4096,
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
        final_text = next(
            (b.text for b in final_response.content if hasattr(b, "text")), ""
        )

        return {
            "research_results": [{"content": final_text, "tool_calls": tool_calls_count}],
            "metrics": [metric_entry],
        }
