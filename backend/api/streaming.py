import json
from collections.abc import Generator

from backend.agents.orchestrator import Orchestrator
from backend.state.schema import AgentState


def _state_to_event(node_name: str, state_update: dict) -> str:
    """Serialize a node name and its state update into an SSE-formatted string."""
    payload = json.dumps({"node": node_name, "update": state_update})
    return f"data: {payload}\n\n"


def stream_run(orchestrator: Orchestrator, query: str) -> Generator[str, None, None]:
    """Yield SSE-formatted events for each agent node as the pipeline executes."""
    for event in orchestrator.stream(query):
        for node_name, state_update in event.items():
            serializable_update = {
                k: v
                for k, v in state_update.items()
                if k not in ("metrics",)
            }
            yield _state_to_event(node_name, serializable_update)
    yield "data: {\"node\": \"__end__\", \"update\": {}}\n\n"
