from typing import Protocol, runtime_checkable


@runtime_checkable
class Tool(Protocol):
    """Interface that all tools must satisfy."""

    name: str
    description: str
    input_schema: dict

    def run(self, **kwargs) -> str:
        """Execute the tool and return a string result."""
        ...

    def to_anthropic_schema(self) -> dict:
        """Return an Anthropic-compatible tool definition dict."""
        ...
