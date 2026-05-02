import httpx

from backend.config import settings


class WebSearchTool:
    """Searches the web via Tavily and returns formatted source snippets."""

    name = "web_search"
    description = (
        "Search the web for current information on a topic. "
        "Returns titles, URLs, and content snippets from top results."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query"},
            "max_results": {
                "type": "integer",
                "description": "Number of results to return",
                "default": 5,
            },
        },
        "required": ["query"],
    }

    _BASE_URL = "https://api.tavily.com/search"

    def run(self, query: str, max_results: int = 5) -> str:
        """Execute a Tavily web search and return formatted results."""
        response = httpx.post(
            self._BASE_URL,
            json={
                "api_key": settings.tavily_api_key,
                "query": query,
                "max_results": max_results,
            },
            timeout=10.0,
        )
        response.raise_for_status()
        results = response.json().get("results", [])
        return "\n\n".join(
            f"Source: {r['url']}\nTitle: {r['title']}\nContent: {r['content']}"
            for r in results
        )

    def to_anthropic_schema(self) -> dict:
        """Return the Anthropic-compatible tool definition."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }
