import httpx
from bs4 import BeautifulSoup


class URLScraperTool:
    """Fetches a URL and extracts its main text content for fact-checking."""

    name = "scrape_url"
    description = (
        "Fetch and extract the main text content from a URL. "
        "Useful for verifying claims or reading source material in depth."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "The URL to scrape"},
        },
        "required": ["url"],
    }

    _MAX_CHARS = 5000
    _STRIP_TAGS = ["script", "style", "nav", "footer", "header"]

    def run(self, url: str) -> str:
        """Scrape a URL and return cleaned text content."""
        response = httpx.get(url, timeout=10.0, follow_redirects=True)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(self._STRIP_TAGS):
            tag.decompose()
        return soup.get_text(separator="\n", strip=True)[: self._MAX_CHARS]

    def to_anthropic_schema(self) -> dict:
        """Return the Anthropic-compatible tool definition."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }
