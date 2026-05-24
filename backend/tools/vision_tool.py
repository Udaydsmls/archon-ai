import anthropic

from backend.config import settings


class VisionTool:
    """Extracts structured text from images and PDFs using Claude's vision capability."""

    name = "extract_from_file"
    description = (
        "Extract and describe information from an uploaded image or PDF. "
        "Returns a detailed text description of the content."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "base64_data": {"type": "string", "description": "Base64-encoded file content"},
            "media_type": {
                "type": "string",
                "description": "MIME type of the file (e.g. image/jpeg, application/pdf)",
            },
            "filename": {"type": "string", "description": "Original filename for context"},
        },
        "required": ["base64_data", "media_type"],
    }

    def __init__(self) -> None:
        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    def run(self, base64_data: str, media_type: str, filename: str = "") -> str:
        """Send the file to Claude as a vision content block and return extracted text."""
        if not base64_data:
            return ""

        source_type = "base64"
        content_type = media_type if media_type.startswith("image/") else "image/jpeg"

        response = self._client.messages.create(
            model=settings.primary_model,
            max_tokens=2048,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": source_type,
                                "media_type": content_type,
                                "data": base64_data,
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                f"Extract all text, data, and key information from this file"
                                f"{f' ({filename})' if filename else ''}. "
                                "Be thorough and structured."
                            ),
                        },
                    ],
                }
            ],
        )
        return next((b.text for b in response.content if hasattr(b, "text")), "")

    def to_anthropic_schema(self) -> dict:
        """Return the Anthropic-compatible tool definition."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }
