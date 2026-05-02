import json
import random

import anthropic

from backend.config import settings

_DRAFT_QUALITY_LEVELS = ["poor", "average", "good", "excellent"]

_GENERATION_PROMPT = """
Generate a synthetic research draft about "{topic}" with {quality} quality.
Then evaluate it as the critic agent would. Respond in JSON:
{{
  "draft": "<the draft text>",
  "score": <float 0-10>,
  "feedback": "<critique feedback>"
}}
"""


class CriticTrainingDataGenerator:
    """Generates synthetic (draft, score, feedback) pairs for fine-tuning the critic model."""

    def __init__(self) -> None:
        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    def generate_sample(self, topic: str) -> dict | None:
        """Generate a single labeled training example for a given topic."""
        quality = random.choice(_DRAFT_QUALITY_LEVELS)
        response = self._client.messages.create(
            model=settings.primary_model,
            max_tokens=2048,
            messages=[
                {
                    "role": "user",
                    "content": _GENERATION_PROMPT.format(topic=topic, quality=quality),
                }
            ],
        )
        raw = next((b.text for b in response.content if hasattr(b, "text")), "")
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    def generate_dataset(self, topics: list[str], samples_per_topic: int = 5) -> list[dict]:
        """Generate a labeled dataset across multiple topics."""
        dataset = []
        for topic in topics:
            for _ in range(samples_per_topic):
                sample = self.generate_sample(topic)
                if sample:
                    dataset.append(sample)
        return dataset

    def save_dataset(self, dataset: list[dict], output_path: str) -> None:
        """Write the dataset to a JSONL file for fine-tuning."""
        with open(output_path, "w") as f:
            for item in dataset:
                f.write(json.dumps(item) + "\n")
