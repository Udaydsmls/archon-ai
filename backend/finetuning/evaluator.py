import json

import anthropic

from backend.config import settings


class CriticEvaluator:
    """Compares fine-tuned and baseline critic model outputs on a held-out test set."""

    def __init__(self, fine_tuned_model: str) -> None:
        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self._fine_tuned_model = fine_tuned_model
        self._baseline_model = settings.critic_model

    def _score_draft(self, model: str, draft: str, query: str) -> dict:
        """Run critic inference on a draft and parse the JSON response."""
        response = self._client.messages.create(
            model=model,
            max_tokens=512,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Query: {query}\nDraft: {draft}\n"
                        "Evaluate and return JSON: {\"score\": <float>, \"feedback\": \"<str>\"}"
                    ),
                }
            ],
        )
        raw = next((b.text for b in response.content if hasattr(b, "text")), "{}")
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"score": 0.0, "feedback": raw}

    def evaluate(self, test_samples: list[dict]) -> dict:
        """Return accuracy comparison between fine-tuned and baseline models."""
        fine_tuned_errors = []
        baseline_errors = []

        for sample in test_samples:
            true_score = sample["score"]
            ft_result = self._score_draft(self._fine_tuned_model, sample["draft"], "")
            bl_result = self._score_draft(self._baseline_model, sample["draft"], "")
            fine_tuned_errors.append(abs(ft_result["score"] - true_score))
            baseline_errors.append(abs(bl_result["score"] - true_score))

        return {
            "fine_tuned_mae": sum(fine_tuned_errors) / len(fine_tuned_errors),
            "baseline_mae": sum(baseline_errors) / len(baseline_errors),
            "samples_evaluated": len(test_samples),
        }
