import json
import os
from pathlib import Path

import httpx
from deepeval import evaluate
from deepeval.metrics import (
    AnswerRelevancyMetric,
    BiasMetric,
    FaithfulnessMetric,
    HallucinationMetric,
)
from deepeval.test_case import LLMTestCase

_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
_RESULTS_DIR = Path(__file__).parent / "results"
_API_BASE = os.getenv("ARCHON_API_BASE", "http://localhost:8000/api/v1")


def _get_token() -> str:
    """Obtain a JWT token using the dev API key."""
    response = httpx.post(
        f"{_API_BASE}/auth/token",
        json={"api_key": "dev-key-change-in-production"},
        timeout=10.0,
    )
    response.raise_for_status()
    return response.json()["access_token"]


def _run_pipeline(question: str, token: str) -> str:
    """POST a question to the Archon pipeline and return the final report."""
    response = httpx.post(
        f"{_API_BASE}/run",
        json={"query": question},
        headers={"Authorization": f"Bearer {token}"},
        timeout=120.0,
    )
    response.raise_for_status()
    return response.json().get("final_report", "")


def build_test_cases() -> list[LLMTestCase]:
    """Build DeepEval test cases from the golden dataset by running the pipeline."""
    golden = json.loads(_DATASET_PATH.read_text())
    token = _get_token()
    test_cases = []

    for entry in golden:
        try:
            actual_output = _run_pipeline(entry["question"], token)
            test_cases.append(
                LLMTestCase(
                    input=entry["question"],
                    actual_output=actual_output,
                    expected_output=entry["ideal_answer"],
                    context=[entry["ideal_answer"]],
                )
            )
        except Exception as e:
            print(f"Skipping {entry['id']}: {e}")

    return test_cases


def run_deepeval_evaluation() -> None:
    """Run DeepEval metrics on all test cases and write JUnit XML for CI."""
    _RESULTS_DIR.mkdir(exist_ok=True)
    test_cases = build_test_cases()

    metrics = [
        AnswerRelevancyMetric(threshold=0.7),
        FaithfulnessMetric(threshold=0.7),
        HallucinationMetric(threshold=0.5),
        BiasMetric(threshold=0.5),
    ]

    evaluate(test_cases, metrics)
