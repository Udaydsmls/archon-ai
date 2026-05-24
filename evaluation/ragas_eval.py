import json
import os
from datetime import datetime
from pathlib import Path

import httpx
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

_RESULTS_DIR = Path(__file__).parent / "results"
_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
_API_BASE = os.getenv("ARCHON_API_BASE", "http://localhost:8000/api/v1")
_API_TOKEN = os.getenv("ARCHON_API_TOKEN", "")


def _get_token() -> str:
    """Obtain a JWT token using the dev API key."""
    response = httpx.post(
        f"{_API_BASE}/auth/token",
        json={"api_key": "dev-key-change-in-production"},
        timeout=10.0,
    )
    response.raise_for_status()
    return response.json()["access_token"]


def _run_pipeline(question: str, token: str) -> dict:
    """POST a question to the Archon pipeline and return the full response."""
    response = httpx.post(
        f"{_API_BASE}/run",
        json={"query": question},
        headers={"Authorization": f"Bearer {token}"},
        timeout=120.0,
    )
    response.raise_for_status()
    return response.json()


def run_ragas_evaluation() -> dict:
    """Run all golden dataset questions through Archon and evaluate with RAGAS metrics."""
    golden = json.loads(_DATASET_PATH.read_text())
    token = _get_token()

    questions, answers, contexts, ground_truths = [], [], [], []

    for entry in golden:
        try:
            result = _run_pipeline(entry["question"], token)
            questions.append(entry["question"])
            answers.append(result.get("final_report", ""))
            contexts.append([result.get("final_report", "")])
            ground_truths.append(entry["ideal_answer"])
        except Exception as e:
            print(f"Skipping {entry['id']}: {e}")

    dataset = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )

    scores = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )

    results = {
        "timestamp": datetime.utcnow().isoformat(),
        "num_questions": len(questions),
        "scores": scores.to_pandas().mean().to_dict(),
    }

    _RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_path = _RESULTS_DIR / f"ragas_{timestamp}.json"
    output_path.write_text(json.dumps(results, indent=2))
    print(f"RAGAS results saved to {output_path}")
    return results
