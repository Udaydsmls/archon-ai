import argparse
import sys


def _run_ragas() -> bool:
    """Run RAGAS evaluation suite and return True if all scores meet thresholds."""
    from evaluation.ragas_eval import run_ragas_evaluation

    results = run_ragas_evaluation()
    scores = results.get("scores", {})
    thresholds = {
        "faithfulness": 0.7,
        "answer_relevancy": 0.7,
        "context_precision": 0.6,
        "context_recall": 0.6,
    }
    passed = True
    for metric, threshold in thresholds.items():
        score = scores.get(metric, 0.0)
        status = "PASS" if score >= threshold else "FAIL"
        print(f"  [{status}] {metric}: {score:.3f} (threshold: {threshold})")
        if score < threshold:
            passed = False
    return passed


def _run_deepeval() -> bool:
    """Run DeepEval suite. Returns True (failures surface via JUnit XML in CI)."""
    from evaluation.deepeval_eval import run_deepeval_evaluation

    run_deepeval_evaluation()
    return True


def main() -> None:
    """CLI entry point: python -m evaluation.run_eval --suite ragas|deepeval|all"""
    parser = argparse.ArgumentParser(description="Archon evaluation harness")
    parser.add_argument(
        "--suite",
        choices=["ragas", "deepeval", "all"],
        default="all",
        help="Which evaluation suite to run",
    )
    args = parser.parse_args()

    passed = True
    if args.suite in ("ragas", "all"):
        print("\n── RAGAS Evaluation ──")
        passed = _run_ragas() and passed

    if args.suite in ("deepeval", "all"):
        print("\n── DeepEval Evaluation ──")
        passed = _run_deepeval() and passed

    if not passed:
        print("\nEvaluation failed: one or more metrics below threshold.")
        sys.exit(1)

    print("\nAll evaluations passed.")


if __name__ == "__main__":
    main()
