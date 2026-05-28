"""
Formats and saves evaluation results to results/eval_<timestamp>.json
and prints a readable summary table to stdout.
"""
import json
import os
from datetime import datetime


def save_and_print(runner_results: list[dict], ragas_scores: dict) -> str:
    """Saves full results to JSON and prints summary. Returns the output file path."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"eval_{timestamp}.json")

    report = {
        "timestamp": timestamp,
        "num_cases": len(runner_results),
        "ragas_scores": ragas_scores,
        "avg_latency_seconds": round(
            sum(r["latency_seconds"] for r in runner_results) / len(runner_results), 2
        ),
        "per_question": [
            {
                "question": r["question"],
                "user_role": r["user_role"],
                "answer": r["answer"],
                "ground_truth": r["ground_truth"],
                "sources": r["sources"],
                "latency_seconds": r["latency_seconds"],
                "chunks_retrieved": len(r["contexts"]),
            }
            for r in runner_results
        ],
    }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Test cases:       {report['num_cases']}")
    print(f"Avg latency:      {report['avg_latency_seconds']}s")
    print()
    def fmt(v):
        return f"{v:.3f}" if isinstance(v, float) else str(v)

    print("RAGAS Scores (0.0 = worst, 1.0 = best):")
    print(f"  Faithfulness:      {fmt(ragas_scores.get('faithfulness', 'N/A'))}")
    print(f"  Answer Relevancy:  {fmt(ragas_scores.get('answer_relevancy', 'N/A'))}")
    print(f"  Context Precision: {fmt(ragas_scores.get('context_precision', 'N/A'))}")
    print(f"  Context Recall:    {fmt(ragas_scores.get('context_recall', 'N/A'))}")
    print()
    print("Per-question answers:")
    for r in runner_results:
        chunks = len(r["contexts"])
        print(f"  Q: {r['question'][:55]}")
        print(f"     Role={r['user_role']} | Chunks={chunks} | {r['latency_seconds']}s")
        print(f"     A: {r['answer'][:80]}...")
        print()
    print(f"Full report saved to: {output_path}")
    print("=" * 60)

    return output_path
