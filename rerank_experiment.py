"""Exercise 3.5 — measure Context Recall/Precision before and after reranking.

Offline: reads the saved retrieval traces, reorders the SAME chunks with
``rerank_by_overlap(chunks, question)`` and re-scores them. No API calls.
The reranker only sees the question (never the expected answer) to avoid leakage.

Run: python rerank_experiment.py
"""

from __future__ import annotations

import json
from pathlib import Path

from template import RAGASEvaluator, rerank_by_overlap

ROOT = Path(__file__).resolve().parent


def main() -> None:
    golden = json.loads((ROOT / "golden_dataset.json").read_text(encoding="utf-8"))
    actual = json.loads(
        (ROOT / "artifacts" / "actual_answers.json").read_text(encoding="utf-8")
    )
    expected_by_id = {p["id"]: p["expected_answer"] for p in golden["qa_pairs"]}
    evaluator = RAGASEvaluator()

    rows = []
    for record in actual["answers"]:
        chunks = [c["text"] for c in record["retrieved_contexts"]]
        reranked = rerank_by_overlap(chunks, record["question"])
        assert sorted(reranked) == sorted(chunks), "reranking must keep the same chunk set"
        expected = expected_by_id[record["id"]]
        rows.append((
            record["id"],
            evaluator.evaluate_context_recall(chunks, expected),
            evaluator.evaluate_context_recall(reranked, expected),
            evaluator.evaluate_context_precision(chunks, expected),
            evaluator.evaluate_context_precision(reranked, expected),
        ))

    print("| ID | Recall before | Recall after | Precision before | Precision after | Delta Precision |")
    print("|---|---:|---:|---:|---:|---:|")
    for rid, r0, r1, p0, p1 in rows:
        print(f"| {rid} | {r0:.3f} | {r1:.3f} | {p0:.3f} | {p1:.3f} | {p1 - p0:+.3f} |")
    n = len(rows)
    avg = [sum(row[i] for row in rows) / n for i in range(1, 5)]
    print(f"| **Avg ({n})** | {avg[0]:.3f} | {avg[1]:.3f} | {avg[2]:.3f} | {avg[3]:.3f} | {avg[3] - avg[2]:+.3f} |")


if __name__ == "__main__":
    main()
