import json
import sys
from pathlib import Path

import agent
from rag.evaluation.metrics import ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank

SOURCE = "lec 1 (1).pdf"
CASES = (
    ("Which parts make up a microprocessor?", 1),
    ("How many addresses can a 16-bit address bus generate?", 1),
    ("How many bits are the 8086 registers?", 1),
    ("What is the maximum number of locations for a 16-bit offset?", 1),
    ("Which processor handles arithmetic, logic and control activities?", 2),
    ("How large is a word in bytes?", 2),
    ("What internal storage speeds up processor operations?", 2),
    ("Which section declares initialized data or constants?", 2),
    ("How many words are needed to store hexadecimal 0534?", 4),
    ("How many bits represent a word address in a one megabyte memory?", 4),
    ("How many opcode bits are in a 16-bit instruction with a 12-bit address bus?", 5),
)
UNRELATED_QUERIES = (
    "How does photosynthesis work in plants?",
    "What is the capital city of Japan?",
    "How to bake a chocolate cake?",
    "Explain the water cycle and evaporation.",
    "Who won the football world cup?",
)
TOP_K = 4
MIN_RECALL = 0.9


def main() -> int:
    indexed = agent.collection.get(
        where={"source": SOURCE},
        include=["metadatas"],
    )
    available_chunks = {
        metadata.get("chunk")
        for metadata in indexed["metadatas"] or []
        if metadata
    }
    missing_chunks = sorted(
        {expected for _, expected in CASES} - available_chunks
    )
    if missing_chunks:
        print(
            f"Benchmark needs {SOURCE} with chunks {missing_chunks}; "
            "re-index that document or update evaluate_retrieval.py."
        )
        return 2

    hits = 0
    reciprocal_rank = 0.0
    precisions = []
    ndcgs = []
    case_results = []
    for query, expected_chunk in CASES:
        results = agent.retrieve_docs(query, limit=TOP_K)
        retrieved_ids = [
            f"{result['source']}:{result['chunk']}" for result in results
        ]
        relevant_ids = {f"{SOURCE}:{expected_chunk}"}
        rank = next(
            (
                index
                for index, result in enumerate(results, start=1)
                if result["source"] == SOURCE and result["chunk"] == expected_chunk
            ),
            None,
        )
        if rank:
            hits += 1
            reciprocal_rank += 1 / rank
        precisions.append(precision_at_k(relevant_ids, retrieved_ids, TOP_K))
        ndcgs.append(ndcg_at_k(relevant_ids, retrieved_ids, TOP_K))
        case_results.append({
            "query": query,
            "expected": f"{SOURCE}:{expected_chunk}",
            "rank": rank,
        })
        print(
            f"{'HIT' if rank else 'MISS'} "
            f"expected_chunk={expected_chunk} rank={rank or '-'} "
            f"query={query}"
        )

    false_positives = []
    for query in UNRELATED_QUERIES:
        results = agent.retrieve_docs(query, limit=TOP_K)
        if results:
            false_positives.append(query)
        print(
            f"{'FALSE_POSITIVE' if results else 'NO_MATCH'} query={query}"
        )

    recall = hits / len(CASES)
    mrr = reciprocal_rank / len(CASES)
    print(
        f"Recall@{TOP_K}: {hits}/{len(CASES)} ({recall:.0%}); "
        f"MRR@{TOP_K}: {mrr:.3f}; "
        f"unrelated matches: {len(false_positives)}/{len(UNRELATED_QUERIES)}"
    )
    report = {
        "top_k": TOP_K,
        "recall_at_k": recall,
        "mrr_at_k": mrr,
        "mean_precision_at_k": sum(precisions) / len(precisions) if precisions else 0.0,
        "mean_ndcg_at_k": sum(ndcgs) / len(ndcgs) if ndcgs else 0.0,
        "cases": case_results,
        "false_positives": false_positives,
        "automated": True,
        "manual_verification": False,
    }
    report_dir = Path("evaluation/reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "latest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return 0 if recall >= MIN_RECALL and not false_positives else 1


if __name__ == "__main__":
    sys.exit(main())
